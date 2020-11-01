# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:51:35 2020

@author: Shaodong
"""
import os
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.InteractiveSession(config=config)
K.set_session(sess)
from model import Unet, optimizer, train, loss
import time, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy import genfromtxt
import random
import gc
import pickle
import shutil

tf.random.set_seed(1234)
random.seed(1234)
tf.keras.backend.set_floatx('float32')


INPUT_HEIGHT = 512 
INPUT_WIDTH = 512 
OUTPUT_HEIGHT = 512 
OUTPUT_WIDTH = 512 
BATCH_SIZE = 4
NUM_EPOCH = 700

in_path = '../data/objects_cad_animal_sun_081920/'
out_path = '../results/objects_cad_animal_sun_101920/'
checkpoint_dir = '../data/tmp/trained_model/'+\
    'objects_cad_animal_sun_101920_checkpoints/'

"""
Load data from digital twin for neural network training and testing.
args:
    in_path: str
        path of the dataset, e.g. in_path = '../data/objects_cad_animal_sun_081820/'
return:
    train_ds: tf dataset with batch
        train_ds = (X_train, y_train)
        X_train: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_train: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_train[:,:,:,0] is the ground truth of depth (z).
        y_train[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
        Note train_ds will be shuffled every training epoch.
    test_ds: tf dataset with batch
        test_ds = (X_test, y_test)
        X_test: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_test: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_test[:,:,:,0] is the ground truth of depth (z).
        y_test[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    image_idx_test: 1D numpy array
        Image index/name of the testing data, which could be used to 
        index the depth plot.
"""
def load_digital_data(in_path):
    X_lst = []
    y_lst = []
    mask_lst = []
    image_idx_lst = []
    # load data to lists
    for i in os.listdir(in_path):
        if not os.path.exists(os.path.join(in_path, i, 'z.csv')):
            continue
        x_loc = os.path.join(in_path, i, '2.png')
        y_loc = os.path.join(in_path, i, 'z.csv')
        mask_loc = os.path.join(in_path, i, 'mask.csv')
        tmp_x = imageio.imread(x_loc).astype('float32')[1:-1, 16:-16]
        tmp_y = genfromtxt(y_loc, delimiter=',', dtype='float32')[1:-1, 16:-16]
        tmp_mask = genfromtxt(mask_loc, delimiter=',', dtype='float32')[1:-1, 16:-16]
        if np.isnan(tmp_x).sum()==0 and np.isnan(tmp_y).sum()==0:
            X_lst.append(tmp_x)
            y_lst.append(tmp_y)
            mask_lst.append(tmp_mask)
            image_idx_lst.append(i)
            
    # transfer loaded data in lists to numpy array
    X = np.array(X_lst).astype('float32')
    y = np.array(y_lst).astype('float32')
    mask = np.array(mask_lst).astype('float32')
    del X_lst, y_lst, mask_lst
    
    # Add a channels dimension to fit the input of the neural network
    X = X[..., tf.newaxis]
    y = y[..., tf.newaxis]
    mask = mask[..., tf.newaxis]
    # y[:,:,:,1] is defined mask
    y = np.concatenate((y,mask),axis=3)
    
    TOTAL_NUM_SAMPLE = int(X.shape[0])
    TRAIN_NUM_SAMPLE = int(X.shape[0] * 0.7)
    VALIDATE_NUM_SAMPLE = int(X.shape[0] * 0.15)
    TEST_NUM_SAMPLE = TOTAL_NUM_SAMPLE - TRAIN_NUM_SAMPLE - VALIDATE_NUM_SAMPLE
    print(' Total number of samples:', TOTAL_NUM_SAMPLE,
        '\n Number of training samples', TRAIN_NUM_SAMPLE,
        '\n Number of validating samples', VALIDATE_NUM_SAMPLE,
        '\n Number of testing samples', TEST_NUM_SAMPLE)
    
    # randomly split the loaded dataset to training data and testing data
    train_selected = random.sample(range(TOTAL_NUM_SAMPLE),TRAIN_NUM_SAMPLE)
    not_train_selected = [i for i in range(TOTAL_NUM_SAMPLE) if i not in train_selected]
    validate_selected = random.sample(not_train_selected, VALIDATE_NUM_SAMPLE)
    test_selected = [i for i in not_train_selected if i not in validate_selected]
    
    X_train = X[train_selected]
    y_train = y[train_selected]
    X_validate = X[validate_selected]
    y_validate = y[validate_selected]
    X_test = X[test_selected]
    y_test = y[test_selected]
    #X_test = np.delete(X,train_selected,0)
    #y_test = np.delete(y,train_selected,0)
    #image_idx_test = np.delete(np.array(image_idx_lst),train_selected,0)
    image_idx_validate = np.array(image_idx_lst)[validate_selected]
    image_idx_test = np.array(image_idx_lst)[test_selected]
    del X,y
    
    # load data in numpy to tf dataset with batch as their first dimension
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
        ).shuffle(TRAIN_NUM_SAMPLE,reshuffle_each_iteration=True).batch(BATCH_SIZE)
    validate_ds = tf.data.Dataset.from_tensor_slices(
        (X_validate, y_validate)
        ).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
        ).batch(BATCH_SIZE)
    
    del X_train,y_train
    del X_test, y_test
    gc.collect()
    
    # save test data for evaluation
    #with open('../data/tmp/digital_test.pickle', 'wb') as f:
    #    pickle.dump((test_ds,image_idx_test),f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return train_ds, validate_ds, test_ds, image_idx_validate, image_idx_test

"""
Load data from SSFPP for neural network training and testing.
(SFFPP: Single-Shot 3D Shape Reconstruction Data Sets. Available online: https://figshare.com/articles/Single-
Shot_Fringe_Projection_Dataset/7636697)
args:
    in_path: str
        path of the dataset, e.g. in_path = '../data/objects_cad_animal_sun_081820/'
return:
    train_ds: tf dataset with batch
        train_ds = (X_train, y_train)
        X_train: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_train: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_train[:,:,:,0] is the ground truth of depth (z).
        y_train[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
        Note train_ds will be shuffled every training epoch.
    test_ds: tf dataset with batch
        test_ds = (X_test, y_test)
        X_test: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_test: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_test[:,:,:,0] is the ground truth of depth (z).
        y_test[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    image_idx_test: 1D numpy array
        Image index/name of the testing data, which could be used to 
        index the depth plot.
"""
def load_SSFPP_data(in_path):
    X_train = np.load(in_path+'X_train_1.npy').astype('float32')
    y_train = np.load(in_path+'Z_train.npy').astype('float32')
    X_test = np.load(in_path+'X_test_1.npy').astype('float32')
    y_test = np.load(in_path+'Z_test.npy').astype('float32')
    mask_train = (y_train > 0) * 1
    mask_test = (y_test > 0) * 1
    y_train = np.concatenate((y_train, mask_train), axis = 3).astype('float32')
    y_test = np.concatenate((y_test, mask_test), axis = 3).astype('float32')

    TRAIN_NUM_SAMPLE = X_train.shape[0]
    TEST_NUM_SAMPLE = int(X_test.shape[0] * 0.5)
    VALIDATE_NUM_SAMPLE = X_test.shape[0] - TEST_NUM_SAMPLE
    X_validate = X_test[0:VALIDATE_NUM_SAMPLE]
    y_validate = y_test[0:VALIDATE_NUM_SAMPLE]
    X_test = X_test[VALIDATE_NUM_SAMPLE:]
    y_test = y_test[VALIDATE_NUM_SAMPLE:]
    image_idx_test = range(VALIDATE_NUM_SAMPLE, X_test.shape[0])
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
        ).shuffle(TRAIN_NUM_SAMPLE,reshuffle_each_iteration=True).batch(BATCH_SIZE)
    validate_ds = tf.data.Dataset.from_tensor_slices(
        (X_validate, y_validate)
        ).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
        ).batch(BATCH_SIZE)
    del X_train,y_train
    del X_test, y_test
    gc.collect()
    return train_ds, validate_ds, test_ds, image_idx_test

"""
Make predicted depth plot for digital twin testing data and restore them in the out_path.
args:
    in_path: str
        path of the loaded (testing) data
    out_path: str
        output path of the plots
    image_idx_test: 1D numpy array
        Image index/name of the testing data, which could be used to 
        index the depth plot.
    test_ds: tf Dataset
        test data with batch.
        test_ds = (X_test, y_test)
        X_test: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_test: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_test[:,:,:,0] is the ground truth of depth (z).
        y_test[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    epoch: int32
        epoch of training
    model: tf Model
        Neural network model to make prediction of depth map.
"""
def plot_digital_test_output(in_path, out_path, image_idx_test, test_ds, model, epoch):
    if not os.path.exists(out_path+str(epoch)):
        os.mkdir(out_path+str(epoch))
    test_ds_b1 = test_ds.unbatch().batch(1)
    for (x, y),image_idx0 in zip(test_ds_b1,image_idx_test):
        dst = out_path+str(epoch)+'/'+str(image_idx0)
        if not os.path.exists(dst):
            shutil.copytree(in_path+str(image_idx0),dst)
        pred_test = model(x,training=False)
        image0 = x[0,:,:,0].numpy()
        truth0 = y[0,:,:,0].numpy()
        mask0 = y[0,:,:,1].numpy()
        pred0 = pred_test[0,:,:,0].numpy()
        np.savetxt(dst+'/z_prediction.csv',pred0,delimiter=",")
        pred0[~mask0.astype(bool)] = np.nan
        truth0[~mask0.astype(bool)] = np.nan
        max_v = max(np.nanmax(truth0), np.nanmax(pred0))
        min_v = min(np.nanmin(truth0), np.nanmin(pred0))
        
        plt.imshow(image0);plt.colorbar();plt.title("input")
        plt.savefig(dst+'/input.png');plt.close()
        plt.imshow(truth0,vmax=max_v,vmin=min_v);plt.colorbar();plt.title("truth")
        plt.savefig(dst+'/truth.png');plt.close()
        plt.imshow(pred0,vmax=max_v,vmin=min_v);plt.colorbar();plt.title("prediction")
        plt.savefig(dst+'/prediction.png');plt.close()
        truth0 = truth0[~np.isnan(truth0)]
        pred0 = pred0[~np.isnan(pred0)]
        plt.hist(truth0.flatten(),30);plt.title("truth");plt.xlim([min_v,max_v])
        plt.savefig(dst+'/truth_hist.png');plt.close()
        plt.hist(pred0.flatten(), 30);plt.title("prediction");plt.xlim([min_v,max_v])
        plt.savefig(dst+'/prediction_hist.png');plt.close()
    return

"""
Make predicted depth plot for SFFPP testing data and restore them in the out_path.
(SFFPP: Single-Shot 3D Shape Reconstruction Data Sets. Available online: 
 https://figshare.com/articles/Single-Shot_Fringe_Projection_Dataset/7636697)
args:
    out_path: str
        output path of the plots
    image_idx_test: 1D numpy array
        Image index/name of the testing data, which could be used to 
        index the depth plot.
    test_ds: tf Dataset
        test data with batch.
        test_ds = (X_test, y_test)
        X_test: [BATCH, INPUT HEIGHT, INPUT WIDTH, 1]
        y_test: [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]
        y_test[:,:,:,0] is the ground truth of depth (z).
        y_test[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    epoch: int32
        epoch of training
    model: tf Model
        Neural network model to make prediction of depth map.
"""
def plot_SSFPP_test_output(out_path, test_ds, image_idx_test, model, epoch):
    if not os.path.exists(out_path+str(epoch)):
        os.mkdir(out_path+str(epoch))
    test_ds_b1 = test_ds.unbatch().batch(1)
    for (x, y),image_idx0 in zip(test_ds_b1,image_idx_test):
        dst = out_path+str(epoch)+'/'+str(image_idx0)
        if not os.path.exists(dst):
            os.mkdir(out_path+str(epoch)+'/'+str(image_idx0))
        pred_test = model(x,training=False)
        image0 = x[0,:,:,0].numpy()
        truth0 = y[0,:,:,0].numpy()
        mask0 = y[0,:,:,1].numpy()
        pred0 = pred_test[0,:,:,0].numpy()
        np.savetxt(dst+'/z_prediction.csv',pred0,delimiter=",")
        np.savetxt(dst+'/z_truth.csv', truth0,delimiter=",")
        np.savetxt(dst+'/mask.csv',mask0,delimiter=",")
        pred0[~mask0.astype(bool)] = np.nan
        truth0[~mask0.astype(bool)] = np.nan
        max_v = max(np.nanmax(truth0), np.nanmax(pred0))
        min_v = min(np.nanmin(truth0), np.nanmin(pred0))
        
        plt.imshow(image0);plt.colorbar();plt.title("input")
        plt.savefig(dst+'/input.png');plt.close()
        plt.imshow(truth0,vmax=max_v,vmin=min_v);plt.colorbar();plt.title("truth")
        plt.savefig(dst+'/truth.png');plt.close()
        plt.imshow(pred0,vmax=max_v,vmin=min_v);plt.colorbar();plt.title("prediction")
        plt.savefig(dst+'/prediction.png');plt.close()
        truth0 = truth0[~np.isnan(truth0)]
        pred0 = pred0[~np.isnan(pred0)]
        plt.hist(truth0.flatten(),30);plt.title("truth");plt.xlim([min_v,max_v])
        plt.savefig(dst+'/truth_hist.png');plt.close()
        plt.hist(pred0.flatten(), 30);plt.title("prediction");plt.xlim([min_v,max_v])
        plt.savefig(dst+'/prediction_hist.png');plt.close()

if __name__ == "__main__":
    # load data
    if not 'SS_FPP_CNN' in in_path:
        train_ds, validate_ds, test_ds, image_idx_validate, image_idx_test = load_digital_data(in_path)
    else:
        train_ds, validate_ds, test_ds, image_idx_test = load_SSFPP_data(in_path)
    
    # create an instance of neural network model
    unet = Unet()
    
    # Create checkpoint for the training of the neural network model.
    # Neural network models including trained parameters will be stored 
    # in the checkpoint_dir.
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=unet)
    
    # create a csv file to record training loss
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(out_path+'training_loss.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['epoch', 'train_loss', 'vali_loss',
                             'time_used', 'learning_rate','iteration'])
    
    # start training from here
    for epoch in range(NUM_EPOCH):
        start = time.time()
        train_loss = []
        for x, y in train_ds:
            current_train_loss = train(unet, x, y, optimizer, loss)
            train_loss.append(current_train_loss)
        val_loss = []
        for x, y in validate_ds:
            pred_val = unet(x,training=False)
            current_val_loss = loss(y, pred_val)
            val_loss.append(current_val_loss)
        end = time.time()
        template = 'Epoch {}, Train Loss: {:.6f}, Vali Loss: {:.6f}, Time used: {}s, Learning rate: {:.6f}, iteration: {}'
        print(template.format(epoch + 1,
                              np.mean(train_loss),
                              np.mean(val_loss),
                              np.round(end-start),
                              optimizer._decayed_lr(var_dtype=tf.float32),
                              optimizer.iterations.numpy()))
        with open(out_path+'training_loss.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([epoch + 1, np.mean(train_loss),
                              np.mean(val_loss),
                              np.round(end-start),
                              optimizer._decayed_lr(var_dtype=tf.float32).numpy(),
                              optimizer.iterations.numpy()])
        if epoch % 50 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        if epoch % 50 == 0 and not epoch == 0:
            if not 'SS_FPP_CNN' in in_path:
                plot_digital_test_output(in_path, out_path, image_idx_validate, validate_ds, unet, epoch)
                
            else:
                plot_SSFPP_test_output(out_path, test_ds, image_idx_test, unet, epoch)
    
    test_loss = []
    for x, y in test_ds:
        pred_test = unet(x,training=False)
        current_test_loss = loss(y, pred_test)
        test_loss.append(current_test_loss)
    print('test loss:', np.mean(test_loss))
    
    
    
    
    
    
    
    
    