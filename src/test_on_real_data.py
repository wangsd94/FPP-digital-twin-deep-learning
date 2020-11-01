# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:41:06 2020

@author: Shaodong
"""

from model import Unet, optimizer, train, loss
import tensorflow as tf
import os,time, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy import genfromtxt
import random
import gc
import pickle
import shutil

in_path = '../data/real_data/'
out_path = '../results/real_data/'
checkpoint_dir = '../data/tmp/trained_model/'+\
    'UNET_cad_animal_sun_081920_checkpoints/'

"""
Plot predicted depth map from trained neural network model on real data
args:
    out_path: str
        path of the plots/outputs
    object_names: list
        a list of object names of real data
    model: str
        a trained neural network model, e.g. unet.
"""
def plot_real_data(out_path, object_names, real_ds, model):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for x,obj_name0 in zip(real_ds, object_names):
        print(obj_name0)
        dst = out_path + '/' +obj_name0
        if not os.path.exists(dst):
            shutil.copytree(in_path +obj_name0, dst)
        pred_test = model(x,training=False)
        image0 = x[0,:,:,0].numpy()
        pred0 = pred_test[0,:,:,0].numpy()
        np.savetxt(dst+'/z_prediction.csv',pred0,delimiter=",")
        max_v = pred0.max()
        min_v = pred0.min()
        
        plt.imshow(image0);plt.colorbar();plt.title("input")
        plt.savefig(dst+'/input.png');plt.close()
        plt.imshow(pred0,vmax=max_v,vmin=min_v);plt.colorbar();plt.title("prediction")
        plt.savefig(dst+'/prediction.png');plt.close()
        plt.hist(pred0.flatten(), 30);plt.title("prediction");plt.xlim([min_v,max_v])
        plt.savefig(dst+'/prediction_hist.png');plt.close()

"""
Load data from experiments.
args:
    in_path: str
        path of the real data
return:
    real_ds: tf Dataset
        tf Dataset of real data with #batch = 1
    object_names: list
        a list of object name of real data
"""
def load_real_data(in_path):
    X_lst = []
    y_lst = []
    object_names = []
    
    for i in os.listdir(in_path):
        x_loc = os.path.join(in_path, i, '2.png')
        tmp_x = imageio.imread(x_loc)
        #tmp_x = np.transpose(tmp_x)
        if np.isnan(tmp_x).sum()==0:
            X_lst.append(tmp_x)
            object_names.append(i)
        
    X = np.array(X_lst).astype(float)[:,1:-1,16:-16]
    #X = np.array(X_lst).astype(float)[:,16:-16,1:-1]
    y = np.array(y_lst).astype(float)
    del X_lst, y_lst
    # Add a channels dimension
    X = X[..., tf.newaxis]
    y = y[..., tf.newaxis]
    real_ds = tf.data.Dataset.from_tensor_slices(X).batch(1)
    return real_ds, object_names

if __name__ == '__main__':
    # create an instance of neural network model
    unet = Unet()
    
    # create checkpoint instance to load neural network model
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=unet)
    # load model
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    # load real data
    real_ds, object_names = load_real_data(in_path)
    
    # make inferences/predictions on real data
    plot_real_data(out_path, object_names, real_ds, unet)
    
    