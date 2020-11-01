# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:47:01 2020

@author: Shaodong
"""

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras import Model


tf.keras.backend.set_floatx('float32')

"""
Unet neural network
"""
class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1_1 = Conv2D(32, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn1_1 = BatchNormalization()
        self.relu1_1 = LeakyReLU()
        self.conv1_2 = Conv2D(32, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn1_2 = BatchNormalization()
        self.relu1_2 = LeakyReLU()
        self.maxpool1 = MaxPool2D(pool_size= 2, strides=2,padding='same')
        self.drop1 = Dropout(.1)
        
        self.conv2_1 = Conv2D(64, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn2_1 = BatchNormalization()
        self.relu2_1 = LeakyReLU()
        self.conv2_2 = Conv2D(64, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn2_2 = BatchNormalization()
        self.relu2_2 = LeakyReLU()
        self.maxpool2 = MaxPool2D(pool_size= 2, strides=2,padding='same')
        self.drop2 = Dropout(.1)

        self.conv3_1 = Conv2D(128, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn3_1 = BatchNormalization()
        self.relu3_1 = LeakyReLU()
        self.conv3_2 = Conv2D(128, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn3_2 = BatchNormalization()
        self.relu3_2 = LeakyReLU()
        self.maxpool3 = MaxPool2D(pool_size= 2, strides=2,padding='same')
        self.drop3 = Dropout(.1)
        
        self.conv4_1 = Conv2D(256, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn4_1 = BatchNormalization()
        self.relu4_1 = LeakyReLU()
        self.conv4_2 = Conv2D(256, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn4_2 = BatchNormalization()
        self.relu4_2 = LeakyReLU()
        self.maxpool4 = MaxPool2D(pool_size= 2, strides=2,padding='same')
        self.drop4 = Dropout(.1)
        
        self.conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn5_1 = BatchNormalization()
        self.relu5_1 = LeakyReLU()
        self.conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn5_2 = BatchNormalization()
        self.relu5_2 = LeakyReLU()
        self.drop5 = Dropout(.1)
        
        self.up6 = Conv2DTranspose(filters=256,kernel_size=5, strides=2,padding='same')
        self.conv6_1 = Conv2D(256, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn6_1 = BatchNormalization()
        self.relu6_1 = LeakyReLU()
        self.merge6 = Concatenate(axis = 3) # [drop4, conv6_1]
        self.conv6_2 = Conv2D(256, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn6_2 = BatchNormalization()
        self.relu6_2 = LeakyReLU()
        self.conv6_3 = Conv2D(256, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn6_3 = BatchNormalization()
        self.relu6_3 = LeakyReLU()
        self.drop6 = Dropout(.1)
        
        self.up7 = Conv2DTranspose(filters=128,kernel_size=5, strides=2,padding='same')
        self.conv7_1 = Conv2D(128, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn7_1 = BatchNormalization()
        self.relu7_1 = LeakyReLU()
        self.merge7 = Concatenate(axis = 3) # [conv3_2, conv7_1]
        self.conv7_2 = Conv2D(128, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn7_2 = BatchNormalization()
        self.relu7_2 = LeakyReLU()
        self.conv7_3 = Conv2D(128, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn7_3 = BatchNormalization()
        self.relu7_3 = LeakyReLU()
        self.drop7 = Dropout(.1)
        
        self.up8 = Conv2DTranspose(filters=64,kernel_size=5, strides=2,padding='same')
        self.conv8_1 = Conv2D(64, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn8_1 = BatchNormalization()
        self.relu8_1 = LeakyReLU()
        self.merge8 = Concatenate(axis = 3) # [conv2_2, conv8_1]
        self.conv8_2 = Conv2D(64, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn8_2 = BatchNormalization()
        self.relu8_2 = LeakyReLU()
        self.conv8_3 = Conv2D(64, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn8_3 = BatchNormalization()
        self.relu8_3 = LeakyReLU()
        self.drop8 = Dropout(.1)
        
        self.up9 = Conv2DTranspose(filters=32,kernel_size=5, strides=2,padding='same')
        self.conv9_1 = Conv2D(32, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn9_1 = BatchNormalization()
        self.relu9_1 = LeakyReLU()
        self.merge9 = Concatenate(axis = 3) # [conv1_2, conv9_1]
        self.conv9_2 = Conv2D(32, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn9_2 = BatchNormalization()
        self.relu9_2 = LeakyReLU()
        self.conv9_3 = Conv2D(32, 5, padding='same', kernel_initializer='glorot_normal')
        self.bn9_3 = BatchNormalization()
        self.relu9_3 = LeakyReLU()
        self.drop9 = Dropout(.1)
                
        self.conv10 = Conv2D(1, 5, padding='same', activation='linear',kernel_initializer='glorot_normal')

    def call(self, x, training = False):
        conv1_1_out = self.conv1_1(x)
        conv1_1_out = self.bn1_1(conv1_1_out)
        conv1_1_out = self.relu1_1(conv1_1_out)
        conv1_2_out = self.conv1_2(conv1_1_out)
        conv1_2_out = self.bn1_2(conv1_2_out)
        conv1_2_out = self.relu1_2(conv1_2_out)
        maxpool1_out = self.maxpool1(conv1_2_out)
        maxpool1_out = self.drop1(maxpool1_out, training = training)
        
        conv2_1_out = self.conv2_1(maxpool1_out)
        conv2_1_out = self.bn2_1(conv2_1_out)
        conv2_1_out = self.relu2_1(conv2_1_out)
        conv2_2_out = self.conv2_2(conv2_1_out)
        conv2_2_out = self.bn2_2(conv2_2_out)
        conv2_2_out = self.relu2_2(conv2_2_out)
        maxpool2_out = self.maxpool2(conv2_2_out)
        maxpool2_out = self.drop2(maxpool2_out, training = training)
        
        conv3_1_out = self.conv3_1(maxpool2_out)
        conv3_1_out = self.bn3_1(conv3_1_out)
        conv3_1_out = self.relu3_1(conv3_1_out)
        conv3_2_out = self.conv3_2(conv3_1_out)
        conv3_2_out = self.bn3_2(conv3_2_out)
        conv3_2_out = self.relu3_2(conv3_2_out)
        maxpool3_out = self.maxpool3(conv3_2_out)
        maxpool3_out = self.drop3(maxpool3_out, training = training)
        
        conv4_1_out = self.conv4_1(maxpool3_out)
        conv4_1_out = self.bn4_1(conv4_1_out)
        conv4_1_out = self.relu4_1(conv4_1_out)
        conv4_2_out = self.conv4_2(conv4_1_out)
        conv4_2_out = self.bn4_2(conv4_2_out)
        conv4_2_out = self.relu4_2(conv4_2_out)
        maxpool4_out = self.maxpool4(conv4_2_out)
        maxpool4_out = self.drop4(maxpool4_out, training = training)
        
        conv5_1_out = self.conv5_1(maxpool4_out)
        conv5_1_out = self.bn5_1(conv5_1_out)
        conv5_1_out = self.relu5_1(conv5_1_out)
        conv5_2_out = self.conv5_2(conv5_1_out)
        conv5_2_out = self.bn5_2(conv5_2_out)
        conv5_2_out = self.relu5_2(conv5_2_out)
        drop5_out = self.drop5(conv5_2_out, training = training)
        
        up6_out = self.up6(drop5_out)
        conv6_1_out = self.conv6_1(up6_out)
        conv6_1_out = self.bn6_1(conv6_1_out)
        conv6_1_out = self.relu6_1(conv6_1_out)
        merge6_out = self.merge6([conv4_2_out, conv6_1_out])
        conv6_2_out = self.conv6_2(merge6_out)
        conv6_2_out = self.bn6_2(conv6_2_out)
        conv6_2_out = self.relu6_2(conv6_2_out)
        conv6_3_out = self.conv6_3(conv6_2_out)
        conv6_3_out = self.bn6_3(conv6_3_out)
        conv6_3_out = self.relu6_3(conv6_3_out)
        conv6_3_out = self.drop6(conv6_3_out, training = training)
        
        up7_out = self.up7(conv6_3_out)
        conv7_1_out = self.conv7_1(up7_out)
        conv7_1_out = self.bn7_1(conv7_1_out)
        conv7_1_out = self.relu7_1(conv7_1_out)
        merge7_out = self.merge7([conv3_2_out, conv7_1_out])
        conv7_2_out = self.conv7_2(merge7_out)
        conv7_2_out = self.bn7_2(conv7_2_out)
        conv7_2_out = self.relu7_2(conv7_2_out)
        conv7_3_out = self.conv7_3(conv7_2_out)
        conv7_3_out = self.bn7_3(conv7_3_out)
        conv7_3_out = self.relu7_3(conv7_3_out)
        conv7_3_out = self.drop7(conv7_3_out, training = training)
        
        up8_out = self.up8(conv7_3_out)
        conv8_1_out = self.conv8_1(up8_out)
        conv8_1_out = self.bn8_1(conv8_1_out)
        conv8_1_out = self.relu8_1(conv8_1_out)
        merge8_out = self.merge8([conv2_2_out, conv8_1_out])
        conv8_2_out = self.conv8_2(merge8_out)
        conv8_2_out = self.bn8_2(conv8_2_out)
        conv8_2_out = self.relu8_2(conv8_2_out)
        conv8_3_out = self.conv8_3(conv8_2_out)
        conv8_3_out = self.bn8_3(conv8_3_out)
        conv8_3_out = self.relu8_3(conv8_3_out)
        conv8_3_out = self.drop8(conv8_3_out, training = training)
        
        up9_out = self.up9(conv8_3_out)
        conv9_1_out = self.conv9_1(up9_out)
        conv9_1_out = self.bn9_1(conv9_1_out)
        conv9_1_out = self.relu9_1(conv9_1_out)
        merge9_out = self.merge9([conv1_2_out, conv9_1_out])
        conv9_2_out = self.conv9_2(merge9_out)
        conv9_2_out = self.bn9_2(conv9_2_out)
        conv9_2_out = self.relu9_2(conv9_2_out)
        conv9_3_out = self.conv9_3(conv9_2_out)
        conv9_3_out = self.bn9_3(conv9_3_out)
        conv9_3_out = self.relu9_3(conv9_3_out)
        conv9_3_out = self.drop9(conv9_3_out, training = training)
        
        conv10_out = self.conv10(conv9_3_out)
        return conv10_out
    
"""
Root mean squared error that ignore the background area (mask)
args:
    target_y: tf tensor
        [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]. 
        target_y[:,:,:,0] is the ground truth of depth (z).
        target_y[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    predicted_y: tf tensor
        [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 1].
        Outputs of the neural network / predicted depth (z) of the objects.
return:
    rmse: float, 
        Root mean squared error between gound truth and predicted depth of the objects.
"""
@tf.function
def loss(target_y, predicted_y):
    mask = target_y[:,:,:,1] # mask == 0 means background/invalid region
    truth = target_y[:,:,:,0]
    pred = predicted_y[:,:,:,0]
    num = tf.math.multiply(mask, tf.math.squared_difference(truth, pred))
    squared_loss = tf.math.divide(tf.math.reduce_sum(num), tf.math.reduce_sum(mask))
    rmse = tf.sqrt(squared_loss)
    return rmse

"""
Conduct one step of training the neural network.
args:
    model: 
        neural network model (tf)
    x: tf tensor
        [BATCH, INPUT HEIGHT, INPUT WIDTH, 1].
        Fringe image of objects.
    y: tf tensor
        [BATCH, OUTPUT HEIGHT, OUTPUT WIDTH, 2]. 
        y[:,:,:,0] is the ground truth of depth (z).
        y[:,:,:,1] is the mask of background. 'Mask == 0' indicates background
    optimizer: 
        tf optimizer, e.g. RMSprop, Adam
return:
    current loss: float
        Root mean square error of the current training step
"""
@tf.function
def train(model, x, y, optimizer, loss):
    with tf.GradientTape() as tape:
        predicted_y = model(x,training=True)
        current_loss = loss(y, predicted_y)
        gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss

"""
Define the optimizer for the neural network, which can be RMSprop, Adam, etc.
"""
step = tf.Variable(0, trainable=False)
schedule_lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [15000,25000,35000], [ 1e-3, 1e-4, 1e-5,1e-6])
optimizer = tf.keras.optimizers.RMSprop(schedule_lr)
#optimizer = tf.keras.optimizers.Adam(schedule_lr)

