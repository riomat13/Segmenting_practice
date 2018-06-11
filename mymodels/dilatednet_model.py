#!/usr/bin/env python

from __future__ import absolute_import

import numpy as np
import keras
from keras import layers
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, \
                         Flatten, concatenate, Dense,\
                         SeparableConv2D, BatchNormalization
                         # LeakyReLU, GlobalAveragePooling2D, Cropping2D,\
                         # merge, Reshape, Deconv2D, ZeroPadding2D

from utils.layers import conv2D_batchnorm_simple, separableconv2D_batchnorm_simple, \
                         encode_separableconv2D_batchnorm_simple, \
                         encode_conv2D_batchnorm_simple, decode_conv_batch2D

def dilated_net(inputs, num_classes):
    """Dilated Convlutional Network 5 layers"""
    num_classes += 1 # add background
    # encoding
    dil1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                  padding="same", 
                  dilation_rate=(2,2), 
                  activation='relu')(inputs)
    dil1a = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                  padding="same", 
                  dilation_rate=(4,4), 
                  activation='relu')(inputs)
    dil1 = concatenate([dil1, dil1a])
    
    conv1 = clayers.encode_conv2D_batchnorm_simple(inputs,
                                           filters=32,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode1')

    dil2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
              padding="same", 
              dilation_rate=(2,2), 
              activation='relu')(conv1)
    dil2a = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                  padding="same", 
                  dilation_rate=(4,4), 
                  activation='relu')(conv1)
    dil2 = concatenate([dil2, dil2a])

    conv2 = clayers.encode_conv2D_batchnorm_simple(conv1,
                                           filters=32,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode2')

    dil3 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
              padding="same", 
              dilation_rate=(2,2), 
              activation='relu')(conv2)
    dil3a = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                  padding="same", 
                  dilation_rate=(4,4), 
                  activation='relu')(conv2)
    dil3 = concatenate([dil3, dil3a])
    conv3 = clayers.encode_conv2D_batchnorm_simple(conv2,
                                           filters=32,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode3')
    
    dil4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
              padding="same", 
              activation='relu')(conv3)
    dil4a = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
              padding="same", 
              dilation_rate=(2,2), 
              activation='relu')(conv3)
    dil4b = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                  padding="same", 
                  dilation_rate=(4,4), 
                  activation='relu')(conv3)
    x = concatenate([dil4, dil4a, dil4b])
    

    # 1x1 Convolution
    x = Conv2D(256, kernel_size=1, strides=1, activation='relu', name='fcn1')(x)
#     x = Conv2D(1024, kernel_size=1, strides=1, activation='relu', name='fcn2')(x)

    # decoding
    x = clayers.decode_conv_batch2D(x,
                            filters=64,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv2,
                            concat2=dil3,
                            name='decode3')

    x = clayers.decode_conv_batch2D(x,
                            filters=64,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv1,
                            concat2=dil2,
                            name='decode4')

    x = clayers.decode_conv_batch2D(x,
                            filters=64,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=inputs,
                            concat2=dil1,
                            name='decode5')

    output = Conv2D(num_classes, kernel_size=1, strides=1,
                    padding='same', activation='softmax',
                    name='output')(x)

    return output
