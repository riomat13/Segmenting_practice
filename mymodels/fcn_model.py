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


def fcn_sep_simple(inputs, num_classes):
    """Fully Convlutional Network 5 layers"""
    num_classes += 1 # add background
    # encoding
    conv1 = encode_separableconv2D_batchnorm_simple(inputs,
                                                    filters=16,
                                                    kernel_size=(3,3),
                                                    strides=(1,1),
                                                    padding="same",
                                                    activation='relu',
                                                    conv_num=2,
                                                    pool_size=2,
                                                    name='encode1')

    conv2 = encode_separableconv2D_batchnorm_simple(conv1,
                                                    filters=32,
                                                    kernel_size=(3,3),
                                                    strides=(1,1),
                                                    padding="same",
                                                    activation='relu',
                                                    conv_num=2,
                                                    pool_size=2,
                                                    name='encode2')

    conv3 = encode_separableconv2D_batchnorm_simple(conv2,
                                                    filters=64,
                                                    kernel_size=(3,3),
                                                    strides=(1,1),
                                                    padding="same",
                                                    activation='relu',
                                                    conv_num=2,
                                                    pool_size=2,
                                                    name='encode3')

    conv4 = encode_separableconv2D_batchnorm_simple(conv3,
                                                    filters=128,
                                                    kernel_size=(3,3),
                                                    strides=(1,1),
                                                    padding="same",
                                                    activation='relu',
                                                    conv_num=2,
                                                    pool_size=2,
                                                    name='encode4')

    conv5 = encode_separableconv2D_batchnorm_simple(conv4,
                                                    filters=256,
                                                    kernel_size=(3,3),
                                                    strides=(1,1),
                                                    padding="same",
                                                    activation='relu',
                                                    conv_num=2,
                                                    pool_size=2,
                                                    name='encode5')

    # 1x1 Convolution
    x = Conv2D(1024, kernel_size=1, strides=1, activation='relu', name='fcn1')(conv5)
    x = Conv2D(1024, kernel_size=1, strides=1, activation='relu', name='fcn2')(x)

    # decoding
    x = decode_conv_batch2D(x,
                            filters=128,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv4,
                            name='decode1')

    x = decode_conv_batch2D(x,
                            filters=64,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv3,
                            name='decode2')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv2,
                            name='decode3')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=conv1,
                            name='decode4')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(5,5),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=2,
                            concat=inputs,
                            name='decode5')

    output = Conv2D(num_classes, kernel_size=1, strides=1,
                    padding='same', activation='softmax',
                    name='output')(x)

    return output


def fcn_simple(inputs, num_classes):
    """Fully Convlutional Network 5 layers"""
    num_classes += 1 # add background
    # encoding
    conv1 = encode_conv2D_batchnorm_simple(inputs,
                                           filters=16,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode1')

    conv2 = encode_conv2D_batchnorm_simple(conv1,
                                           filters=32,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode2')

    conv3 = encode_conv2D_batchnorm_simple(conv2,
                                           filters=64,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode3')

    conv4 = encode_conv2D_batchnorm_simple(conv3,
                                           filters=128,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode4')

    conv5 = encode_conv2D_batchnorm_simple(conv4,
                                           filters=256,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           activation='relu',
                                           conv_num=1,
                                           pool_size=2,
                                           name='encode5')

    # 1x1 Convolution
    x = Conv2D(1024, kernel_size=1, strides=1, activation='relu', name='fcn1')(conv5)
    x = Conv2D(1024, kernel_size=1, strides=1, activation='relu', name='fcn2')(x)

    # decoding
    x = decode_conv_batch2D(x,
                            filters=128,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=1,
                            concat=conv4,
                            name='decode1')

    x = decode_conv_batch2D(x,
                            filters=64,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=1,
                            concat=conv3,
                            name='decode2')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=1,
                            concat=conv2,
                            name='decode3')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=1,
                            concat=conv1,
                            name='decode4')

    x = decode_conv_batch2D(x,
                            filters=32,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            activation="relu",
                            conv_num=1,
                            concat=inputs,
                            name='decode5')

    output = Conv2D(num_classes, kernel_size=1, strides=1,
                    padding='same', activation='softmax',
                    name='output')(x)

    return output
