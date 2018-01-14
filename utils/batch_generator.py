#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy import misc
import os
import keras
from keras.preprocessing.image import Iterator
from keras import backend as K

from . import data_preprocess as pre


class Batch_generator(Iterator):
    """Generating dataset batch for .fit_generator"""
    def __init__(self, train, target=None,
                 image_shape=(128,128,3),
                 num_classes=30,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 segment=False):

        assert len(train)>0, "Input image can not be found."

        # if segmenting, need mask images, else need just input images
        if segment:
            assert len(target)>0, "Target data can not be found."
            assert len(train)==len(target), "Input image sizes are not matched."

            from utils import helper
            self.color_dict = helper.class_color_dict()

            self.data_tuples = list(zip(train, target))
            self.n = len(self.data_tuples)

        else:
            self.data = train
            self.n = len(self.data)

        # if segmentation, add background class
        add = 1 if segment else 0
        self.num_classes = num_classes + add

        self.segment = segment
        self.image_shape = image_shape

        super(Batch_generator, self).__init__(self.n, batch_size, shuffle, seed)


    def next(self):
        """generating image batch"""
        # Use threading.Lock() to make this thread safe
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # Initialize batch form
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        if self.segment:
            batch_y = np.zeros((current_batch_size,) + self.image_shape[:2] + (1,), dtype=K.floatx())
        else:
            batch_y = np.zeros((current_batch_size, self.num_classes), dtype=K.floatx())

        for n, i in enumerate(index_array):
            # for segmentation
            if self.segment:
                x, y = self.data_tuples[i]
                class_lists = list(map(int, os.path.basename(x).split('_')[:-1])) # exract class data from file name
                # read image and resize if required
                x = misc.imread(x)
                y = misc.imread(y)
                if x.shape != self.image_shape:
                    x = misc.imresize(x, self.image_shape)
                    y = misc.imresize(y, self.image_shape)

                batch_x[n] = x

                # color-class dict
                color_dict = self.color_dict
                # create pixelwisely classified array
                for n_class in class_lists:
                    # if match with class number, 1(True), else 0(False)
                    batch_y[n] += np.expand_dims((y==color_dict[n_class]).all(axis=-1), axis=-1) * n_class

            # for simple classification
            else:
                x = self.data[i]
                n_class = int(os.path.basename(x)[:2])-1 # number starts from 0
                x = misc.imread(x)
                if x.shape != self.image_shape:
                    x = misc.imresize(x, self.image_shape)
                batch_x[n] = x
                batch_y[n][n_class] = 1

        batch_x[n] = x/255.0
        # transform to categorical array
        if self.segment:
            batch_y = pre.to_categorical_matrix(batch_y, self.num_classes)

        return batch_x, batch_y
