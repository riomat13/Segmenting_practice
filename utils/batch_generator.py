#!/usr/bin/env python

import numpy as np
import keras
from keras.preprocessing.image import Iterator

from keras import backend as K


class Batch_generator(Iterator):
    """Generating dataset batch for .fit_generator"""
    def __init__(self, train, target=None, num_classes=None, batch_size=32, shuffle=True, seed=None):
        assert len(train)>0, "Training data is empty."
        assert len(target)>0, "Target data is empty."
        assert len(train)==len(target), "Numbers of input and target data are not matched."

        self.train = train
        self.image_shape = train[0].shape
        self.target = target
        self.num_classes = num_classes

        self.data_tuples = list(zip(train, target))
        self.n = len(self.data_tuples)

        super(Batch_generator, self).__init__(self.n, batch_size, shuffle, seed)


    def next(self):
        """generating image batch"""
        # Use threading.Lock() to make this thread safe
        # more detail about .lock and .index_generator, check code at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/_impl/keras/preprocessing/image.py
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # Initialize batch form
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size,) + self.image_shape[:2] + (self.num_classes,), dtype=K.floatx())

        for n, i in enumerate(index_array):
            x, y = self.data_tuples[i]

            batch_x[n] = x
            batch_y[n] = y

        return batch_x, batch_y
