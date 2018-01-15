#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from time import time
import os
import numpy as np
from scipy import misc
import keras
from keras import backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend as KTF
from keras import layers, models
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from utils import batch_generator, model_helper
from utils import data_preprocess as pre


def run_cnn(X_train, X_val, X_test,
            network_func,
            n_classes=30,
            img_shape=(128, 128, 3),
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            callbacks=None,
            batch_size=32,
            steps_per_epoch=64,
            num_epochs=300,
            validation_steps=50,
            path_save_weights=None,
            path_plot_model=None,
            dropout=False):

    train_iter = batch_generator.Batch_generator(X_train, image_shape=img_shape,
                                                 num_classes=n_classes,
                                                 batch_size=batch_size, segment=False)
    val_iter = batch_generator.Batch_generator(X_val, image_shape=img_shape,
                                               num_classes=n_classes,
                                               batch_size=batch_size, segment=False)

    old_session = KTF.get_session()

    # with tf.Graph().as_default():
    session = tf.Session()
    KTF.set_session(session)

    # if using dropout (a situation when networks would be different between training and testing phase)
    if dropout:
        KTF.set_learning_phase(1)

    inputs = layers.Input(img_shape)
    output_layer = network_func(inputs, n_classes)
    model = models.Model(inputs, output_layer)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    history = model.fit_generator(train_iter,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=num_epochs,
                                    verbose=1,
                                    validation_data=val_iter,
                                    validation_steps=validation_steps,
                                    callbacks=callbacks,
                                    workers=2)

    # save model
    # model.save('./test_log/models/fcn_model.hdf5')
    if path_save_weights != None:
        try:
            model.save_weights(path_save_weights)
        except:
            print("Could not save weigths data.")

    if path_plot_model != None:
        try:
            plot_model(model, to_file=path_plot_model)
        except:
            print("Could not plot model properly.")

    # test the prediction
    print("\n")
    print("Testing model...")
    score = model_helper.predict_from_path(model, X_test, img_shape, n_classes)
    print("\n")
    print("====="*13)
    print("\n")
    print("Test score    : {:.6f}".format(score[0]))
    print("Test accuracy : {:.6f}\n".format(score[1]))

    KTF.set_session(old_session)


def run_segmentation(X_train, y_train, X_val, y_val, X_test, y_test,
                     network_func,
                     n_classes=30,
                     img_shape=(128, 128, 3),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     callbacks=None,
                     batch_size=32,
                     steps_per_epoch=500,
                     num_epochs=30,
                     validation_steps=65,
                     path_save_weights=None,
                     path_save_pred_images=None,
                     num_save_pred_image=None,
                     dropout=False):

#     import tensorflow as tf
#     from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    train_iter = batch_generator.Batch_generator(X_train, target=y_train, image_shape=img_shape,
                                                 num_classes=n_classes,
                                                 batch_size=batch_size, segment=True)
    val_iter = batch_generator.Batch_generator(X_val, target=y_val, image_shape=img_shape,
                                               num_classes=n_classes,
                                               batch_size=batch_size, segment=True)

    old_session = KTF.get_session()

    # with tf.Graph().as_default():
    session = tf.Session(config=config)
    KTF.set_session(session)

    # if using dropout (a situation when networks would be different between training and testing phase)
    if dropout:
        KTF.set_learning_phase(1)

    inputs = layers.Input(img_shape)
    output_layer = network_func(inputs, n_classes)
    model = models.Model(inputs, output_layer)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    model.fit_generator(train_iter,
                        steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=val_iter,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        workers=2)

    # save model
    # model.save('./test_log/models/fcn_model.hdf5')
    if path_save_weights != None:
        try:
            model.save_weights(path_save_weights)
        except:
            print("Path to save weigths data is not valid.")

    # predicting and calculate IoU
    print("\n")
    print("====="*15)
    print("\n")
    print("Testing model...")
    start = time()
    # extract data
    test = np.vstack([np.expand_dims(misc.imresize(misc.imread(t), img_shape), axis=0) for t in X_test])

    pred = model_helper.prediction(model, test) # predicted data
    target = pre.pixelwise_class_array(y_test) # ground truths

    iou = model_helper.ave_iou_score(pred, target)
    end = time()
    print("\n")
    print("IoU score    : {:.6f}".format(iou))
    print("Calcuration time : {:.6f} sec.".format(end-start))

    # Save predicted image
    if path_save_pred_images != None:
        print("\n")
        print("====="*15)
        print("\n")
        print("Saving predict image...")

        path_save_pred_images = os.path.join(path_save_pred_images, 'predictions')

        # create directory
        pre.makedirs_if_none(path_save_pred_images)

        # reduce save data if need
        if num_save_pred_image != None:
            pred = pred[:num_save_pred_image]
            X_test = X_test[:num_save_pred_image]

        # convert from class array to image(rgb) array
        pred = pre.pixelwise_array_to_img(pred)

        # Save data
        for img, file_path in zip(pred, X_test):
            misc.imsave(os.path.join(path_save_pred_images, os.path.basename(file_path)), img)

        print("Done.")

    # close current session
    KTF.set_session(old_session)
