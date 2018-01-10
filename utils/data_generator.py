#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import random
import os
from glob import glob
from scipy import misc
from utils import data_preprocess, helper
from PIL import Image as pil_image



def _makedirs_if_none(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# TODO: add pos_var for shift images
def _image_concat(data, num_images, in_shape):
    """ Image concatenate to one image
    data have to be (image, mask) dataset which has num_images length"""
    init_image_size = 400

    if len(data) != num_images:
        print("Image size is not correct.")
        return None

    # initial parameter
    new_init_shape = init_image_size*3 # concatenated image shape before resize, 400 is initial image size
    new_img = pil_image.new('RGB', (new_init_shape, new_init_shape), (0, 0, 0)) # new palette to paste data for rgb image
    new_mk = pil_image.new('RGB', (new_init_shape, new_init_shape), (0, 0, 0)) # new palette to paste data for mask image

    class_group = []
    image_location = {
        4 : [(150,150), (150, 650), (650, 150), (650,650)],
        6 : [(150, 0),(150, 400),(150,800),(650,0),(650,400),(650,800)],
        9 : [(0,0),(0,400),(0,800),(400,0),(400,400),(400,800),(800,0),(800,400),(800,800)]
    }

    for n , ((img, mk), loc) in enumerate(zip(data, image_location[num_images])):
        # Check input image is properly provided
        if _data_check(img, mk):
            return "error"

        # image
        try:
            # TODO : add random.gauss()
            # shift location to paste image
            # if pos_var:
            #     loc = (random.gauss(0, pos_var)+loc[0], random_gauss(0, pos_var)+loc[1])

            class_group.append(int(os.path.basename(img)[:2])) # beginning of file name is class(object) number
            img = pil_image.open(img)
            new_img.paste(img, loc)
            mk = pil_image.open(mk)
            new_mk.paste(mk, loc)
        except:
            print("Could not open images, check these images")
            return "error"

    # Check both images are properly concatenated
    if image_check_filter(new_img, new_mk) == None:
        print("These images have not properly been concatenated.")
        return "error"

    # set image size to generate
    # if not attributed size of image, the size is going to be the same as image size before processed
    new_img = new_img.resize(in_shape)
    new_mk = new_mk.resize(in_shape)
    return (new_img, new_mk, class_group)


# TODO: add shift parameter "pos_var"
# randomly input data and generate concatenated image for test
def mixed_image_generator(img_path,
                          img_shape=400,
                          num_data=100,
                          # pos_var=None,
                          to_path=None,
                          shuffle=True,
                          classify=False,
                          progress_check=True):
    """
    return mixed image and mask image dataset
    in this generator, NO reuse data
    arguments:
        img_path: path to deractory which contains rgb and mask images, these data has not to be processed
        img_shape: int, list or tuple (height and width). if int, set as square.
        num_data: number of generating data
        to_path: if not None, save generated data to indicated directory
        shuffle: if True, shuffle input image data
        classify: if True, return pixelwisely classified matrix
        progress: if True, apply tqdm
    return:
        imgs, masks: generated dataset
        class_arr : return only when classify is True, pixelwisely classified matrix
        class_list: obclass number list in image, maximum 9 images in one image
    """
    import random
    _images = sorted(glob(os.path.join(img_path, 'images', '*.png')))
    _masks = sorted(glob(os.path.join(img_path, 'masks', '*.png')))
    _dataset = zip(_images, _masks)

    assert len(_images)>0, "Input rgb image is empty."
    assert len(_masks)>0, "Input mask image is empty."

    if type(img_shape)==int:
        img_shape=(img_shape, img_shape)

    if shuffle == True:
        # if seed is not None:
        #     random.seed(seed)
        random.shuffle(_dataset)

    if to_path != None:
        _makedirs_if_none(os.path.join(to_path, 'images'))
        _makedirs_if_none(os.path.join(to_path, 'masks'))

    # initial parameter
    image_num = [4,6,9] # one image has 4, 6 or 9 objects each
    imgs = []
    masks = []
    class_lists = []
    img_index = 0
    stop = False # stop generate when it turns into True

    if progress_check:
        from tqdm import tqdm
        num_range = tqdm(xrange(num_data))
    else:
        num_range = xrange(num_data)

    # load data and resize using PIL
    for i in num_range:
        # set how many images put
        random.shuffle(image_num)
        _num = image_num[0]

        # Extract image data to concatenate
        data = _dataset[img_index:img_index+_num]
        # Concatenate data returned as tuple (image, mask, class_group)
        data = _image_concat(data, _num, img_shape)
        img_index += _num
        # if returned error, print which data has problem
        if data == "error":
            print("Error data :\n{}".format([os.path.basename(d[0]) for d in _dataset[img_index-_num:img_index]]))
            continue # continue until loop is done

        # stop generate because no more data to use
        if stop:
            break

        # save data if need
        if to_path != None:
            file_name = ['{:02d}'.format(d) for d in data[2]]
            file_name = '_'.join(file_name) + '_{:04d}.png'.format(i)
            data[0].save(os.path.join(to_path, 'images', file_name))
            data[1].save(os.path.join(to_path, 'masks', file_name))

        # add to list the processed data
        imgs.append(np.expand_dims(data[0], axis=0))
        masks.append(np.expand_dims(data[1], axis=0))
        class_lists.append(data[2])

    assert len(imgs)>0, "Data is not generated."

    # transform list to array with np.vstack
    imgs = np.vstack(imgs)
    masks = np.vstack(masks)

    if classify:
        # Initialize shape
        class_arr = np.zeros((num_data,) + img_shape + (1,))
        # import color-class dict
        color_dict = helper.class_color_dict()

        # create pixelwisely classified array
        for n, one_class_list in enumerate(class_lists):
            for num_class in one_class_list:
                # if match with class number, 1(True), else 0(False)
                class_arr[n] += np.expand_dims((masks[n]==color_dict[num_class]).all(axis=-1), axis=-1) * num_class

        return imgs, class_arr, class_lists

    return imgs, masks, class_lists


def _data_check(img1, img2):
    """data name check"""
    if os.path.basename(img1) != os.path.basename(img2):
        print("Input image names are not same.")
        print("image1 : {}, image2 : {}".format(os.path.basename(img1), os.path.basename(img2)))
        return False


def image_check_filter(img, mask):
    """Check mask image is correctly masked with image"""
    if np.allclose(np.clip(np.sum(img, axis=2), 0, 1), np.clip(np.sum(mask, axis=2), 0, 1)):
        return img, mask
    else:
        return None

"""
if __name__ == '__main__':
    path = input("Put a path which contains images/masks derectory")
    num_data = input("Put number of data ")
    # TODO: set generating process
"""
