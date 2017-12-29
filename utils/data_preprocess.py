#!/usr/bin/env python

import random
import numpy as np
import os
import zipfile, shutil
from scipy import misc
from glob import glob
from tqdm import tqdm
from PIL import Image
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image

def makedirs_if_none(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def unzip_tless_files(path, to_path, num_folders=30):
    """Unzip files specifically for T-less files
    This function is only to unzip for T-Less dataset"""
    makedirs_if_none(to_path)
    for i in xrange(1,num_folders+1):
        with zipfile.ZipFile(os.path.join(path, 't-less_v2_train_kinect_{:02d}.zip'.format(i)), 'r') as zf:
            zf.extractall(path=os.path.join(to_path))


def _shuffle_data(data, SEED):
    random.seed(SEED)
    random.shuffle(data)


def _add_extracted_files(path, nums):
    """
    Extract indicated numbered files
        example: nums = [1, 5] -> all xxx1.png and xxx5.png such as 0341.png, 1195.png
    """
    paths = []
    for n in nums:
        paths.extend(glob(os.path.join(path, '*{}.png'.format(n))))
    return paths


def _savedata(paths, to_path, copy=True):
    """Save images"""
    makedirs_if_none(to_path)
    if copy:
        for p in paths:
            shutil.copyfile(p, os.path.join(to_path, os.path.basename(p)))
    else:
        for p in paths:
            os.rename(p, os.path.join(to_path, os.path.basename(p)))


def _add_word_to_filename(file_path, word, to_path, copy=False):
    """Add word to file name
        Example:
            0321.png and word is "05" -> 05_0321.png
    """
    if file_path is None:
        raise ValueError("Input file does not exist")
    _to_path = os.path.join(to_path, word+'_'+os.path.basename(file_path))
    if copy:
        shutil.copyfile(file_path, _to_path)
    else:
        os.rename(file_path, _to_path)


def extract_rgb_data(path, to_path, copy=True):
    """Extract rgb data from dataset
    This is specificully adjusted to T-Less dataset
    """
    for l in os.listdir(path):
        _to_path = os.path.join(to_path, l)
        makedirs_if_none(_to_path)
        _path = os.path.join(path, l, 'rgb')
        files = glob(os.path.join(_path, '*.png'))
        for f in files:
            _savedata(f, _to_path, copy)


def pick_data_roughly(path, to_path, use_ratio=1.0, copy=True, two_levels=False, add_num=False):
    """Pick out using data to control amount of dataset and save to other folder
    Arguments:
        path: path to folder which has dataset
        to_path: path to folder to save picked data
        use_ratio: ratio to pick data (rough not precise),
                   choose a number between 0.1 and 1.0
                   if 1.0, extract all data
        copy: if True, current data will be reserved
        two_levels: if True, assumed to be files stored a two levels folder structure
            example: two_levels=True
                - train (path)
                    - 01 (two_levels)
                        - 001.png
                        - 002.png
                    - 02 (two_levels)
                        - 001.png
                        - 002.png
        add_num: if True, add number to each file name and save to one folder(it works only when two_level is True)
    """
    if use_ratio < 0.1 or use_ratio > 1.0:
        raise ValueError("Input value is not valid")
    makedirs_if_none(to_path)
    use_ratio = int(use_ratio*10+0.5) # rounding
    ## pick using image file number to extract data
    nums = [i for i in xrange(0, 10, 10//use_ratio)][:use_ratio]

    if two_levels:
        lists = [l for l in os.listdir(path) if l.isdigit()] # only work for numbered directory
        for n in lists:
            files = _add_extracted_files(os.path.join(path, n), nums)
            if add_num:
                for f in files:
                    _add_word_to_filename(f, n, to_path, copy)
            else:
                _to_path = os.path.join(to_path, n)
                _savedata(files, _to_path, copy)
    else:
        files = _add_extracted_files(path, nums)
        for f in files:
            _to_path = os.path.join(to_path, n)
            _savedata(f, _to_path, copy)


def path_to_tensor(img_path, img_size):
    """ Convert image data to tensor"""
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def input_img(img_path, img_size=None, expand=False):
    """Convert image data to tensor
    Arguments:
        img_path: path to image folder
        img_size: target image size
            int, tuple or list
            int -> img_size = (int, int)
        expand: if False, return 3-D array, if True, return 4-D array
    Return:
        if expand is True
            4-D numpy array [1, image height, image width, channels]
        if expand is False
            3-D numpy array [image height, image width, channels]
    """
    if len(img_path) == 0:
        raise ValueError("Input is empty. Check input path.")
    if type(img_size)==int:
        img_size = (img_size, img_size)
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    if expand:
        return np.expand_dims(x, axis=0)
    else:
        return x


def input_img_from_dir(dir_path, img_size=192, two_levels=False):
    """Convert image data to tensor
    Arguments:
        dir_path: path to image folder
        img_size: target image size
            int, tuple or list
            int -> img_size = (int, int)
        two_levels: if True, assumed to be files stored a two levels folder structure
    Return:
        4-D numpy array
    """
    if two_levels:
        paths = glob(os.path.join(dir_path, '*', '*.png'))
    else:
        paths = glob(os.path.join(dir_path, '*.png'))
    if len(paths) == 0:
        raise ValueError("Input is empty. Check the directory path.")
    if type(img_size)==int:
        img_size = (img_size, img_size)
    img_tensors = [path_to_tensor(img_path, img_size) for img_path in tqdm(paths)]
    return np.vstack(img_tensors)


def mask_img_from_paths(path, colors):
    """Change object into unit color one and save to folder named "masks"
    Color is decided by class number from file name
        example:
            file name : 05_0443.png -> color number is 5
    Arguments:
        path: folder path which has images directory containing image data
        img_shape: input image shape, 2D list or tuple
        colors: n colors list,  nx3x1 numpy array
            [n, r, g, b] color
    """
    makedirs_if_none(os.path.join(path, 'masks'))
    files = glob(os.path.join(path, 'images', '*.png'))
    for f in tqdm(files):
        n = int(os.path.basename(f)[:2])-1 # class number -> color number
        img = cv2.imread(f)
        img = pixelwise_classify_img(img, colors[n], colored=True)
        Image.fromarray(img.astype('uint8')).save(os.path.join(path, 'masks', os.path.basename(f)), format='png')


def input_train_target_from_dir(dir_path, img_size=192, two_levels=False, shuffle=True, seed=None):
    """Convert image data to tensor and generate class tensor
    Arguments:
        dir_path: path to image folder
        img_size: target image size
            int, tuple or list
            int -> img_size = (int, int)
        two_levels: if True, assumed to be files stored a two levels folder structure
        shuffle: if True, returned data will be shuffled
        seed: int, initial internal state of the random number generator
    Return:
        4-D numpy array(image), 1-D numpy array(image class)
    """
    if type(img_size)==int:
        img_size = (img_size, img_size)

    if two_levels:
        if os.listdir(dir_path)==[]:
            raise InputError("No directory found")
        # extract data and convert to tensor directory each, then zip with class number
        imgs = []
        for n in os.listdir(dir_path):
            paths = glob(os.path.join(dir_path, n, '*.png'))
            x = [(path_to_tensor(p, img_size), int(n)) for p in paths]
            imgs.extend(x)
    else:
        paths = glob(os.path.join(dir_path, '*.png'))
        imgs = [(path_to_tensor(p, img_size), int(os.path.basename(p)[:2])-1) for p in paths]

    if shuffle:
        _shuffle_data(imgs, seed)

    imgs, target = zip(*imgs)
    return np.vstack(imgs), np.vstack(target)


def to_categorical_tensor(tensor, num_classes=30):
    """transform image tensor into categorical tensor"""
    return np_utils.to_categorical(tensor, num_classes)


def pixelwise_classify_img(img, num, colored=False):
    """Convert channel to class number in image which has one object
    (This is adjusted to t-less training data)
    Arguments:
        img : 3x1 array [height, width, 3(rgb)]
        num: int, class number
        colored:
    Return:
        [height, width, 1(num) or 3(rgb)]
    """
    # replace rgb with class number on pixels of object in image data
    if colored:
        if type(num) == int:
            raise ValueError("Input has to be 3x1 list or tuple.")
        return np.expand_dims(np.clip(np.sum(img, axis=2), 0, 1), axis=2) * \
                np.tile(color_list[t[0]], img.shape[0]*img.shape[1]).reshape(img.shape)
    else:
        if type(num) != int:
            raise ValueError("Input has to be an integer.")
        return np.expand_dims(np.clip(np.sum(img, axis=2), 0, 1)*num, axis=2)


def pixelwise_classify_images_for_one_obj(inputs, targets, num_classes=30, colored=False):
    """Create pixelwisely added class number to image from array
    Arguments:
        inputs: 4-D array containing image data
        targets: 1-D array which has class numbers related to images
        num_classes: int, total number of class
        colored : if True, each pixel will have rgb
                  if False, each pixel will have class number
    Return:
        imgs: 4-D numpy array [batch, image height, image depth, 1(class) or 3(rgb)]
    """
    if len(inputs) == 0:
        raise ValueError("Input is empty.")

    if colored:
        # prepare colors for setting as mask
        from utils import helper
        color_list = helper.color_rgb_list()
        # output shape => [number of data, height, width, rgb channel]
        arr = np.zeros((inputs.shape[:3])+(3,))
        for n, (img, t) in enumerate(zip(inputs, targets)):
            # if a pixel has color, replace fixed color of target class
            arr[n] = np.expand_dims(np.clip(np.sum(img, axis=2), 0, 1), axis=2) * \
                    np.tile(color_list[t[0]], img.shape[0]*img.shape[1]).reshape(img.shape)

    else:
        # output shape => [number of data, height, width, 1(class number)]
        arr = np.zeros((inputs.shape[:3])+(1,))
        for n, (img, t) in enumerate(zip(inputs, targets)):
            # replace rgb with class number on pixels of object in image data
            arr[n] = np.expand_dims(np.clip(np.sum(img, axis=2), 0, 1)*t, axis=2)

    return arr

##  Not use on this time
# def create_6d_data():
#     """
#     merge rgb data and depth dataset
#     """
# return data
