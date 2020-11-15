# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import argparse
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from send2trash import send2trash

import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input

keras.backend.clear_session()
# -

#define paths and constants
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
#data_path = "/Users/victorialiu/git/creatica/code/data/"
batch_size = 32
TARGET_SIZE = 299

# ### Command Line Argument Parser
# We want to be able to run our code from the command line (at least in the `.py` version of this notebook), so we use an argument parser to translate command line arguments. We require a subdirectory from which to validate images to see if they are broken.

## Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--img-directory',
        help='subdirectory to validate images from using PIL' +
            'train, test',
        required = True)
#     parser.add_argument('-t', '--test-or-train',
#         help='directory to validate images from using ImageDataGenerator' +
#             'test, train',
#         required = True)

    return parser.parse_args()

# ### Data Validation

def validate_images_PIL(img_directory = 'train'):
    """
    Sends to trash jpg images that cannot be opened with PIL.Image
    """
    bad_files = []
    
    #img_directory is the /train/ or /test/ directory
    img_directory = os.path.join(data_path, img_directory)
    
    #categories_dir gives the full path of
    #/train/category1 or /test/category1 etc
    categories_dirs = [
            os.path.join(img_directory, category) 
            for category in os.listdir(img_directory)
            if os.path.isdir(os.path.join(img_directory, category))
        ]
    

    #go through the full path of each category
    for category_dir in categories_dirs:
        #go through each file of /train/category1 or /test/category1 etc
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg'):
                try:
                    Image.open(
                        os.path.join(category_dir, filename)
                        )
                except:
                    bad_files.append(
                        os.path.join(category_dir, filename)
                        )
                    send2trash(os.path.join(category_dir, filename))
    print(f'bad files according to PIL: {bad_files}')
    print('removed all bad files to trash')
    return bad_files

# validate_images_PIL(img_directory = 'test')   

# ### ImageDataGenerator Validation

# ### Image Pre-processing with InceptionV3 net
# Even though we've validated our images with `PIL.Image`, we would like to validate it with the actual package we are augmenting our images with, `ImageDataGenerator`. Here, we will validate those images. Rather than automatically removing images to the trash, we return a list of incorrect files that we will manually delete. The description of `image_data_augment` and `get_images` is given below, and the actual validation code is given in the code cell block afterwards.
#
# Next, we do data augmentation in order to "create" more data to train from. Data augmentation includes shifting the image in small ways such that the same image can be trained from multiple perspective (i.e. a rotated image of a hot dog is still a hot dog, and now we'll have more training data). We also make sure to normalize the image by dividing by the maximum pixel value of $255$. We also write a helper function to easily call based on whether we are using testing or training data.

# +
def image_data_augment(rescale=1/255, shear_range = False, zoom_range = False, horizontal_flip = False):
    """
    declare ImageDataGenerator class for augmenting images using shear, zoom, and flips
    normalize with 1./255
    """
    return (ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip))

def get_images(train_or_test):
    if train_or_test == 'train':
        datagen = image_data_augment(shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    else:
        datagen = image_data_augment()
        
    generator = datagen.flow_from_directory(
        os.path.join(data_path, train_or_test),
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    return generator
# -

# Although we have done manual data validation through eyeballing broken images, sometimes human error happens when we are looking through thousands of images! After hours of debugging despair, we decided it's much easier to have the computer debug broken images for us. Here is a function that will tell us all the file names of the broken images. This code is a little clunky, since we are doing it after we've already made the generator.

def determine_invalid_images(gen):
    """
    data validation
    """
    incorrect_files = []
    for i in range(len(gen)):
        try:
            a = gen[i]
        except:
            incorrect_files.append(gen.filenames[i])
            print(f'bad index at: {i}')
            print(f'bad filename DataImageGenerator: {gen.filenames[i]}')
            print('need to manually remove')
    return incorrect_files
# gen = get_images('train')
# determine_invalid_images(gen)

# ### Main command line function

def main():
    
# # comment this out when running from command line!
#     img_directory = 'test'
    
#     comment out when not running from cmdline
#     get cmdline args
    args = parse_args()
    img_directory = args.img_directory


    # validate images with PIL
    validate_images_PIL(img_directory = img_directory)   
    
    #validate images of ImageDataGenerator
    gen = get_images(img_directory)
    determine_invalid_images(gen)
    
    return True



if __name__ == "__main__":
    main()
    

# **Authors**: Victoria Liu and Gloria Liu
#
# **Last modified**: November 2020
#
# **Description**: A script to remove broken images and return a list of images incompatible with ImageDataGenerator


