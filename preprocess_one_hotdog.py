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

import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input

keras.backend.clear_session()

# +
##define paths and constants

# Remove src from cwd if necessary
cwd = os.getcwd()
print(cwd)
if os.path.basename(cwd) == 'src': cwd = os.path.dirname(cwd)

# Create img directory to save images if needed
os.makedirs(os.path.join(cwd, 'demo/test'), exist_ok=True)

#define future datapath
single_img_path = os.path.join(cwd, 'demo')
batch_size = 32
TARGET_SIZE = 299
# -

def image_data_augment(rescale=1/255, shear_range = False, zoom_range = False, horizontal_flip = False):
    #declare ImageDataGenerator class for augmenting images using shear, zoom, and flips
    #normalize with 1./255
    return (ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip))

# +
def get_image():
    datagen = image_data_augment()  
    generator = datagen.flow_from_directory(
        single_img_path,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        #batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    return generator

def preprocess_inception():
    """
    inception for transfer learning
    """
    #transfer learning with InceptionV3, a pre-trained cnn
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    

    generator = get_image()
    bottleneck_features = model.predict(generator, len(generator), verbose=1)

    #save model with the bottleneck features
    np.savez(f'inception_features_test_image', features=bottleneck_features)
    
    return True



def get_single_image_data():
    #augment images and use inception net
    preprocess_inception()

    #load training data and define labels, where 0 is hotdog and 1 is nothotdog
    test_image_data = np.load('inception_features_test_image.npz')['features']
    
    return test_image_data

# # load the output of inceptionv3 for our test image
# test_image_data = get_single_image_data()
# -

# **Authors**: Victoria Liu and Gloria Liu

# **Last modified**: November 2020

# Description: A script to pre-process a single image that is dropped into the image drop of the Flask app

# **Credits** The data augmentation code is heavily modified from [J-Yash's open-source code](https://github.com/J-Yash/Hotdog-Not-Hotdog).