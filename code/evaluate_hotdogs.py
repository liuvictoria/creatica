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
import simplejson
import matplotlib.pyplot as plt
import cv2

import numpy as np

import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input

keras.backend.clear_session()
# -

#define paths and constants
data_path = "/Users/victorialiu/git/creatica/code/data/"
test_img_path = os.path.join(data_path, 'test/')
batch_size = 32
TARGET_SIZE = 299

## Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name',
        help='prefix for saved trained model we want to evaluate ' +
            '(e.g. dense_arch1, conv_regularize05, etc.)',
        required=True)
    return parser.parse_args()

# +
def get_data():
#     #augment images and use inception net
#     preprocess_inception()

    #load training data and define labels, where 0 is hotdog and 1 is nothotdog
    #to get inception_features_train.npz, make sure to run train_hotdogs.py first
    train_data = np.load('inception_features_train.npz')['features']

    #requires the number of hotdog and nothotdog samples to be the exact same
    train_data_type_count = int(len(train_data) / 2)
    train_labels = np.array([0] * train_data_type_count + [1] * train_data_type_count)

    #load testing data and define labels, where 0 is hotdog and 1 is nothotdog
    #to get inception_features_test.npz, make sure to run train_hotdogs.py first
    test_data = np.load('inception_features_test.npz')['features']

    #requires the number of hotdog and nothotdog samples to be the exact same
    test_data_type_count = int(len(test_data) / 2)
    test_labels = np.array([0] * test_data_type_count + [1] * test_data_type_count)
    
    return (train_data, train_labels, test_data, test_labels)

# # # Importing the hotdog dataset
# # #may take a few seconds
# everything = get_data()


# +
def incorrect_labels_indices(rounded_predictions, test_labels):
    incorrect_labels = []
    incorrect_classification_indices = []
    for i in range(rounded_predictions.shape[0]):
        if not rounded_predictions[i] == test_labels[i]:
            incorrect_classification_indices.append((i, test_labels[i], rounded_predictions[i]))
            incorrect_labels.append(rounded_predictions[i][0])
    return (incorrect_classification_indices, incorrect_labels)



def get_incorrect_classification_fpaths(incorrect_classification_indices):
    incorrect_classification_file_paths = []
    for incorrect_info in incorrect_classification_indices:
        #if it is actually a hotdog

        if incorrect_info[1] == 1:
            i = incorrect_info[0]
            incorrect_classification_file_paths.append(
                'nothotdog/'+ os.listdir(os.path.join(data_path, 'test/nothotdog'))[i - 140])
        else:
            i = incorrect_info[0]
            incorrect_classification_file_paths.append(
                'hotdog/'+ os.listdir(os.path.join(data_path, 'test/hotdog'))[i])
    return incorrect_classification_file_paths
        
        


def get_image(file_path, pred_value, true_value, cwd, model_name, i):
    if os.path.isfile(test_img_path + file_path):
        image_bgr = cv2.imread(test_img_path + file_path,cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_resized = cv2.resize(image_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation = cv2.INTER_CUBIC)
        plt.imshow(image_rgb_resized)
        plt.axis("off")
        plt.title('Pred: %s\nTrue: %s' % (pred_value, true_value))
        plt.savefig(os.path.join(cwd, 'img', '%s_mistake_%s.png') % (model_name, str(i)))
        plt.show()


# +
def main():
    # comment out if using command line
    model_name = 'debugging'
    
#     args = parse_args()
#     model_name = args.model_name



    # Remove src from cwd if necessary
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'src': cwd = os.path.dirname(cwd)

    # Create img directory to save images if needed
    os.makedirs(os.path.join(cwd, 'img'), exist_ok=True)

    # Create model directory to save models if needed
    os.makedirs(os.path.join(cwd, 'model'), exist_ok=True)
    model_weights_fname = os.path.join(cwd, 'model', model_name + '.h5')
    model_json_fname = os.path.join(cwd, 'model', model_name + '.json')

    # Load hotdog model and its weights
    with open(model_json_fname, 'r') as f: model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_fname)

    # Get hotdog data
    (train_data, train_labels, test_data, test_labels) = get_data()

    # Compile model and evaluate its performance on training and test data
    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])

    score = model.evaluate(train_data, train_labels, verbose=0)
    print()
    print('Training loss:', score[0])
    print('Training accuracy:', score[1])

    score = model.evaluate(test_data, test_labels, verbose=0)
    print()
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    
    
    #print and save pictures that were inaccurately classified?
    # use previously written helper functions to find incorrect classification indices,
    # and the file paths to incorrectly classified images
    rounded_predictions = model.predict_classes(test_data, batch_size=batch_size, verbose=1)
    
    #get incorrectly classified indices and their labels
    incorrect_classification_indices, incorrect_labels = incorrect_labels_indices(
        rounded_predictions, test_labels
        )
    
    #get file paths for incorrectly categorized pics to plot them later
    incorrect_classification_file_paths = get_incorrect_classification_fpaths(incorrect_classification_indices)
    
    print('Some incorrectly classified images')
    
    for i, incorrect_info in enumerate(incorrect_classification_indices):
        if i // 10 == 0:
            if 'nothotdog' in incorrect_classification_file_paths[i]:
                true_value = "not hotdog"
            else:
                true_value = "hotdog"

            index = incorrect_info[0]
            if incorrect_labels[i] == 1:
                pred_value = "not hotdog"
            else:
                pred_value = "hotdog"
            get_image(
                incorrect_classification_file_paths[i],
                pred_value, true_value,
                cwd, model_name, i
                )
    return True

if __name__ == '__main__': main()

# main()
# -

# **Authors**: Victoria Liu and Gloria Liu

# **Last modified**: November 2020

# Description: A script to evaluate a saved neural net that should recognize hot dogs vs.
# non-hot dogs.

# **Credits**: Parts of the code are originally part of a Caltech extra credit assignment (CS 156a), where Aadyot Bhatnagar wrote the parse_args() function. The image augmentation code is heavily modified from [J-Yash's open-source code](https://github.com/J-Yash/Hotdog-Not-Hotdog).
