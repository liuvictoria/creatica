# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import argparse
import simplejson
import matplotlib.pyplot as plt
import cv2

import numpy as np

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3, preprocess_input

keras.backend.clear_session()

""
#define paths and constants
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
#data_path = "/Users/victorialiu/git/creatica/code/data/"
batch_size = 32
TARGET_SIZE = 299

""
## Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name',
        help='prefix for file to save trained model to ' +
            '(e.g. dense_arch1, conv_regularize05, etc.)',
        required=True)
    parser.add_argument('-r', '--regularizer-strength',
        help='strength of l2 regularization to use',
        type=float, default=0.00)

    return parser.parse_args()

""
def image_data_augment(rescale=1/255, shear_range = False, zoom_range = False, horizontal_flip = False):
    #declare ImageDataGenerator class for augmenting images using shear, zoom, and flips
    #normalize with 1./255
    return (ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip))

""
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

def preprocess_inception():
    """
    inception for transfer learning
    """
    #transfer learning with InceptionV3, a pre-trained cnn
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    
    for train_or_test in ['train', 'test']:
        generator = get_images(train_or_test)

        bottleneck_features = model.predict(generator, len(generator), verbose=1)
        
        #save model with the bottleneck features
        np.savez(f'inception_features_{train_or_test}', features=bottleneck_features)
    
    return True



def get_data():
    #augment images and use inception net
    preprocess_inception()

    #load training data and define labels, where 0 is hotdog and 1 is nothotdog
    train_data = np.load('inception_features_train.npz')['features']

    #requires the number of hotdog and nothotdog samples to be the exact same
    train_data_type_count = int(len(train_data) / 2)
    train_labels = np.array([0] * train_data_type_count + [1] * train_data_type_count)

    #load testing data and define labels, where 0 is hotdog and 1 is nothotdog
    test_data = np.load('inception_features_test.npz')['features']

    #requires the number of hotdog and nothotdog samples to be the exact same
    test_data_type_count = int(len(test_data) / 2)
    test_labels = np.array([0] * test_data_type_count + [1] * test_data_type_count)
    
    return (train_data, train_labels, test_data, test_labels)

# # Importing the hotdog dataset
# #may take a few seconds
# everything = get_data()

# train_data, train_labels = everything[0], everything[1]


def build_conv_net(reg_param, train_data_shape):
    #train_data_shape = train_data.shape[1:] = (8, 8, 2048)
    model = Sequential()
    
    #convolutional layer with 32 3x3 trainable filters, using rectified linear units.
    #Padding to result in the same shape as the original picture. Add reg_param if true
    #use l2 regularization if reg_param is given in command line
    model.add(Conv2D(
        32, (3, 3),
        activation='relu',
        input_shape=train_data_shape,
        padding='same',
        kernel_regularizer=l2(reg_param)
    ))
    
    # convolutional layer with 32 3x3 trainable filters, using rectified linear units.
    # Padding to result in the same shape as the original picture.
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    
    # max pooling for noise reduction
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    # dropout for more regularization
    model.add(Dropout(0.25))

    # second block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    # use softmax for multi-class classifications
    # (i.e. model.add(Activation('softmax')))
    model.add(Dense(1, activation='sigmoid'))
    return model


""
# train_data, train_labels, test_data, test_labels = get_data()
# train_data_shape = train_data.shape[1:]

""
def main():
   
    
# # comment this out when running from command line!
#     model_name = 'debugging'
#     regularizer_strength = .00001


#     comment out when not running from cmdline
    ## get cmdline args
    args = parse_args()
    model_name = args.model_name
    # get regularization strength, if defined. Otherwise, it is 0
    regularizer_strength = args.regularizer_strength
    
    

    # Remove src from cwd if necessary
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'src': cwd = os.path.dirname(cwd)

    # Create img directory to save images if needed
    os.makedirs(os.path.join(cwd, 'img'), exist_ok=True)
    plot_fname = os.path.join(cwd, 'img', '%s_learn.png' % model_name)

    # Create model directory to save models if needed
    os.makedirs(os.path.join(cwd, 'model'), exist_ok=True)
    model_weights_fname = os.path.join(cwd, 'model', model_name + '.h5')
    model_json_fname = os.path.join(cwd, 'model', model_name + '.json')



    # Importing the hotdog dataset
    #may take a few seconds
    (train_data, train_labels, test_data, test_labels) = get_data()
    train_data_shape = train_data.shape[1:]




    # build model
    model = build_conv_net(regularizer_strength, train_data_shape)
    
    # Print a summary of the layers and weights in the model
    model.summary()

    # Have our model minimize the binary cross entropy loss with the adam
    # optimizer (fancier stochastic gradient descent that converges faster)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy']
                 )

    # #set checkpointer to use in callback, to only keep the best model weights
    checkpointer = ModelCheckpoint(
        filepath='/Users/victorialiu/git/creatica/tmp',
        verbose=1, 
        save_weights_only=True,
        )

    #time to fit
    history = model.fit(train_data, train_labels,
            epochs=8,
            batch_size=batch_size,
            validation_split=0.3,
            verbose=2,
            callbacks=[checkpointer],
            shuffle=True)



    #load best model???
    model.load_weights('/Users/victorialiu/git/creatica/tmp')

    # Save model weights and json spec describing the model's architecture
    model.save(model_weights_fname)
    model_json = model.to_json()
    with open(model_json_fname, 'w') as f:
        f.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

        

    # Plot accuracy learning curve
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('%s accuracy' % model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.savefig(plot_fname)

    # Plot loss learning curve
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('%s loss' % model_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_fname)
    plt.show()



if __name__ == "__main__":
    main()



# 
# **Authors**: Victoria Liu and Gloria Liu

# **Last modified**: November 2020

# Description: A script to train and save a neural net to recognize hot dogs vs.
# non-hot dogs.

# **Credits** Parts of the code are originally part of a Caltech extra credit assignment (CS 156a), where
# Aadyot Bhatnagar wrote the parse_args() and main() functions. The conv-net code is heavily modified from [J-Yash's open-source code](https://github.com/J-Yash/Hotdog-Not-Hotdog).
# 


