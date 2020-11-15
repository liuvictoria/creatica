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
import cv2
import matplotlib.pyplot as plt
# -

#define paths
data_path = "/Users/victorialiu/git/creatica/code/data/"
TARGET_SIZE = 299

# print(cv2.__version__)
img_path = os.path.join(data_path, 'test/')
def get_image(file_path):
    if os.path.isfile(img_path + file_path):
        image_bgr = cv2.imread(img_path + file_path,cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_resized = cv2.resize(image_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation = cv2.INTER_CUBIC)
        plt.imshow(image_rgb_resized)
        plt.title(file_path)
        plt.axis("off")
        plt.show()

# +
def main():
    #get non-hotdog pictures from directory
    nothotdog_list = os.listdir(os.path.join(data_path, 'test/nothotdog'))
    nothotdog_pics = [os.path.join('nothotdog/', nothotdog_list[i]) for i in range(len(nothotdog_list))]

    #get hotdog pictures from directory
    hotdog_list = os.listdir(os.path.join(data_path, 'test/hotdog'))
    hotdog_pics = [os.path.join('hotdog/', hotdog_list[i]) for i in range(len(hotdog_list))]

    #concat
    all_pics = nothotdog_pics + hotdog_pics

    #plot images
    for i in range(0, len(all_pics), 40):
        print(all_pics[i])
        get_image(all_pics[i])
    return True

if __name__ == '__main__': main()
# -

# **Authors**: Victoria Liu and Gloria Liu
#
# **Last modified**: November 2020
#
# Description: A script to visualize hot dogs before doing training. See what we're working with, without data snooping!
