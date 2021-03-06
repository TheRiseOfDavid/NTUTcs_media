#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 8 10:58:26 2021

"""

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import train_test_split

data_amount = 90

# get dog data
dog = list()
for x in range(data_amount):
    img = cv2.imread('../resize_dog/dog_%03d.jpg' % (x + 1))
    dog.append(img)
dog = np.array(dog)

# get cat data
cat = list()
for x in range(data_amount):
    img = cv2.imread('../resize_cat/cat_%03d.jpg' % (x + 1))
    cat.append(img)
cat = np.array(cat)

# let dog's target is 0, cat's is 1
target_dog = [0] * data_amount
target_cat = [1] * data_amount
target = target_dog + target_cat

# put dog and cat images in images list
images = list()
for x in dog:
    images.append(x)
for x in cat:
    images.append(x)

# hog
hog_images = []
for image in images:
    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(9, 9),
        cells_per_block=(1, 1),
        visualize=True,
    )
    hog_images.append(fd)

# split some data to be the test data
#x_train, x_test, y_train, y_test = train_test_split(hog_images, target, test_size=0.2, random_state=0)


test_images = list()
for x in range(10):
    img = cv2.imread('../resize_dog/dog_%03d.jpg' % (x + 91))
    test_images.append(img)
for x in range(10):
    img = cv2.imread('../resize_cat/cat_%03d.jpg' % (x + 91))
    test_images.append(img)
    
test_images = np.array(test_images)
    
test_hog = []
for image in test_images:
    fd, test_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(9, 9),
        cells_per_block=(1, 1),
        visualize=True,
    )
    test_hog.append(fd)
    

test_target = [0] * 10 + [1] * 10


# svm train
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(hog_images, target)

print("accuracy")
#print(hog_images[0])
print("train:", clf.score(hog_images, target))
#print(test_images[0])
print("test:", clf.score(test_hog, test_target), "\n")
    
#cv2.waitKey(0)
#cv2.destroyWindow()