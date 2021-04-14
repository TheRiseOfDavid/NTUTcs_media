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

data_amount = 300

# get people data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
image_people = lfw_people['images'][:data_amount]

# get car data
car = list()
for x in range(data_amount):
    img = cv2.imread('./car/%05d.jpg' % (x + 1))
    img = cv2.resize(img, (37, 50))
    car.append(img)
car = np.array(car)

# let people's target is 0, car is 1
target_people = [0] * data_amount
target_car = [1] * data_amount
target = target_car + target_people

# put people and car images in images list
images = list()
for x in car:
    images.append(x)
for x in image_people:
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
x_train, x_test, y_train, y_test = train_test_split(hog_images, target, test_size=0.2, random_state=0)

# svm train
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(x_train, y_train)


print("accuracy")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
    
#cv2.waitKey(0)
#cv2.destroyWindow()