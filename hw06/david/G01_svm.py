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

data_amount = 100
train_amount = 90
test_amount = 10

dogs = list()
cats = list()
for i in range(1,data_amount+1):
  image = cv2.imread("../Doraemon/images (%d).jpg" % i)
  image = cv2.resize(image, (349,256))
  dogs.append(image)
for i in range(1,data_amount+1):
  image = cv2.imread("../conan/images (%d).jpg" % i)  
  image = cv2.resize(image, (349,256))
  cats.append(image)

hog_dogs = list()
hog_cats = list()
for image in dogs:
  fd, hog_image = hog(
      image,
      orientations=8,
      pixels_per_cell=(9,9),
      cells_per_block=(1,1),
      visualize=True,
      )
  hog_dogs.append(fd)
for image in cats:
  fd, hog_image = hog(
      image,
      orientations=8,
      pixels_per_cell=(9,9),
      cells_per_block=(1,1),
      visualize=True,
      )
  hog_cats.append(fd)      
  
# train model
train_target_dog = [0] * train_amount
train_target_cat = [1] * train_amount
train_target = train_target_dog + train_target_cat
train_images = hog_dogs[:train_amount] + hog_cats[:train_amount]

# test model
test_target_dog = [0] * test_amount
test_target_cat = [1] * test_amount
test_target = test_target_dog + test_target_cat      
test_images = hog_dogs[-test_amount:] + hog_cats[-test_amount:] #last

#svm train
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(train_images, train_target)

print("accuracy")
print("train:", clf.score(train_images, train_target))
print("test:", clf.score(test_images, test_target), "\n")


