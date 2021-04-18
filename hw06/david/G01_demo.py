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
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split

data_amount = 100
train_amount = 90
test_amount = 10

dogs = list()
cats = list()



for i in range(1,data_amount+1):
  image = cv2.imread("../resize_dog/dog_%.3d.jpg" % i)
  dogs.append(image)
for i in range(1,data_amount+1):
  image = cv2.imread("../resize_cat/cat_%.3d.jpg" % i)  
  cats.append(image)

#sift_dogs
sift_dogs = list()
for image in dogs:
  sift = cv2.SIFT_create()
  kp, des = sift.detectAndCompute(image, None)
  sift_dogs.append(des)

#sift_cats
sift_cats = list() 
for image in cats:
  sift = cv2.SIFT_create()
  kp, des = sift.detectAndCompute(image, None)
  sift_cats.append(des)      

train_sift = sift_dogs[:train_amount] + sift_cats[:train_amount]
test_sift = sift_dogs[-test_amount:] + sift_cats[-test_amount:]

# KMeans
descriptors = train_sift[0]
for it in train_sift[1:]:
  descriptors = np.vstack((descriptors, it))  

k = 20
train_voc, train_variance = kmeans(descriptors,k,1)

#features histogram
im_features = np.zeros((data_amount*2, k), "float32")
for i in range(data_amount*2):
  words, distance = vq(sift[i], voc)
  for j in words:
    im_features[i][j] += 1

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


