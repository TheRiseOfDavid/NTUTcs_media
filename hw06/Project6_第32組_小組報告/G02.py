# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:57:30 2021

@author: user
"""
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
import hw06_fn

data_amount = 100
train_amount = 80
test_amount = 20

#return list
# dogs = hw06_fn.read("../resize_dog/dog_%.3d.jpg", data_amount)
# cats = hw06_fn.read("../resize_cat/cat_%.3d.jpg", data_amount)

dogs = hw06_fn.read("../Doraemon/images (%d).jpg", data_amount)
cats = hw06_fn.read("../conan/images (%d).jpg", data_amount)

sift_dogs = hw06_fn.sift(dogs)
sift_cats = hw06_fn.sift(cats)

train_sift = sift_dogs[:train_amount] + sift_cats[:train_amount]
test_sift = sift_dogs[-test_amount:] + sift_cats[-test_amount:]

train_features = hw06_fn.kmeans_return_features(20, train_sift)
test_features = hw06_fn.kmeans_return_features(20, test_sift)
train_target = [0] * train_amount + [1] * train_amount
test_target = [0] * test_amount + [1] * test_amount

clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(train_features, train_target)

print("doraemon and conan:")
print("accuracy")
print("train:", clf.score(train_features, train_target))
print("test:", clf.score(test_features, test_target))
