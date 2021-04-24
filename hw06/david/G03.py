# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:28:29 2021

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

dogs = hw06_fn.read("../Doraemon/images (%d).jpg", data_amount)
cats = hw06_fn.read("../conan/images (%d).jpg", data_amount)

images = dogs + cats
sift_feature = cv2.SIFT_create()
sift = hw06_fn.sift(images, sift_feature)

features = hw06_fn.kmeans_return_features(20, sift)
target = [0] * data_amount + [1] * data_amount

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(x_train, x_test)

print("accuracy")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test))