# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:28:01 2021

@author: User
"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape

x = lfw_people.data
n_features = x.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

n_components = 150
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, x_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', 
          whiten=True).fit(x_train) #降維至 150 層
print("done in %0.3fs" % (time() - t0))

#重新定義圖片大小
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(x_train) #進行降維
x_test_pca = pca.transform(x_test) #進行降維
print("done in %0.3fs" % (time() - t0))

# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#參數，用來告訴 gridsearchCV 的懲罰與誤差，是圖像決定好的
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.01, 0.05, 0.01, 0.05, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
fit = clf.fit(x_train_pca, y_train) #開始訓練，找最適合的
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_) #找到最佳參數組合及其準確率。

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(x_test_pca) #預測
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

