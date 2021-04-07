#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 1 11:30:34 2021

@author: kuotzulin
"""
# 小組專案

from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = datasets.load_wine()
#print(wine)

# 以所有特徵為輸入
x = wine.data
y = wine.target

# 訓練/測試比例：0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# kernel=linear, C=1, gamma=auto
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# kernel=rbf(Radial Basis Function)
clf=svm.SVC(kernel='rbf', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy kenel_func=rbf")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# kernel=poly(polynomial)
clf=svm.SVC(kernel='poly', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy kenel_func=poly")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# kernel=sigmoid
clf=svm.SVC(kernel='sigmoid', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy kenel_func=sigmoid")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# C=0.3
clf=svm.SVC(kernel='linear', C=0.3, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy C=0.3")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# C=2.7
clf=svm.SVC(kernel='linear', C=2.7, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy C=2.7")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# gamma=scale
clf=svm.SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(x_train, y_train)
print("accuracy gamma=scale")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# gamma=0.1
clf=svm.SVC(kernel='rbf', C=1, gamma=0.1)
clf.fit(x_train, y_train)
print("accuracy gamma=0.1")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# gamma=9.9
clf=svm.SVC(kernel='rbf', C=1, gamma=9.9)
clf.fit(x_train, y_train)
print("accuracy gamma=9.9")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# 訓練/測試比例：0.4
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy test_size=0.4")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")
# 訓練/測試比例：0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=0)
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy test_size=0.7")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# 以alcohol, malic_acid特徵為輸入
x = wine.data[:,0:2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy features:alcohol, malic_acid")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# 以total_phenols, flavanoids, nonflavanoid_phenols特徵為輸入
x = wine.data[:,5:8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy features:total_phenols, flavanoids, nonflavanoid_phenols")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")

# 以alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols特徵為輸入
x = wine.data[:,0:6]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf=svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(x_train, y_train)
print("accuracy features:alcohol, malic_acid, ..., total_phenols")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")



#plt.scatter(x_train[:,0], x_train[:,1])
#plt.plot([0,1,3,4], [1,3,11,14])
#plt.show()

#cv2.waitKey(0)
#cv2.destroyWindow()