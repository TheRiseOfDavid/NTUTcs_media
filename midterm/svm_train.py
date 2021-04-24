# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 14:12:06 2021

@author: User
"""

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
import joblib
import hw07_fn #這是我自己寫的 functional progamming，請自行將上面的 function python code
#與此檔案放在同個資料夾底下

#hw07_fn.rename_image("./scissors/", "scissor")
# hw07_fn.rename_image("./papers/", "paper")
# hw07_fn.rename_image("./rocks/", "rock")

scissors = hw07_fn.read("./scissors/")
papers = hw07_fn.read("./papers/")
rocks = hw07_fn.read("./rocks/")

sift_scissors = hw07_fn.sift(scissors)
sift_papers = hw07_fn.sift(papers)
sift_rocks = hw07_fn.sift(rocks)

x = hw07_fn.kmeans_return_features(30, sift_scissors + sift_rocks + sift_papers)
y = ["剪刀"] * len(sift_scissors) + ["石頭"] * len(sift_rocks) + ["布"] * len(sift_papers)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = svm.SVC(kernel="linear", C=1, gamma="auto", probability=True)
clf.fit(x_train, y_train) #開始進行訓練

print("accuracy") #準確率
print("train:", clf.score(x_train, y_train)) #訓練集分數
print("test:", clf.score(x_test, y_test)) #測試集分數

joblib.dump(clf, "svc_paper_scissor_rock_game.pkl")

