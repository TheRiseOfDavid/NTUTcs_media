# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:26:14 2021

@author: User
"""
import cv2
import joblib
import hw07_fn
from sklearn.decomposition import PCA
import numpy as np
import os

def sift(image):
    clf = joblib.load("./sift_paper_scissor_rock_game.pkl") 
    sift_image = hw07_fn.sift([image])
    x = hw07_fn.kmeans_return_features(20, sift_image)
    predict = clf.predict(x)
    print("sift 辨識")
    print(predict)
    print(clf.predict_proba(x))
    return predict
    
def hog(image):
    clf = joblib.load("./hog_paper_scissor_rock_game.pkl") 
    x = hw07_fn.hog([image])
    predict = clf.predict(x)
    print("hog 辨識")
    print(predict)
    print(clf.predict_proba(x))
    return predict

#hw07_fn.rename_image("./game/", "image")p1 = hw07_fn.random_image("./game")

p1 = hw07_fn.random_image("./game")
#cv2.imshow("player1", p1)
p2 = hw07_fn.random_image("./game")
#cv2.imshow("player1", p1)

predict1 = sift(p1) 
predict2 = sift(p2) 

#predict1 = hog(p1) 
#predict2 = hog(p2)

lose = cv2.imread("lose.jpg")
lose = cv2.resize(lose, (400,313))
win = cv2.imread("win.jpg")
win = cv2.resize(win, (400,313))
tie = cv2.imread("tie.png")
tie = cv2.resize(tie, (400,313))

if(predict1 == "剪刀" and predict2 == "石頭"):
    result = np.hstack((p1,lose,p2))
elif(predict1 == "石頭" and predict2 == "布"):
    result = np.hstack((p1,lose,p2))
elif(predict1 == "布" and predict2 == "剪刀"):
    result = np.hstack((p1,lose,p2)) 
elif(predict1 == "剪刀" and predict2 == "布"):
    result = np.hstack((p1,win,p2))
elif(predict1 == "布" and predict2 == "石頭"):
    result = np.hstack((p1,win,p2))
elif(predict1 == "石頭" and predict2 == "剪刀"):
    result = np.hstack((p1,win,p2))
else:
    result = np.hstack((p1,tie,p2))

cv2.imshow("result", result)
#cv2.imshow("result", np.hstack((p1,tie,p2)))

