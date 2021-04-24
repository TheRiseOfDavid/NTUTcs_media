# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:46:55 2021

@author: User
"""

import os 
import cv2

def dog():
    for i in range(1,101):
        print(i)
        image = cv2.imread("./dog/images (%d).jfif" % i)
        image = cv2.resize(image,(100,100))
        
        cv2.imwrite("./data/images_%d.jpg" % i, image)    

def cat():
    for i in range(101,201):
        print(i)
        image = cv2.imread("./cat/images (%d).jfif" % (i-100))
        image = cv2.resize(image,(100,100))
        
        cv2.imwrite("./data/images_%d.jpg" % i, image)

dog()
cat()
