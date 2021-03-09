# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:57:34 2021

@author: User
"""

import cv2
import numpy as np 

img_old = cv2.imread("hw01.jpg")
w = int(img_old.shape[1] * 0.5)
h = int(img_old.shape[0] * 0.5)

def re(img):

    
    # resize 
    # (x, y) is tuple
    img2 = cv2.resize(img, (w, h), \
                      interpolation = cv2.INTER_CUBIC )
    #cv2.imshow("donburi", img2) 
    cv2.imwrite("./resize.jpg", img2)
    return img2

def gray(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    #cv2.imshow("donburi_gray", img2)
    cv2.imwrite("./gray.jpg", img2)
    return img2 

def HSV(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("donburi_HSV", img2)
    cv2.imwrite("./HSV.jpg", img2)
    return img2 

def YcrCb(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #cv2.imshow("donburi_YcrCb", img2)
    cv2.imwrite("./YcrCb.jpg", img2)
    return img2 

def RGBpass(img):
    cname = ["B", "G", "R"]
    for i in range(2,-1,-1):
        img2 = img[:, :, i]
        #cv2.imshow("pass" + cname[i], img2)
        
        img2 = np.zeros((h, w, 3), np.uint8)
        img2[:, :, i] = img[:, :, i]
        #cv2.imshow("look" + cname[i], img2)
        cv2.imwrite(cname[i] + ".jpg", img2)

#gray
img = re(img_old)
gray(img)
RGBpass(img)

#hsv
img = re(img_old)
img = HSV(img)

#YcrCb
img = re(img_old)
img = YcrCb(img)

#cv2.waitKey(0)
#cv2.destroyWindows()

