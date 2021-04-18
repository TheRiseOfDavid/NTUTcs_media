# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:17:58 2021

@author: user
"""

import numpy as np
import cv2

img = cv2.imread("./floor.jpg")
cv2.imshow('ori', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

#blocksize 點越小，越沒辦法判斷轉角
#ksize 最大只能等於 blocksize
#ksize 在使用 gauss-sobel 此公式時的大小，如果越大則細節處理越不精細
#ksize 如果跟 blocksize 一樣大就只做一次 blocksize，越小做越多 blocksize
#k 越大則轉角必須要越明顯，才可以知道，反之越小時，只需要小轉角就可以判斷
dst = cv2.cornerHarris(gray,5,1,0.1)
dst = cv2.dilate(dst, None)
img[dst>0.01*dst.max()] = [0,0,255]
cv2.imshow('result', img)

