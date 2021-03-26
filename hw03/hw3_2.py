#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:07:19 2021

@author: kuotzulin
"""

import cv2
import numpy as np

# hw3_2: 利用霍夫線標磁磚邊緣
img_floor = cv2.imread("./pic/floor.jpg")
w = int(img_floor.shape[1]*0.2)
h = int(img_floor.shape[0]*0.2)
kernel = np.ones((2,2), np.uint8)

# 調整大小
resize_f = cv2.resize(img_floor, (w, h), \
                      interpolation = cv2.INTER_CUBIC )
# 灰階
gray_f = cv2.cvtColor(resize_f, cv2.COLOR_BGR2GRAY)
# 二值化
ret, thres_f = cv2.threshold(gray_f, 90, 255, cv2.THRESH_BINARY_INV)
# 侵蝕
erode_f = cv2.erode(thres_f, kernel, iterations=1)
# 膨脹
dilation_f = cv2.dilate(erode_f, kernel, iterations=1)

# 霍夫直線偵測
rho = 1
theta = np.pi/2
lines = cv2.HoughLines(dilation_f, rho, theta, 200)
# 畫出直線
for i in range(len(lines)):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(resize_f, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
# cv2.imshow("thres_f", thres_f)
# cv2.imshow("dilate_f", dilation_f)
# cv2.imshow("dilation_f", dilation_f)
cv2.imshow("result", resize_f)
cv2.imwrite("hw3_2.jpg", resize_f)

cv2.waitKey(0)

# 困難：一開始不知道 hough line是偵測白色的部分，所以畫出來的線很多又雜。
# 嘗試很久才發現要改為黑白相反，後來把二值化改為BINARY_INV就可以了。
# HoughLines是存(r,θ)值，不太確定如何轉換為直角坐標，就上網查了資料，找到上面的解法。