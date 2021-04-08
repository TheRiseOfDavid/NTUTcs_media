# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:24:14 2021

@author: User
"""

import cv2
import numpy as np

def show(img):
  cv2.imshow("img", img)
  cv2.waitKey(0)  

#input
img = cv2.imread("./pic/coin.jpg")
size = img.shape
print(size) #225,400

#resize
w = 800 
h = 450
re = cv2.resize(img, (w, h), \
                  interpolation = cv2.INTER_CUBIC )
    
#gray 
gray = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)

#binary
ret, binary = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)

#blur
blur2 = cv2.GaussianBlur(binary, (5, 5), 0)

#膨脹 or 侵蝕
er = cv2.erode(blur2,np.ones((3, 3)),iterations = 1) #侵蝕
di = cv2.dilate(er,np.ones((3, 3)),iterations = 5) #膨脹
er2 = cv2.erode(er,np.ones((3, 3)),iterations = 7) #侵蝕

#edge
edge = cv2.Canny(er2, 30, 150)

#hough
circle = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 100, param1 = 100,\
                          param2 = 30, minRadius = 100, maxRadius = 200) 

(cnts, _)= cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,\
                               cv2.CHAIN_APPROX_SIMPLE)
cntLabel, label, stat, centroids = cv2.connectedComponentsWithStats(edge, \
                                                                    connectivity = 8)
    
show(blur2)
show(di)
show(edge)
#print("coin count", len(cnts))
print("cnts_label = ", cntLabel)
#print("label = ", label)
#print("stat = ", stat)



