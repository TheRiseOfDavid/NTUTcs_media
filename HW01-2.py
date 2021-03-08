# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:53:57 2021

@author: User
"""

import cv2
import numpy as np 

img = np.zeros((400, 400, 3), np.uint8)
img.fill(200)

cv2.line(img, (16, 16), (100, 100), (240, 169, 169), 8)
cv2.rectangle(img, (200, 200), (232, 232), (154, 224, 253), -1)
cv2.rectangle(img, (100, 100), (132, 132), (194, 109, 165), 8)
cv2.circle(img, (300, 100), 80, (118, 122, 27), 3)
cv2.circle(img, (100, 300), 16, (135, 133, 255), -1)

#cv2.imshow("hello world", img)
cv2.imwrite("picture.jpg", img)
