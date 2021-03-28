# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:28:28 2021

@author: user
"""

import cv2
import numpy as np

a = cv2.imread("./pic/apple.jpg")
b = cv2.imread("./pic/orange.jpg")

gaussA = [a.copy()]
gaussB = [b.copy()]
copyA = a.copy()
copyB = b.copy()
for i in range(6): # more small
  copyA = cv2.pyrDown(copyA)
  copyB = cv2.pyrDown(copyB)
  
  gaussA.append(copyA)
  gaussB.append(copyB)

copyA = gaussA[5]
laplacianA = [copyA]
copyB = gaussB[5]
laplacianB = [copyB]

for i in range(5, 0, -1): #more big
  tempA = cv2.pyrUp(gaussA[i])
  tempB = cv2.pyrUp(gaussB[i])

  subA = cv2.subtract(gaussA[i-1], tempA)
  subB = cv2.subtract(gaussB[i-1], tempB)
  laplacianA.append(subA)
  laplacianB.append(subB)
  #cv2.imshow("subA", subA)
  #cv2.waitKey(0)

stack = []

for itA, itB in zip(laplacianA, laplacianB): #merge
  h, w, ch = itA.shape
  merge = np.hstack((itA[0:h, 0:(w // 2)], itB[0:h, (w // 2):w]))
  stack.append(merge)
  
result = stack[0]
for i in range(1, 6): #stack picture
  result = cv2.pyrUp(result)
  result = cv2.add(result, stack[i])

cv2.imshow("a", a)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.imwrite('result.jpg',result)
