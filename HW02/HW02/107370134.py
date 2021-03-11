import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('circle.jpg',0)

ret, circle1 = cv2.threshold(img1 ,127,255,cv2.THRESH_OTSU)
circle2 = cv2.dilate(circle1,np.ones((3,3)),iterations = 7)
circle3 = cv2.erode(circle2,np.ones((3,3)),iterations = 7)
cv2.imshow("circle",circle3)
cv2.waitKey(0)

img2 = cv2.imread('man.jpg',0)
ret, man1 = cv2.threshold(img2 ,127,255,cv2.THRESH_BINARY_INV)
man2 = cv2.dilate(man1,np.ones((3,3)),iterations =3)
man4=cv2.bitwise_and(man2,img2)

cv2.imshow("man",man4)
cv2.waitKey(0)
