import cv2
import numpy as np

# from matplotlib import pyplot as plt

img = cv2.imread('floor.jpg',1)

img1 = cv2.imread('floor.jpg',0)

#cv2.imshow("1",img1)

img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA)

img1 = cv2.resize(img1, (int(img1.shape[1]/5), int(img1.shape[0]/5)), interpolation=cv2.INTER_AREA)

#cv2.imshow("1",img)

ret, floor1 = cv2.threshold(img1 ,90,255,cv2.THRESH_BINARY)

floor1 = cv2.GaussianBlur(floor1,(5,5),0)

floor1 = cv2.Canny(floor1,0,255)
#coin1 = cv2.dilate(coin1,np.ones((3,3)),iterations = 2)
cv2.imshow('oginal', floor1)
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 5
max_line_gap = 5


lines = cv2.HoughLinesP(floor1,rho,theta,threshold,np.array([]),
                            min_line_length,max_line_gap)
print(lines)
for i in lines:
    cv2.line(img,(i[0][0],i[0][1]),(i[0][2],i[0][3]),(0,0,255),2)

cv2.imwrite('output2.jpg',img)
cv2.imshow('output',img)
cv2.waitKey()
