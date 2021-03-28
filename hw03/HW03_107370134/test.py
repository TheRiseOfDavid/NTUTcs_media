import cv2
import numpy as np

img = cv2.imread("./coin.jpg")
h, w, ch = img.shape
print(h, w)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(img, cv2.CV_16S, 1, 0)
img = cv2.resize(img, (w//5, h//5), interpolation=cv2.INTER_NEAREST) #important

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
sobel = cv2.Sobel(img, cv2.CV_16S, 1, 0)
scharr = cv2.Scharr(img, cv2.CV_64F, 1, 0)
laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)
canny = cv2.Canny(img, 50, 150)

cv2.imshow("canny", canny)
cv2.imshow("laplacian", laplacian)
cv2.imshow("scharr", scharr)
cv2.imshow("sobel", sobel)