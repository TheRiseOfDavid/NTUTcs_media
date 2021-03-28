import cv2
import numpy as np


img = cv2.imread("picture.jpg")
h, w, ch = img.shape


rho = 1.0
theta = np.pi / 2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("hey", gray) 
line = cv2.HoughLines(gray, rho, theta, 70)
circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 70, param1=100, param2=30, \
                          minRadius=10, maxRadius=100)
c = np.uint16(np.around(circle))
print(circle)

print(c[0][0])
cv2.circle(img, (c[0][0][0], c[0][0][1]), c[0][0][2], (0,255,0), 2)
for it in line:
  for rho, theta in it:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + h*(-b))
    y1 = int(y0 + h*(a))
    x2 = int(x0 - h*(-b))
    y2 = int(y0 - h*(a))
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

   
cv2.imshow("img", img)