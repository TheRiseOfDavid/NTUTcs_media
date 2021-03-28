import cv2
import numpy as np

img = cv2.imread("../pic/floor.jpg")
h, w, ch = img.shape
w //= 5
h //= 5
print(h)

img = cv2.resize(img, (w,h), interpolation = cv2.INTER_NEAREST)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,1)
ret, gray = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

gray = cv2.erode(gray, np.ones((2,2)), iterations=1)
gray = cv2.dilate(gray, np.ones((2,2)), iterations=1)

#cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imwrite("gray3-2.jpg", gray)
# cv2.waitKey(0)

rho = 1.0
theta = np.pi / 2
line = cv2.HoughLines(gray, rho, theta, 200)

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
  
cv2.imshow("result", img)
cv2.imwrite("result3-2.jpg", img)
cv2.waitKey(0)



