import cv2
import numpy as np

img = cv2.imread("../pic/coin2.jpg")
h, w, ch = img.shape

img = cv2.resize(img, (w//5, h//5), interpolation=cv2.INTER_NEAREST) #important

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

gray = cv2.erode(gray, np.ones((2,2)), iterations=2)
gray = cv2.dilate(gray, np.ones((2,2)), iterations=1)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)

cnt = 0
ans = 0

for it in stats:
  itX = it[0]+it[2]
  itY = it[1]+it[3]
  #cv2.rectangle(img, (it[0], it[1]), (itX, itY), (200, 31, 31), 2)

  if(it[2] >= 60 and it[2] < 75): #1 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 0, 255), 2) #BGR 
    ans += 1
  if(it[2] >= 75 and it[2] < 90): #5 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 165, 255), 2) #BGR
    ans += 5
  if(it[2] >= 90 and it[2] < 100): #10 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 255, 255), 2) #BGR 
    ans += 10
  if(it[2] >= 100 and it[2] < 115): #50 dallars  
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 128, 0), 2) #BGR
    ans += 50

print("圖上共有 ",ans ,"元")
cv2.imshow("gray", gray)  
cv2.imshow("img", img)

cv2.imwrite('ans.jpg', img)
cv2.imwrite('gray.jpg', gray)
cv2.waitKey(0)


