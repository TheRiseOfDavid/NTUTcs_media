import cv2
import numpy as np

img = cv2.imread("../pic/coin2.jpg")
h, w, ch = img.shape

img = cv2.resize(img, (w//5, h//5), interpolation=cv2.INTER_NEAREST) #important

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

gray = cv2.erode(gray, np.ones((3,3)), iterations=3)
gray = cv2.dilate(gray, np.ones((3,3)), iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)
print(stats)

cnt = 0
ans = 0

for it in stats:
  itX = it[0]+it[2]
  itY = it[1]+it[3]
  itArea = it[2]*it[3]
  print(itArea)
#  cv2.rectangle(img, (it[0], it[1]), (itX, itY), (200, 31, 31), 2)

  if(itArea >= 1600 and itArea < 2000): #1 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 0, 255), 2) #BGR 
    ans += 1
  if(itArea >= 2000 and itArea < 2400): #5 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 165, 255), 2) #BGR
    ans += 5
  if(itArea >= 3000 and itArea < 3400): #10 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 255, 255), 2) #BGR 
    ans += 10
  if(itArea >= 3600 and itArea < 4100): #50 dallars  
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (0, 128, 0), 2) #BGR
    ans += 50
  if(itArea >= 55000 and itArea < 57000): #100 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (200, 31, 31), 2)
    ans += 100
  if(itArea >= 53000 and itArea < 55000): #500 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (128, 0, 128), 2)
    ans += 500
  if(itArea >= 57000 and itArea < 59000): #1000 dallars
    cv2.rectangle(img, (it[0], it[1]), (itX, itY), (255, 255, 255), 2)
    ans += 1000

print("圖上共有 ",ans ,"元")
cv2.imshow("gray", gray)  
cv2.imshow("img", img)

cv2.imwrite('ans3-3.jpg', img)
cv2.imwrite('gray3-3.jpg', gray)
cv2.waitKey(0)


