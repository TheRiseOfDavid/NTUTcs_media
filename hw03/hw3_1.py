
import cv2
import numpy as np

# hw3_1: 框硬幣
img_coin = cv2.imread("./pic/coin.jpg")
w = 800
h = 450
kernel = np.ones((2,2), np.uint8)

# 調整大小
resize_c = cv2.resize(img_coin, (w, h), \
                      interpolation = cv2.INTER_CUBIC )
# 灰階
gray_c = cv2.cvtColor(resize_c, cv2.COLOR_BGR2GRAY)
# 二值化
ret, thres_c = cv2.threshold(gray_c, 90, 255, cv2.THRESH_BINARY)
# 侵蝕
erode_c = cv2.erode(thres_c, kernel, iterations=2)
# 膨脹
dilation_c = cv2.dilate(erode_c, kernel, iterations=1)
# 連通物件 stats[x, y, h, w, area]
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation_c, connectivity=8)
count = 0
sum = 0 # 金額總和初值
# 依照框的長來辨別硬幣，並用rectangle框起來
for i in range(num_labels):
    if 60 < stats[i][2] <= 80:  # one dollar
        cv2.rectangle(resize_c, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 0, 255), 2)
        print('1: stats = ',stats[i])
        print('centroids = ',centroids[i])
        count += 1 
        sum += 1
    if 80 < stats[i][2] <= 90:  # five dollar
        cv2.rectangle(resize_c, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 128, 255), 2)
        print('5: stats = ',stats[i])
        print('centroids = ',centroids[i])
        count += 1
        sum += 5
    if 90 < stats[i][2] <= 100:  # ten dollar
        cv2.rectangle(resize_c, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 255, 255), 2)
        print('10: stats = ',stats[i])
        print('centroids = ',centroids[i])
        count += 1
        sum += 10
    if 100 < stats[i][2] <= 110:  # fifty dollar
        cv2.rectangle(resize_c, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 255, 0), 2)
        print('50: stats = ',stats[i])
        print('centroids = ',centroids[i])
        count += 1
        sum += 50

# print("count = ", count)        
# cv2.imshow("erode", erode_c)
# cv2.imshow("dilation", dilation_c)
print("sum = ", sum)
cv2.imshow("result", resize_c)
cv2.imwrite("hw3_1.jpg", resize_c)

cv2.waitKey(0)
cv2.destroyWindow()

# 困難：測試二值化、膨脹和侵蝕等的數值花了些時間，還有判斷不同硬幣的大小也是透過print出stats的數值微小差異才成功