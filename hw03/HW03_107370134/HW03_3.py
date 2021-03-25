import cv2
import numpy as np

# from matplotlib import pyplot as plt

img = cv2.imread('coin2.jpg',1)

img1 = cv2.imread('coin2.jpg',0)

#cv2.imshow("1",img1)

img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA)

img1 = cv2.resize(img1, (int(img1.shape[1]/5), int(img1.shape[0]/5)), interpolation=cv2.INTER_AREA)

#cv2.imshow("1",img)

ret, coin1 = cv2.threshold(img1 ,45,255,cv2.THRESH_BINARY)

coin1 = cv2.medianBlur(coin1,3)
coin1 = cv2.erode(coin1,np.ones((3,3)),iterations = 7)

coin1 = cv2.dilate(coin1,np.ones((3,3)),iterations = 3)

coin1 = cv2.medianBlur(coin1,3)
coin1 = cv2.dilate(coin1,np.ones((3,3)),iterations = 3)





num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(coin1, connectivity=8)
cv2.imshow("circle",coin1)
#cv2.waitKey(0)
print(num_labels)

print(stats)

print(centroids)

print(labels)

#output = np.zeros((img1.shape[0]/5, img1.shape[1], 3), np.uint8)
sum = 0
for i in range(1, num_labels):

    if(stats[i][2] * stats[i][3] < 1800):
        sum +=1
        color = (0,0,255)
    elif (stats[i][2] * stats[i][3] < 2200):
        sum += 5
        color = (0,165,255)
    elif (stats[i][2] * stats[i][3] < 3200):
        sum += 10
        color = (0,255,255)
    elif (stats[i][2] * stats[i][3] < 3900):
        sum += 50
        color = (0,255,0)
    elif (stats[i][2] * stats[i][3] < 54883):
        sum += 500
        color = (240,32,160)
    elif (stats[i][2] * stats[i][3] < 55611):
        sum += 100
        color = (255,0,0)
    else:
        sum += 1000
        color = (255,0,255)
    output = cv2.rectangle(img,(stats[i][0]-2,stats[i][1]-2),(stats[i][0]+stats[i][2]+2,stats[i][1]+stats[i][3]+2),color,2)
    #output[:, :, 0][mask] = np.random.randint(0, 255)
    #output[:, :, 1][mask] = np.random.randint(0, 255)
    #output[:, :, 2][mask] = np.random.randint(0, 255)
print("硬幣總和為: " ,sum)
cv2.imwrite('output3.jpg',output)
cv2.imshow('oginal', output)
cv2.waitKey()

#img2 = cv2.imread('man.jpg',0)
#ret, man1 = cv2.threshold(img2 ,127,255,cv2.THRESH_BINARY_INV)
#man2 = cv2.erode(man1,np.ones((3,3)),iterations = 3)
#man3 = cv2.GaussianBlur(man2,(15,15),0)
#man4=cv2.bitwise_and(img2,man3)

#cv2.imshow("man",man4)
#cv2.waitKey(0)
