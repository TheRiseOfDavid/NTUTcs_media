import cv2 as cv
from random import randrange

i = randrange(1,201)




image = cv.imread("./data/images_%d.jpg" % i)
cv.imshow('rand',image)

print(i)
