# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:58:29 2021

@author: user
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 8 10:58:26 2021

"""

import cv2
import numpy as np
from skimage.feature import feature_hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split

def read(path, data_amount):
  data = list()
  for i in range(1, data_amount+1):
    image = cv2.imread(path % i)
    image = cv2.resize(image, (349,256))
    data.append(image)
  return data

def sift(data):
  output = list()
  for image in data:
    sift_feature = cv2.SIFT_create()
    kp, des = sift_feature.detectAndCompute(image, None)
    output.append(des)
  return output

def hog(data):
    output = list()
    for image in data:
        fd, hog_image = feature_hog(
        image,
        orientations=8,
        pixels_per_cell=(9,9),
        cells_per_block=(1,1),
        visualize=True,
        )
    return output
    
def kmeans_return_features(k, sift):
  descriptors = sift[0]
  for it in sift[1:]:
    descriptors = np.vstack((descriptors, it))  
  
  voc, train_variance = kmeans(descriptors,k,1)
  
  #features histogram
  im_features = np.zeros((len(sift), k), "float32")
  for i in range(len(sift)):
    words, distance = vq(sift[i], voc)
    for j in words:
      im_features[i][j] += 1
      
  return im_features    
  


