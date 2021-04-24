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
from skimage.feature import hog as feature_hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
import os 
import random

def path_check(path):
    if(path[-1] != '/'):
        path += '/'
    return path
        
def rename_image(folder, category):
    folder = path_check(folder)
    images = os.listdir(folder) 
    i = 1
    
    for old_name in images:
        new_name = "%s-%d.jpg" % (category, i)
        if(not os.path.isfile(folder+new_name)):
            os.rename(folder+old_name, folder+new_name)
            print("修改當前檔案", old_name, new_name)
        i += 1

def read(path):
  data = list()
  images = os.listdir(path)
  for image in images:
    #print("正在讀取", path + image)
    image = cv2.imread(path + image) #必須是相對路徑
    #image = cv2.resize(image, (2375,3137))
    image = cv2.resize(image, (237,313))
    data.append(image)
  return data

def sift(data):
  output = list()
  sift_feature = cv2.SIFT_create()
  for image in data:
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
        output.append(fd)
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
  
def random_image(folder):
    folder = path_check(folder)
    images = os.listdir(folder)
    random_filename = random.choice(images)
    image = cv2.imread(folder + random_filename)
    image = cv2.resize(image, (237,313))
    print(folder + random_filename)
    return image
    