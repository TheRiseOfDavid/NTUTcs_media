import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import train_test_split

data_amount = 300

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
images_people = lfw_people['images'][:data_amount] #要注意後面的資料長度

min_size_H = 9999
min_size_W = 9999

cars = list()
for i in range( data_amount):
  fname = "../tzulin/car/%.5d.jpg" % (i+1) 
  car = cv2.imread(fname)
  min_size_H = min(min_size_H, car.shape[0])
  min_size_W = min(min_size_W, car.shape[1])
  car = cv2.resize(car, (37,50))
  cars.append(car)
cars = np.array(cars)



print("min_size_H", min_size_H)
print("min_size_W", min_size_W)

target_people = [0] * data_amount
target_car = [1] * data_amount
target = target_car + target_people

images = list()
for car in cars:
  images.append(car)
for people in images_people:
  images.append(people)
  
hog_images = list()
for image in images:
  out, hog_image = hog(image, 
                       orientations=8,
                       pixels_per_cell=(9,9),
                       cells_per_block=(1,1),
                       visualize=True,
#                       multichannel=True
               )
  hog_images.append(out)

# split some data to be the test data
x_train, x_test, y_train, y_test = train_test_split(hog_images,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0)

# svm train
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(x_train, y_train)

print("accuracy")
print("train:", clf.score(x_train, y_train))
print("test:", clf.score(x_test, y_test), "\n")



  


