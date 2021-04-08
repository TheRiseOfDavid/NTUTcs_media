from sklearn import svm
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import json 

iris = datasets.load_iris()
print(iris.keys())
print(iris["feature_names"])
print(iris["1"])
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_test)
clf = svm.SVC(kernel="linear", C=1, gamma="auto")
clf.fit(x_train, y_train) 
#print(clf.predict(x_train))
#print(clf.predict(x_test))

#print("Accuracy")       
#print(clf.score(x_train, y_train))
#print(clf.score(x_test, y_test))



