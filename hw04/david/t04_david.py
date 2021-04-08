from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#test_size 越小，準確率越高，越大則準確率越低
#在learning_rate 越大，感覺越好，n_estimators注意不要 overfitting
clf = AdaBoostClassifier(n_estimators=40, learning_rate=1, random_state=0)
clf.fit(x_train, y_train)
print("AdaBoost accuracy")
print(clf.score(x, y))
print()

#感覺不到差異
neigh = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance')
neigh.fit(x_train, y_train)
print("KNN accuracy")
print(neigh.predict(x_test))

xx = neigh.predict(x_test)
cnt = 0
for i in range(0, len(x_test)):
  if(xx[i] == y_test[i]):
    cnt+=1
print("Accuracy", cnt/len(x_test))

