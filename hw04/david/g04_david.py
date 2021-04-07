from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = datasets.load_wine()

x = wine.data
#x = np.array([[d[0], d[1], d[2], d[3]]  for d in wine.data])
y = wine.target



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=0)

clf = svm.SVC(kernel='linear', C=1, gamma='auto')
#clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
clf.fit(x_train, y_train)


#print("predict")
#print(clf.predict(x_train))
#print(clf.predict(x_test))

print("accuracy")
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))
#plt.scatter(x_train[:,0], x_train[:,2])
#plt.show()
