# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3: Model Persistence

# joblib Method

# from sklearn.externals import joblib
# clf = joblib.load('mymodel.pkl') 

# Testing
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)


# Pickle Method
import pickle
clf = pickle.load(open("mymodel2.pkl","rb"))

from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

print(clf.predict(X_test)[:20])
print(y_test[:20])
accuracy = clf.score(X_test,y_test)
print(accuracy)
