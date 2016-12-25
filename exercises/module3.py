# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3: Supervised Learning

# Classification

# Load data and split data
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# Step 1: Load the classifer

# K Nearest Neighbors Classificaiton
#from sklearn import neighbors
#clf = neighbors.KNeighborsClassifier()
#clf = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree')

# Support Vector Machine
# from sklearn import svm
# clf = svm.SVC()
#clf = svm.SVC(kernel='rbf',degree=3)

# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

# from sklearn import naive_bayes

# clf = naive_bayes.GaussianNB()

# from sklearn import tree

# clf = tree.DecisionTreeClassifier()

# Step 2: Training

# clf.fit(X_train,y_train)

# Step 3: Testing

# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# score = clf.score(X_test,y_test)
# print(score)

# Step 3: Measure the Performance

# predicted = clf.predict(X_test)
# expected = y_test
# matches = (predicted == expected)

# score = matches.sum()/len(matches)
# print("Score = ", score)

# from sklearn import metrics
# print(metrics.classification_report(expected, predicted))

#print(clf.predict([[1.2,2.5,3,4.5]]))

# Model Persistence
# from sklearn import datasets

# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# from sklearn import svm
# clf = svm.SVC()
# clf.fit(X_train, y_train)

# from sklearn.externals import joblib
#joblib.dump(clf, 'mymodel.pkl') 
# clf = joblib.load('mymodel.pkl') 

# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)

# Regression

# Create a simple dataset
# import numpy as np
# X = np.linspace(1,20,100).reshape(-1,1)
# y = X + np.random.normal(0,1,100).reshape(-1,1)

#import matplotlib.pyplot as plt 
# plt.scatter(X,y)
# plt.show()

# from sklearn import linear_model
# lm = linear_model.LinearRegression()
# lm.fit(X, y) 

# plt.scatter(X,y)
# plt.plot(X,lm.predict(X),'-r')
# plt.show()

# Challenge

# boston = datasets.load_boston()
# X,y = boston.data, boston.target

# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.target.shape)

# Boston Housing Price Challnege
from sklearn import datasets
boston = datasets.load_boston()
X,y = boston.data,boston.target

# from sklearn import linear_model
# lm = linear_model.LinearRegression()
# lm.fit(X,y)

# import matplotlib.pyplot as plt 
# plt.scatter(y,lm.predict(X))
# plt.xlabel('Price')
# plt.ylabel('Predict Price')
# plt.show()

