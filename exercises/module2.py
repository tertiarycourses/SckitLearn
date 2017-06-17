# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

# Module 2: Datasets

# Iris dataset
iris = datasets.load_iris()

# print(iris)
# print(iris.data)
# print(iris.target)
# print(iris.feature_names)

import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# i = 2
# j = 3

# x = iris.data[:,i]
# y = iris.data[:,j]

# plt.scatter(x,y,c=iris.target)
# plt.xlabel(iris.feature_names[i])
# plt.ylabel(iris.feature_names[j])
# plt.colorbar(ticks=[0,1,2])
# plt.show()

# Handwritten digits

#digits = datasets.load_digits()

#print(digits)
#print(digits.data.shape)
#print(digits.target)

#print(digits.images[0])
#plt.clf()
#plt.imshow(digits.images[1259],cmap=plt.cm.gray_r)
#print(digits.target[1259])
# fig = plt.figure(figsize=(12,12))
# for i in range(16*16):
#  	ax = fig.add_subplot(16,16,i+1)
#  	ax.imshow(digits.images[i],cmap=plt.cm.binary)
# plt.show()
# Ex: plot 16x16 images

# Olivetti Human Faces

#faces = datasets.fetch_olivetti_faces()
# plt.imshow(faces.images[100],cmap=plt.cm.bone)
# plt.show()

# fig = plt.figure(figsize=(12,12))
# for i in range(64):
# 	ax = fig.add_subplot(8,8, i+1)
# 	ax.imshow(faces.images[i],cmap=plt.cm.bone)
# plt.show()

# Boston Housing Price

#boston = datasets.load_boston()

#print(boston)
#print(boston.data)
#print(boston.target)

#print(boston.feature_names)
#index = 12
#x = boston.data[:,index]
#y = boston.target
#plt.scatter(x,y)
# plt.xlabel(boston.feature_names[index])
# plt.ylabel('Housing Price')
#plt.show()

# Randomize data
#indices = np.random.permutation(len(X))
#print(indices)
# X = X[indices]
# y = y[indices]

# X_train = X[:-20]
# y_train = y[:-20]

# X_test = X[-20:]
# y_test = y[-20:]

# a = np.array([1,2,3,4,5,6,])
# # print(a)
# indices = np.random.permutation(len(a))
# print(a[indices])

# print(y_test)

#from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#print(y_train)
#print(y_test)

#Exercise
# from sklearn.model_selection import train_test_split

# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# print(y_train)
