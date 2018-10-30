# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 2: Datasets

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

# Iris dataset
# iris = datasets.load_iris()

# print(iris)
# print(iris.data)
# print(iris.target)
# print(iris.feature_names)

# import pandas as pd
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())


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

# Boston Housing Price

# boston = datasets.load_boston()
# X,y = boston.data, boston.target
# print(boston)
# print(boston.data)
# print(boston.target)

# fig = plt.figure(figsize=(8,8))
# for i in range(13):
# 	ax = fig.add_subplot(4,4,i+1)
# 	plt.scatter(X[:,i],y)
# 	plt.xlabel(boston.feature_names[i])
# 	plt.ylabel('Housing Price')
# plt.show()


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#print(y_train)
#print(y_test)

#Exercise
# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)



