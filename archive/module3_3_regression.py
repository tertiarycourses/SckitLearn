# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3.3: Regression

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

# Challenge: Boston dataset

# boston = datasets.load_boston()
# X,y = boston.data, boston.target

# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.target.shape)

# Boston Housing Price Challnege
# from sklearn import datasets
# boston = datasets.load_boston()
# X,y = boston.data,boston.target

# from sklearn.preprocessing import scale
# X = scale(X)

# from sklearn import linear_model
# lm = linear_model.LinearRegression()
# lm.fit(X,y)

# import matplotlib.pyplot as plt 
# plt.scatter(y,lm.predict(X))
# plt.xlabel('Price')
# plt.ylabel('Predict Price')
# plt.show()

