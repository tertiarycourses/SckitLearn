# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016


# Module 4-2: Pricipal Component Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_) 

iris = datasets.load_iris()
X,y = iris.data,iris.target

#print(X[0:10,])
plt.scatter(X[:,0],y,c=y)
# plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# pca = decomposition.PCA()
# pca.fit(X)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)

# comps = pd.DataFrame(pca.components_, columns=iris.feature_names)
# print(comps)

# pca = decomposition.PCA(n_components=1)
# pca.fit(X)
# X = pca.transform(X)
# plt.scatter(X,y,c=y)

# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()


# Challenge
# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# clf = clf.fit(X_train, y_train)

# yhat = clf.predict(X_test)
# print(yhat[0:200:2])
# print(y_test[0:200:2])

# pca = decomposition.PCA()
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# clf = clf.fit(X_train, y_train)

# yhat = clf.predict(X_test)
# print(yhat[0:200:5])
# print(y_test[0:200:5])



