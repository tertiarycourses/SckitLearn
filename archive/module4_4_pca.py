# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.4: Pricipal Component Analysis (PCA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition

# Artificial dataset
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_) 

# Iris dataset
iris = datasets.load_iris()
X,y = iris.data,iris.target

from sklearn.preprocessing import scale 
X = scale(X)

# Step 1: Load Model
from sklearn import decomposition
# pca = decomposition.PCA()
pca = decomposition.PCA(n_components=2)

# Step 2: PCA 
pca.fit(X)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# comps = pd.DataFrame(pca.components_, columns=iris.feature_names)
# print(comps)

# Step 3: Evaluation
X_t = pca.transform(X)
# plt.subplot(1,2,1)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.subplot(1,2,2)
# plt.scatter(X_t[:,0],X_t[:,1],c=y)
# plt.show()

from sklearn.cluster import MeanShift
cluster = MeanShift()

cluster.fit(X)
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=cluster.labels_)

cluster.fit(X_t)
plt.subplot(1,2,2)
plt.scatter(X_t[:,0],X_t[:,1],c=cluster.labels_)
plt.show()

# Challenge
# digits = datasets.load_digits()
# X,y = digits.data, digits.target	