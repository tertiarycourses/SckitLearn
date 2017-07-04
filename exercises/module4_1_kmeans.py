# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.1: K-Means Clustering

import matplotlib.pyplot as plt
import numpy as np 

# X = np.array([
# 	[1,2],
# 	[1.5,1.8],
# 	[5,8],
# 	[8,8],
# 	[1,0.6],
# 	[9,11]
# 	])

# plt.scatter(X[:,0],X[:,1])
# plt.show()

# X = np.array([
# 	[1,2],
# 	[1.5,1.8],
# 	[5,8],
# 	[8,8],
# 	[1,0.6],
# 	[9,11]
# 	])

#plt.scatter(X[:,0],X[:,1])
#plt.show()

# Step 1 Model
# from sklearn import cluster
# clf = cluster.KMeans(n_clusters=2)

# Step 2 Training
# clf.fit(X)

# Step 3 Labeling
# centriods = clf.cluster_centers_
# labels = clf.labels_
# print(labels)

# colors = ["r.","b.","c.","y.","k."]

# for i in range(len(X)):
# 	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25) 
# plt.scatter(centriods[:,0],centriods[:,1],marker='x',s=150,)
# plt.show()

# Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

# from sklearn.preprocessing import scale
# X = scale(iris.data)

from sklearn import cluster
c = cluster.KMeans(n_clusters=3, random_state=5)
c.fit(X) 

print(c.labels_[::10])
print(y[::10])


import pandas as pd
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[c.labels_], s=50)
plt.title('K-Means Classification')
plt.show()

# Digits dataset
# from sklearn import datasets
# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# from sklearn import cluster
# c = cluster.KMeans(n_clusters=10)
# c.fit(X) 

# print(c.labels_[::50])
# print(y[::50])







