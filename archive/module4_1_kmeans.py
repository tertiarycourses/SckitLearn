# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.1: K-Means Clustering

import matplotlib.pyplot as plt
import numpy as np 

# from sklearn.datasets.samples_generator import make_blobs
# centers = [[1,1],[1.2,1.2],[2,2]]
# X,y = make_blobs(n_samples=100,centers=centers, cluster_std=0.1)

# plt.scatter(X[:,0],X[:,1])
# plt.show()

# Step 1 Model
# from sklearn import cluster
# cluster = cluster.KMeans(n_clusters=3)

# # Step 2 Training
# cluster.fit(X)

# Step 3 Evaluation
# plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
# plt.show()

# Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

# from sklearn.preprocessing import scale
# X = scale(iris.data)

# Load KMeans Model
from sklearn import cluster
cluster = cluster.KMeans(n_clusters=3, random_state=5)

# Training
cluster.fit(X) 

# Evaluation

# from sklearn import metrics
# print(metrics.accuracy_score(y,cluster.labels_))

# plt.subplot(1,2,1)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.title('Original')
# plt.subplot(1,2,2)
# plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
# plt.title('K Means Clustering')
# plt.show()


# Challenge: Handwritten Digits dataset
# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.data

# from sklearn import cluster
# cluster = cluster.KMeans(n_clusters=10)

# cluster.fit(X)

# print(cluster.labels_[::50])
# print(y[::50])






