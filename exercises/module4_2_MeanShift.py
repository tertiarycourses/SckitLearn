# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.2: Mean Shift Clustering

import matplotlib.pyplot as plt
import numpy as np 

# from sklearn.datasets.samples_generator import make_blobs
# centers = [[1,1],[1.5,1.5],[2,2]]
# X,y = make_blobs(n_samples=100,centers=centers, cluster_std=0.1)
# # plt.scatter(X[:,0],X[:,1])
# # plt.show()

# # Step 1 Model
# from sklearn.cluster import MeanShift
# cluster = MeanShift()

# # # Step 2 Training
# cluster.fit(X)

# # Step 3 Evaluation
# plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
# plt.show()


# Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

# from sklearn.preprocessing import scale
# X = scale(iris.data)

# # Step 1 Model
from sklearn.cluster import MeanShift
cluster = MeanShift()

# # Step 2 Training
cluster.fit(X)

# # Step 3 Evaluation
# from sklearn import metrics
# print(metrics.accuracy_score(y,cluster.labels_))

# plt.subplot(1,2,1)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.title('Original')
# plt.subplot(1,2,2)
# plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
# plt.title('Mean Shift Clustering')
# plt.show()
