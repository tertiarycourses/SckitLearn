# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.3: Hierachical Clustering

import matplotlib.pyplot as plt
import numpy as np 

# from sklearn.datasets.samples_generator import make_blobs

# centers = [[1,1],[1.5,1.5],[2,2]]

# X,y = make_blobs(n_samples=100,centers=centers, cluster_std=0.1)

# # plt.scatter(X[:,0],X[:,1])
# # plt.show()

# from sklearn.cluster import MeanShift
# clf = MeanShift()
# clf.fit(X)

# labels = clf.labels_
# c = clf.cluster_centers_

# print(c)

# n = len(np.unique(labels))
# print("number of clusters",n)

# colors = ["r.","b.","c.","b.","k."]

# for i in range(len(X)):
# 	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)

# plt.scatter(c[:,0],c[:,1],marker="x",s=150)

# plt.show()

# Exercise
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# from sklearn.cluster import MeanShift
# clf = MeanShift()
# clf.fit(X)

# labels = clf.labels_
# c = clf.cluster_centers_
# n = len(np.unique(labels))
# print("number of clusters",n)


# Agglomerative Clustering

# from sklearn.datasets.samples_generator import make_blobs
# centers = [[1,1],[1.5,1.5],[2,2]]
# X,y = make_blobs(n_samples=100,centers=centers, cluster_std=0.1)
# # plt.scatter(X[:,0],X[:,1])
# # plt.show()

# Step 1 Model
# from sklearn.cluster import AgglomerativeClustering
# clf = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# # Step 2 Training
# clf.fit(X)

# # Step 3 Labeling
# labels = clf.labels_
# print(labels)
# print(y)


from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
plt.figure(figsize=(10, 3))

plt.style.use('seaborn-whitegrid')

from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

k=3
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')
Hclustering.fit(X)
# print(sm.accuracy_score(y, Hclustering.labels_))

# import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, 'ward')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()



