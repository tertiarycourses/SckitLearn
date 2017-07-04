# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

import matplotlib.pyplot as plt
import numpy as np 

# Module 4: Unsupervised Learning

# Clustering

# Hierachical Clustering

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

# Pricipal Component Analysis

# from sklearn import decomposition
# pca = decomposition.PCA()

# pca.fit(X)
# pca.n_components = 2 

# X_reduced = pca.fit_transform(X)

# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
# plt.show()
#print(pca.explained_variance_)


# digits = datasets.load_digits()

# X,y = digits.data, digits.target

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# print(y_test)

# iris = datasets.load_iris()

# X,y = iris.data, iris.target

# Module 5: Intro to Neural Networks

# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# from sklearn import neural_network
# clf = neural_network.MLPClassifier(2,10,30)

# clf.fit(X_train,y_train)

# clf.predict(X_test)
# print(y_test)




