# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 4.3: Hierachical Clustering

import matplotlib.pyplot as plt
import numpy as np 

# from sklearn.datasets.samples_generator import make_blobs
# centers = [[1,1],[1.1,1.1],[2,2]]
# X,y = make_blobs(n_samples=100,centers=centers, cluster_std=0.1)
# # plt.scatter(X[:,0],X[:,1])
# # plt.show()

# Step 1 Model
# from sklearn.cluster import AgglomerativeClustering
# cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# # Step 2 Training
# cluster.fit(X)

# # Step 3 Labeling
# import matplotlib.pyplot as plt
# plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
# plt.show()


# Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

# Step 1: Load Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# Step 2: Training
cluster.fit(X)

# Step 3 Evaluation
# from sklearn import metrics
# print(metrics.accuracy_score(y,cluster.labels_))

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title('Original')
plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=cluster.labels_)
plt.title('Agglomerative Clustering')
plt.show()

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



