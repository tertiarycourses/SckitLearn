from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

from sklearn import cluster

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X) 
print(k_means.labels_[::10])
print(y[::10])