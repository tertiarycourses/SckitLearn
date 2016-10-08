import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

from sklearn import decomposition

pca = decomposition.PCA()
pca.fit(X)
X_reduced = pca.transform(X)
print(pca.explained_variance_)

pca.n_components = 2
X_reduced = pca.fit_transform(X)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)
plt.show()
