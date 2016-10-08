from sklearn import datasets

digits = datasets.load_digits()
X,y = digits.data, digits.target

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)

pca.fit(X)

X_reduced = pca.transform(X)

import matplotlib.pyplot as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.show()