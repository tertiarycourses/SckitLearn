import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-10]]
y_train = y[indices[:-10]]
X_test  = X[indices[-10:]]
y_test  = y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train) 

print(clf.predict(X_test))
print(y_test)
