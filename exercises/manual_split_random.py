from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X,y = iris.data,iris.target

indices = np.random.permutation(len(X))

X_train = X[indices[:-20]]
y_train = y[indices[:-20]]
X_test = X[indices[-20:]]
y_test = y[indices[-20:]]

print(X_train)
print(y_test)