import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()

X = iris.data  
y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-10]]
y_train = y[indices[:-10]]
X_test  = X[indices[-10:]]
y_test  = y[indices[-10:]]


clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)

print(clf.predict(X_test)[:10])
print(y_test[:10])
