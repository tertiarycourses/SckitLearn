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

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.predict(X_test)[:10])
print(y_test[:10])

with open("iris.dot", 'w') as f:
 	tree.export_graphviz(clf, out_file=f,feature_names=iris.feature_names, class_names=iris.target_names)