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

# create the linear model classifier
from sklearn import linear_model

clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
print(y_test)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
#print(iris.target_names[model.predict([[3, 5, 4, 2]])])


