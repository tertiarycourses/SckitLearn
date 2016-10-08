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

# classifier
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y)


# classifier
from sklearn import naive_bayes

clf = naive_bayes.GaussianNB()
clf.fit(X_train, y_train)
# print(clf.predict(X_test))
# print(y_test)

predicted = clf.predict(X_test)
expected = y_test
matches = (predicted == expected)

score = matches.sum()/float(len(matches))
print("Score = {}".format(score))


