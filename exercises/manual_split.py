from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X,y = iris.data,iris.target


X_train = X[:-20]
y_train = y[:-20]
X_test = X[-20:]
y_test = y[-20:]

print(y_test)