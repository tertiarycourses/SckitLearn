from sklearn import preprocessing

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#print(X)

