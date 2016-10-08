import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

# Pricipal Component Analysis

# Neural Networks

iris = datasets.load_iris()
X,y = iris.data, iris.target

from sklearn import neural_network
clf = neural_network.MLPClassifier(2,10,30)

clf.fit(X_train,y_train)

clf.predict(X_test)
print(y_test)

# from sklearn import decomposition

# pca = decomposition.PCA()

#pca.fit(X)
# pca.n_components = 2 

# X_reduced = pca.fit_transform(X)

# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
# plt.show()
#print(pca.explained_variance_)




# Clustering

# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# print(X.shape)
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# plt.scatter(X[:,1],X[:,3])
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
# plt.show()

# Regression

# Data
# X = np.linspace(1,20,100).reshape(-1,1)
# y = X + np.random.normal(0,1,100).reshape(-1,1)

# boston = datasets.load_boston()
# X,y = boston.data, boston.target

# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.target.shape)

# Step 1: Model

# from sklearn import cluster 

# model = cluster.KMeans(n_clusters=10)

# from sklearn import neighbors

# lm = neighbors.KNeighborsRegressor()

#from sklearn import linear_model

#lm = linear_model.LinearRegression()

#lm = linear_model.Ridge()

#lm = linear_model.Lasso()

#lm = linear_model.BayesianRidge()

# Step 2: Training

# model.fit(X)

#lm.fit(X,y)

# Step 3: Predict

# print(model.labels_[-20:])
# print(y[-20:])

# plt.scatter(y,lm.predict(X))
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')

# plt.show()

# input = 100
# print('The prediction is', lm.predict(input))

# plt.scatter(X,y) # raw data 
# y_predicted = lm.predict(X)
# plt.plot(X,y_predicted,color='red',linestyle='-') # prediction
# plt.show()

# Classification

# Load data and split data
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# Step 1: Load the classifer

# from sklearn import neighbors

# clf = neighbors.KNeighborsClassifier()

# from sklearn import svm
# clf = svm.SVC()

# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

# from sklearn import naive_bayes

# clf = naive_bayes.GaussianNB()

# from sklearn import tree

# clf = tree.DecisionTreeClassifier()

# Step 2: Training

# clf.fit(X_train,y_train)

# Step 3: Testing

# print(clf.predict(X_test)[:20])
# print(y_test[:20])

# Step 3: Measure the Performance

# predicted = clf.predict(X_test)
# expected = y_test
# matches = (predicted == expected)

# score = matches.sum()/len(matches)
# print("Score = ", score)

# from sklearn import metrics
# print(metrics.classification_report(expected, predicted))

#print(clf.predict([[1.2,2.5,3,4.5]]))

# a = np.array([1,2,3,4,5,6,])
# print(a)
# indices = np.random.permutation(len(a))
# print(a[indices])

# digits = datasets.load_digits()

# X,y = digits.data, digits.target

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# print(y_test)

# iris = datasets.load_iris()

# X,y = iris.data, iris.target

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# print(y_test)

#indices = np.random.permutation(len(X))
#print(indices)
# X = X[indices]
# y = y[indices]

# X_train = X[:-20]
# y_train = y[:-20]

# X_test = X[-20:]
# y_test = y[-20:]

# print(y_test)

# Boston Housing Price

# boston = datasets.load_boston()

#print(boston.data)
#print(boston.target)

#print(boston.feature_names)
# index = 5
# x = boston.data[:,index]
# y = boston.target
# plt.scatter(x,y)
# plt.xlabel(boston.feature_names[index])
# plt.ylabel('Housing Price')
# plt.show()

# Olivetti Human Faces

#faces = datasets.fetch_olivetti_faces()
# plt.imshow(faces.images[1],cmap=plt.cm.bone)
# plt.show()

# fig = plt.figure(figsize=(12,12))
# for i in range(64):
# 	ax = fig.add_subplot(8,8, i+1)
# 	ax.imshow(faces.images[i],cmap=plt.cm.bone)
# plt.show()
# Handwritten digits

# digits = datasets.load_digits()

#print(digits.data.shape)
#print(digits.target)

#print(digits.images[0])
#plt.clf()
#plt.imshow(digits.images[3],cmap=plt.cm.gray_r)
# fig = plt.figure(figsize=(12,12))
# for i in range(16*16):
# 	ax = fig.add_subplot(16,16,i+1)
# 	ax.imshow(digits.images[i],cmap=plt.cm.binary)

# plt.show()
# Ex: plot 16x16 images
# Iris dataset

# iris = datasets.load_iris()

#print(iris.data)
#print(iris.target)
#print(iris.feature_names)

# i = 1
# j = 3

# x = iris.data[:,i]
# y = iris.data[:,j]

# plt.scatter(x,y,c=iris.target)
# plt.xlabel(iris.feature_names[i])
# plt.ylabel(iris.feature_names[j])
# plt.colorbar(ticks=[0,1,2])
# plt.show()


