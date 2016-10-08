import numpy as np 
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn import datasets

# Module 2 Datasets

# Iris Flower Dataset

# iris = datasets.load_iris()

# #print(iris.data)
# #print(iris.target)

# x = iris.data[:,2]
# y = iris.data[:,3]

# plt.scatter(x,y,s=100,c=iris.target)
# plt.colorbar(ticks=[0,1,2])
# plt.xlabel(iris.feature_names[2])
# plt.ylabel(iris.feature_names[3])
# plt.show()

# Handwritten Digit Dataset

# digits = datasets.load_digits()

#print(digits.data)
#print(digits.target)

#print(digits.images[1])
# plt.imshow(digits.images[1],cmap=plt.cm.binary)
# plt.show()

# fig = plt.figure(figsize=(10,10))
# for i in range(16*16):
# 	ax = fig.add_subplot(16,16,i+1)
# 	ax.imshow(digits.images[i],cmap=plt.cm.binary)
# plt.show()

# Olivetti Faces Dataset

# faces = datasets.fetch_olivetti_faces()
# # plt.imshow(faces.images[0],cmap=plt.cm.bone)
# # plt.show()

# fig = plt.figure(figsize=(10,10))
# for i in range(100):
# 	ax = fig.add_subplot(10,10,i+1)
# 	ax.imshow(faces.images[i],cmap=plt.cm.bone)
# plt.show()

# Boston Housing Price Dataset

# boston = datasets.load_boston()

#print(boston.data)
#print(boston.target)

# for i in range(13):
# 	x = boston.data[:,i]
# 	y = boston.target
# 	plt.figure()
# 	plt.scatter(x,y,s=100)
# 	plt.xlabel(boston.feature_names[i])
# 	plt.ylabel('Housing Price')
# plt.show()

# Splitting data into Training/Testing

# from sklearn import datasets
# iris = datasets.load_iris()

# X,y = iris.data,iris.target

# Method 1: Manual randomization

# indices = np.random.permutation(len(X))

# X_train = X[indices[0:-20]]
# y_train = y[indices[0:-20]]

# X_test = X[indices[-20:]]
# y_test = y[indices[-20:]]

# Randominzation

# a = np.array([1,2,3,4,5,6])
# indices = np.random.permutation(len(a))
# print(indices)

# # Method 2 Randomization
# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# print(y_test)

# Module 3 Supervised Learning

# from sklearn import datasets
# iris = datasets.load_iris()
# #digits = datasets.load_digits()

# #X,y = iris.data,iris.target
# X,y = digits.data,digits.target

# from sklearn.cross_validation import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# KNN Classifier

# Step1 : Choose Model
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(n_neighbors=5)

# # Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:10])
# print(y_test[:10])


# SVM Classifier

# # Step1 : Choose Model
# from sklearn import svm
# clf = svm.SVC()

# # Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:10])
# print(y_test[:10])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)


# SGD Classifier

# Step1 : Choose Model
# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

# # Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:10])
# print(y_test[:10])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)

# Navie Bayes Classifier

# Step1 : Choose Model
# from sklearn import naive_bayes
# clf = naive_bayes.GaussianNB()

# # Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)

# Decision Tree Classifier

# Step1 : Choose Model
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

# # Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# accuracy = clf.score(X_test,y_test)
# print(accuracy)

# Logistics Regression Classifer

# Step1 : Choose Model
# from sklearn import linear_model
# clf = linear_model.LogisticRegression()

# Step 2: Trainig

# clf.fit(X_train,y_train)

# # Step 3 Testing
# print(clf.predict(X_test)[:20])
# print(y_test[:20])

# # matches = clf.predict(X_test)==y_test
# # score = matches.sum()/float(len(matches))
# # print(score)

# accuracy = clf.score(X_test,y_test)
# print(accuracy)

# Regression

# Linear Regression

# X = np.linspace(1,20,100).reshape(-1,1)
# y = X + np.random.normal(0,1,100).reshape(-1,1)

# plt.scatter(X,y)

# boston = datasets.load_boston()

# X = boston.data
# y = boston.target

# X_train = X[:-20]
# y_train = y[:-20]

# # Step1 : Choose Model
# from sklearn import linear_model
# lm = linear_model.LinearRegression()

# # # Step 2: Trainig/Fitting
# lm.fit(X_train,y_train)

#Step 3 Predition
# X_test = X[-20:]
# y_test = y[-20:]

# yp = lm.predict(X_test)
# plt.scatter(y_test,yp,s=100)
# y2 = y_test
# plt.plot(y_test,y2,'r')
# plt.xlabel('Actual Hosing Price')
# plt.ylabel('Predicted Housing Price')
# plt.axis([0,30,0,30])
# plt.show()
# plt.plot(X,lm.predict(X),'-r')
# plt.show()

# print(clf.predict(X_test)[:20])
# print(y_test[:20])

# Ridge/Lasso/SGD/KNN Regression

# boston = datasets.load_boston()

# X = boston.data
# y = boston.target

# # Step1 : Choose Model
# #from sklearn import linear_model
# from sklearn import neighbors

# #lm = linear_model.Ridge()
# #lm = linear_model.Lasso()
# #lm = linear_model.SGDRegressor()
# lm = neighbors.KNeighborsRegressor()

# # Step 2: Trainig/Fitting
# lm.fit(X,y)

# #Step 3 Predition

# yp = lm.predict(X)
# plt.scatter(y,yp,s=100)
# plt.xlabel('Actual Hosing Price')
# plt.ylabel('Predicted Housing Price')
# #plt.axis([0,30,0,30])
# plt.show()

# print(clf.predict(X_test)[:20])
# print(y_test[:20])

# Module 4 Unsupervised Learning

# X = np.array([
# 	[1,2],
# 	[1.5,1.8],
# 	[5,8],
# 	[8,8],
# 	[1,0.6],
# 	[9,11]
# 	])

# plt.scatter(X[:,0],X[:,1],s=100)
# plt.show()

# # Step 1 : Load the Model
# from sklearn import cluster
# clf = cluster.KMeans(n_clusters=4)

# # Step 2: Training

# clf.fit(X)

# # Step 3: Prediction

# c = clf.cluster_centers_
# labels = clf.labels_

# colors = ["r.","b.","c.","k."]

# for i in range(len(X)):
# 	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
# plt.scatter(c[:,0],c[:,1],marker='x',s=150)	
# plt.show()

# Challenge

# digits = datasets.load_digits()

# X,y = digits.data,digits.target

# # Step 1 : Load the Model
# from sklearn import cluster
# clf = cluster.KMeans(n_clusters=10)

# # Step 2: Training

# clf.fit(X)

# # Step 3: Prediction

# c = clf.cluster_centers_
# labels = clf.labels_

# print(labels[::20])
# print(y[::20])

# colors = ["r.","b.","c.","k."]

# for i in range(len(X)):
# 	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
# plt.scatter(c[:,0],c[:,1],marker='x',s=150)	
# plt.show()

# Hierachical Clustering

# from sklearn.datasets.samples_generator import make_blobs

# c = [[1,1],[2,2],[1.2,1.2]]

# X,y = make_blobs(n_samples = 100, centers = c, cluster_std=0.05)

# plt.scatter(X[:,0],X[:,1],s=100)
# plt.show() 

# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# # Step 1 : Load the Model
# from sklearn import cluster

# clf = cluster.MeanShift()

# # Step 2: Training

# clf.fit(X)

# # Step 3: Prediction

# c = clf.cluster_centers_
# labels = clf.labels_

# colors = ["r.","b.","c.","k."]

# for i in range(len(X)):
#  	plt.plot(X[i][0],X[i][2],colors[labels[i]],markersize=25)
# #plt.scatter(c[:,0],c[:,1],marker='x',s=150)	
# plt.show()

# Dimensionality Reduction

# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# # Principal Component Analysis 

# # Step 1: Model
# from sklearn import decomposition
# pca = decomposition.PCA()
# pca.n_components = 2

# # Step 2: Training
# pca.fit(X)

# Step 3: Prediction

# X_reduced = pca.fit_transform(X)

# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
# plt.show()

# Module 5: Neural Network

iris = datasets.load_iris()

X,y = iris.data,iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# Step 1: Model
from sklearn import neural_network

clf = neural_network.MLPClassifier(2,10,30)

# # Step 2: Training

clf.fit(X_train,y_train)

# Step 3: Prediction

print(clf.predict(X_test[:10]))
print(y_test[:10])



