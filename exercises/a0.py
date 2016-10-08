from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# import mysql.connector
# import json

#cnx = mysql.connector.connect(user='root',password='mingliang',database='school3',buffered=True)
# method 1
# f = open('test.csv','w')
# f.write('a')
# f.close()
# f.write('b')

# method 2 
# with open('test.csv','w') as f :
# 	f.write('c')
	
# f.write('d')

#f.write('Hello How are you')

# from sklearn import datasets
# import numpy as np
# import matplotlib.pyplot as plt

# #Clustering

# # iris = datasets.load_iris()
# # X,y = iris.data,iris.target

# # plt.figure()
# # plt.scatter(X[:,0],X[:,1],c=y)

# # from sklearn import decomposition

# # model = decomposition.PCA()

# # model.fit(X)
# # #print(model.explained_variance_)

# # model.n_components = 2
# # X_reduced = model.fit_transform(X)

# # plt.figure()
# # plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
# # plt.show()

# # from sklearn import cluster

# # model = cluster.KMeans(n_clusters=10)

# # model.fit(X)

# # print(model.labels_[1:30])
# # print(y[1:30])

# # a = np.array([1,2,3,4,5,6,7,8,9])
# # print(a)
# # print(a.reshape(-1,1))

# # Regression

# # boston = datasets.load_boston()
# # X,y = boston.data,boston.target

# # from sklearn.preprocessing import StandardScaler
# # Xs= StandardScaler().fit(X.reshape(-1,1))
# # ys = StandardScaler().fit(y)

# # X = Xs.transform(X)
# # y = ys.transform(y)

# # from sklearn import linear_model
# # #lm = linear_model.LinearRegression()
# # lm = linear_model.SGDRegressor()
# # lm.fit(X,y)
# # plt.scatter(y,lm.predict(X))
# # plt.xlabel('Actual Price')
# # plt.ylabel('Predicted Price')
# # plt.show()

# # X = np.linspace(1,20,100).reshape(-1,1)
# # y = X + np.random.normal(0,1,100).reshape(-1,1)

# # from sklearn import linear_model
# # # lm = linear_model.LinearRegression()
# # lm = linear_model.SGDRegressor()

# # lm.fit(X,y)

# # plt.scatter(X,y)
# # plt.plot(X,lm.predict(X),color='red')
# # plt.show()


# # Classification
# # from sklearn import datasets
# # digits = datasets.load_digits()
# # X,y = digits.data,digits.target

# # from sklearn.cross_validation import train_test_split
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# # from sklearn import neighbors
# # clf = neighbors.KNeighborsClassifier()
# # clf.fit(X_train, y_train)

# # # predicted = clf.predict(X_test)
# # # expected = y_test
# # # matches = (predicted == expected)
# # # score1 = sum(matches)*100/len(matches)

# # from sklearn import metrics
# # score1 = metrics.accuracy_score(y_test,clf.predict(X_test) )

# # from sklearn import svm
# # clf = svm.SVC()
# # clf.fit(X_train, y_train)

# # # predicted = clf.predict(X_test)
# # # expected = y_test
# # # matches = (predicted == expected)
# # # score2 = sum(matches)*100/len(matches)
# # score2 = metrics.accuracy_score(y_test,clf.predict(X_test) )

# # from sklearn import linear_model
# # clf = linear_model.SGDClassifier()
# # clf.fit(X_train, y_train)

# # predicted = clf.predict(X_test)
# # expected = y_test
# # matches = (predicted == expected)
# # score3 = sum(matches)*100/len(matches)
# # score3 = metrics.accuracy_score(y_test,clf.predict(X_test) )


# # from sklearn import naive_bayes

# # clf = naive_bayes.GaussianNB()
# # clf.fit(X_train, y_train)

# # predicted = clf.predict(X_test)
# # expected = y_test
# # matches = (predicted == expected )
# # score4 = sum(matches)*100/len(matches)
# # score4 = metrics.accuracy_score(y_test,clf.predict(X_test) )


# # print('KNN score is ',score1)
# # print('SVM score is ',score2)
# # print('SGD score is ',score3)
# # print('GNB score is ',score4)

# # from sklearn import neural_network
# # clf = neural_network.MLPClassifier()
# # clf.fit(X_train, y_train) 
# # print(clf.predict(X_test[0:10]))
# # print(y_test[0:10])

# # from sklearn import cluster

# # k_means = cluster.KMeans(n_clusters=3)
# # k_means.fit(X) 
# # print(k_means.labels_[::10])
# # print(y[::10])

# #Supervised learning

# # a = np.array([9,8,7,6,5,4,3,2,1])
# # print(a[1:10:2])

# # #SGD : Steepest Gradient Descent

# # #Step 1: Load the classfier
# # from sklearn import linear_model
# # clf = linear_model.SGDClassifier()

# # #Step 2: Training

# # digits = datasets.load_digits()
# # X,y = digits.data, digits.target
# # from sklearn.cross_validation import train_test_split
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# # clf.fit(X_train,y_train)

# # #Step 3: Testing

# # print(clf.predict(X_test[0:10]))
# # print(y_test[0:10])

# # #KNN : Kth Nearnest Neighbours

# # #Step 1: Load the classfier
# # from sklearn import neighbors
# # clf = neighbors.KNeighborsClassifier()

# # #Step 2: Training

# # iris = datasets.load_iris()
# # X,y = iris.data, iris.target
# # from sklearn.cross_validation import train_test_split
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# # clf.fit(X_train,y_train)

# # #Step 3: Testing

# # print(clf.predict([[3,5,4,2]]))
# #print(y_test[0:25])


# # Step 1: Load the classfier
# from sklearn import svm
# clf = svm.SVC()

# #Step 2: Training

# iris = datasets.load_iris()
# X,y = iris.data, iris.target
# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# clf.fit(X_train,y_train)

# import pickle
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)

#Step 3: Testing

# print(clf2.predict(X_test[0:10]))
# print(y_test[0:10])



# Data preprocessing

# boston = datasets.load_boston()

# X = boston.data

# from sklearn import preprocessing
# s = preprocessing.StandardScaler().fit(X)
# X = s.transform(X)

# fig = plt.figure(figsize=(12,12))
# for i in range(13):
# 	x = X[:,i]
# 	y = boston.target
# 	ax = fig.add_subplot(4,4,i+1)
# 	ax.scatter(x,y)
# 	plt.xlabel(boston.feature_names[i])
# plt.show()

# Data randomization

# a = np.array([1,2,3,4,5,6])
# indices = np.random.permutation(len(a))
# print(a[indices])

# Splitting data for training & testing

# digits = datasets.load_digits()

# X,y = digits.data,digits.target

# from sklearn.cross_validation import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# indices = np.random.permutation(len(X))

# X_train = X[indices[:-100]]
# y_train = y[indices[:-100]]
# X_test = X[indices[-100:]]
# y_test = y[indices[-100:]]

# iris = datasets.load_iris()

# X,y = iris.data, iris.target

# from sklearn.cross_validation import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# indices = np.random.permutation(len(X))

# X_train = X[indices[:-20]]
# y_train = y[indices[:-20]]
# X_test = X[indices[-20:]]
# y_test = y[indices[-20:]]

#print(y_train)
# print(y_test)


# Diabetes

# diabetes = datasets.load_diabetes()
# print(diabetes.target)

#Boston Housing Price
#boston = datasets.load_boston()
#print(boston.data)
#print(boston.target)
# index= 1
# x = boston.data[:,index]
# y = boston.target

# plt.scatter(x,y)
# plt.xlabel(boston.feature_names[index])
# plt.ylabel('Housing price')
# plt.show()

# fig = plt.figure(figsize=(12,12))
# for i in range(13):
# 	x = boston.data[:,i]
# 	y = boston.target
# 	ax = fig.add_subplot(4,4,i+1)
# 	ax.scatter(x,y)
# 	plt.xlabel(boston.feature_names[i])
# plt.show()


#print(boston.feature_names)
# Olivetti Faces
# faces = datasets.fetch_olivetti_faces()
# fig = plt.figure(figsize=(12,12))
# for i in range(8*8):
# 	ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
# 	ax.imshow(faces.images[i],cmap=plt.cm.bone)
# plt.imshow(faces.images[3],cmap=plt.cm.bone)
# plt.show()

#Handwritten Digits dataset
# digits = datasets.load_digits()

# fig = plt.figure(figsize=(10,10))
# for i in range(16*16):
# 	ax = fig.add_subplot(16,16,i+1,xticks=[],yticks=[])
# 	ax.imshow(digits.images[i],cmap=plt.cm.binary)

#plt.imshow(digits.images[1],cmap=plt.cm.binary)
#plt.show()
#print(digits.images[0])
#print(digits.target[0:40])

# Iris Flower dataset
# iris = datasets.load_iris()
# x = iris.data[:,2]
# y = iris.data[:,3]

# plt.scatter(x,y,c=iris.target)
# plt.colorbar(ticks=[0,1,2])
# plt.xlabel(iris.feature_names[2])
# plt.ylabel(iris.feature_names[3])
# plt.show()




# import numpy as np 
# from sklearn import datasets

# iris = datasets.load_iris()

# X,y = iris.data, iris.target

# plt.scatter(X[:,0],X[:,1],c=y)
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.show()

# indices = np.random.permutation(len(X))

# X_train = X[indices[:-10]]
# y_train = y[indices[:-10]]
# X_test = X[indices[-10:]]
# y_test = y[indices[-10:]]

# X_train = X[:-20]
# y_train = y[:-20]
# X_test = X[-20:]
# y_test = y[-20:]

# from sklearn import neighbors

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train,y_train)
# print(clf.predict(X_test))
# print(y_test)

# from sklearn import datasets

# dia = datasets.load_diabetes()

# X,y = dia.data,dia.target

# X_train = X[:-20]
# y_train = y[:-20]
# X_test = X[-20:]
# y_test = y[-20:]


# from sklearn import linear_model

# model = linear_model.LinearRegression()
# model.fit(X_train,y_train)

# import matplotlib.pyplot as plt 

# plt.close()
# plt.scatter(X_train[:,4],y_train)
# plt.show()
# print(model.predict(X_test))
# print(y_test)
