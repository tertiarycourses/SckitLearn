import sklearn as sk
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow

from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target

# X = np.array([[-1, -1], [-2, -2], [-3, -3], [1, 1], [2, 2], [3, 3]])
# plt.scatter(X[:,0],X[:,1])
# plt.show()

# Step 1: Load the Model
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)

# Step 2: Training
X_reduced = pca.fit_transform(X)

# Step 3: Evaluation
import pandas as pd 
comps = pd.DataFrame(pca.components_, columns=iris.feature_names)
print(comps)

#print(pca.explained_variance_ratio_) 
#print(X_reduced)
# plt.figure(1)
# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
# plt.figure(2)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# # Step 1: Load the Model

# # from sklearn import cluster
# # c = cluster.KMeans(n_clusters=3)

# # from sklearn.cluster import MeanShift,estimate_bandwidth
# # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
# # clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# from sklearn.cluster import AgglomerativeClustering
# clf = AgglomerativeClustering(n_clusters=3)

# # Step 2: Training
# clf.fit(X) 

# # Step 3: Evalution
# print(clf.labels_[::10])
# print(y[::10])

# # X = np.array([
# # 	[1,2],
# # 	[1.5,1.8],
# # 	[5,8],
# # 	[8,8],
# # 	[1,0.6],
# # 	[9,11]
# # 	])


# # # Step 1: Load the Model
# # from sklearn import cluster
# # clf = cluster.KMeans(n_clusters=2)

# # # Step 2: Training
# # clf.fit(X)

# # #Step 3: Evaluation
# # centriods = clf.cluster_centers_
# # print(centriods)
# # labels = clf.labels_
# # print(labels)

# # colors = ["r.","b.","c.","y.","k."]

# # for i in range(len(X)):
# # 	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25) 
# # plt.scatter(centriods[:,0],centriods[:,1],marker='x',s=150,)
# # plt.show()

# # plt.scatter(X[:,0],X[:,1])
# # plt.show()



# # import numpy as np
# # X = np.linspace(1,20,100).reshape(-1,1)
# # y = X + np.random.normal(0,1,100).reshape(-1,1)

# # from sklearn import datasets
# # boston = datasets.load_boston()
# # X,y = boston.data, boston.target

# # # Step 1: Load the Model
# # from sklearn import linear_model
# # lm = linear_model.LinearRegression()

# # # Step 2: Training
# # lm.fit(X,y)

# # # Step 3: Evaluation
# # import matplotlib.pyplot as plt 
# # plt.scatter(y,lm.predict(X))
# # # plt.plot(X,lm.predict(X),'r')
# # plt.show()

# # Step 1: Get Data
# # from sklearn import datasets
# # iris = datasets.load_iris()
# # X,y = iris.data, iris.target

# # digits = datasets.load_digits()
# # X,y = digits.data, digits.target

# Step 3/4 Randomize/Split data
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,
# 	test_size=0.25,random_state=33)

# Step 5: :Load the Model

# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(n_neighbors=3)

# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier()

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

# from sklearn import svm
# clf = svm.SVC()

# Step 6: Training
# clf.fit(X_train,y_train)

# with open("iris.dot", 'w') as f:
# 	f = tree.export_graphviz(clf, out_file=f)

# Step 7: Evaluation
# print(clf.score(X_test,y_test))

# from sklearn import metrics
# predicted = clf.predict(X_test)
# expected = y_test
# print(metrics.classification_report(expected, predicted))

# print(clf.predict(X_test[:20]))
# print(y_test[:20])

# Step 8: Ouput the model

# from sklearn.externals import joblib
# joblib.dump(clf,'DT.pkl')

# import pickle
# pickle.dump(clf,open('KNN.pkl','wb'))
# boston = datasets.load_boston()
# print(boston.data)
# print(boston.target)
# print(boston.feature_names)
# x = boston.data[:,0]
# y = boston.target
# plt.scatter(x,y)
# plt.xlabel(boston.feature_names[0])
# plt.ylabel('price')
# plt.show()

# digits = datasets.load_digits()

# fig = plt.figure(figsize=(12,12))
# for i in range(16*16):
#  	ax = fig.add_subplot(16,16,i+1,xticks=[],yticks=[])
#  	ax.imshow(digits.images[i],cmap=plt.cm.binary)
# plt.show()

# print(digits.data)
# print(digits.target)



# plt.imshow(digits.images[3],cmap=plt.cm.gray_r)
# plt.show()
# iris = datasets.load_iris()
# # print(iris.data)
# print(iris.target)

# plt.figure(1)
# x = iris.data[:,0]
# y = iris.data[:,1]
# plt.scatter(x,y,c=iris.target)
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.figure(2)
# x = iris.data[:,1]
# y = iris.data[:,2]
# plt.scatter(x,y,c=iris.target)
# plt.xlabel(iris.feature_names[1])
# plt.ylabel(iris.feature_names[2])
# plt.figure(3)
# x = iris.data[:,2]
# y = iris.data[:,3]
# plt.scatter(x,y,c=iris.target)
# plt.xlabel(iris.feature_names[2])
# plt.ylabel(iris.feature_names[3])
# plt.show()
