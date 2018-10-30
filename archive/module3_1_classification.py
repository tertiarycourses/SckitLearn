# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3.1: Classification

# Setp 1 Get Data
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data, iris.target

# Step 2 Clean Data
# from sklearn.preprocessing import scale 
# X = scale(X)

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]) 
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])   
# print(enc.n_values_)
# print(enc.transform([[0, 1, 1]]).toarray())

# Load data and split data
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# digits = datasets.load_digits()
# X,y = digits.data, digits.target

# Step 2 Randomize Data and Split Data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# Step 3 Load Model

# K Nearest Neighbors (KNN)
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier()
#clf = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree')

# Support Vector Machine (SVN)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# Step 1: Load the classifer

# K Nearest Neighbors Classificaiton
#from sklearn import neighbors
#clf = neighbors.KNeighborsClassifier()
#clf = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree')

# Support Vector Machine
# from sklearn import svm
# clf = svm.SVC()
#clf = svm.SVC(kernel='rbf',degree=3)

# Stochastics Gradient Descent (SGD)
# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

# Guassian Navie Bayes (GND)
# from sklearn import naive_bayes
# clf = naive_bayes.GaussianNB()

# Decision Tree (DT)
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

# Ensemble Random Forest 
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()

# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier()

# Step 4 Model Training

# clf.fit(X_train,y_train)

# Ouput for Decision Tree
# with open("iris.dot", 'w') as f:
# 	f = tree.export_graphviz(clf, out_file=f)

# copy and paste the output to http://webgraphviz.com/

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




