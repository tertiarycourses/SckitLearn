# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3.2: Metrics

# Confusion Metrics
# from sklearn.metrics import confusion_matrix
# y = ["cat", "ant", "cat", "cat", "ant", "bird"]
# yhat = ["ant", "ant", "cat", "cat", "ant", "cat"]
# print(confusion_matrix(y, yhat, labels=["ant", "bird", "cat"]))

# # Setp 1 Get Data
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# from sklearn.preprocessing import scale 
# X = scale(X)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train,y_train)

# predicted = clf.predict(X_test)
# expected = y_test

# # Score Metric
# score = clf.score(X_test,y_test)
# print(score)

# from sklearn import metrics
# print(metrics.classification_report(expected, predicted))



