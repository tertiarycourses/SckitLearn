# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 5: Intro to Neural Networks

from sklearn.neural_network import MLPClassifier

X = [[0, 0],[0,1],[1,1]]
y = [0,1,1]

clf = MLPClassifier(hidden_layer_sizes=(5, 2))
clf.fit(X, y) 

print(clf.predict([[2,1]]))

# Iris  dataaset
# from sklearn import datasets
# iris = datasets.load_iris()
# X,y = iris.data, iris.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# Step 1 Model
# from sklearn import neural_network
# clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,60,40),activation='relu',solver='adam')

# Step 2 Training
# clf.fit(X_train,y_train)

# # # Step 3 Testing/Prediction
# print(clf.predict(X_test)[:20])
# print(y_test[:20])
# score = clf.score(X_test,y_test)
# print(score)




