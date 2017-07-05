# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3.2a: Joblib

from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

# from sklearn.externals import joblib
# joblib.dump(clf, 'mymodel.pkl') 

# Pickle Method
import pickle
pickle.dump(clf, open("mymodel2.pkl","wb")) 
