# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 3.2b: joblib 

# from sklearn.externals import joblib
# clf = joblib.load('mymodel.pkl') 

# Testing
# new_flower = [[3.1,4.5,5.7,4.6]]
# print(clf.predict(new_flower))

# Pickle Method
import pickle
clf = pickle.load(open("mymodel2.pkl","rb"))

# Testing
# new_flower = [[3.1,4.5,5.7,4.6]]
# print(clf.predict(new_flower))
