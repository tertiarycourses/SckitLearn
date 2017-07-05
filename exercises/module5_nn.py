# Code guide for Python Scikit Learning Essential Training
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 25 Dec 2016

# Module 5: Neural Networks

# Load Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

# Step 1 Load NN Classifer
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(200,100,40),activation='relu',
		solver='adam',max_iter=1000,learning_rate='constant',random_state=17)

# Step 2 Training	
nn.fit(X_train, y_train) 

# Step 3 Testing/Prediction
# print(nn.predict(X_test)[:20])
# print(y_test[:20])
print(nn.score(X_test,y_test))