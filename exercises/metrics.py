from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


from sklearn import neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)


# Rough metrics
# use the model to predict the labels of the test data
# predicted = model.predict(X_test)
# expected = y_test
# matches = (predicted == expected)
# print(matches.sum())
# print(len(matches))
# print(matches.sum() / float(len(matches)))

from sklearn import metrics

# Metrics Classification Report
# print(metrics.classification_report(expected, predicted))

# Confusion Matrix
# print(metrics.confusion_matrix(expected, predicted))

# Accuracy Score
y_train_pred = model.predict(X_train)
print(metrics.accuracy_score(y_train, y_train_pred))

