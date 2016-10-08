from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

from sklearn.svm import SVC
model = SVC()
model.fit(X, y)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
print(iris.target_names[model.predict([[3, 5, 4, 2]])])

