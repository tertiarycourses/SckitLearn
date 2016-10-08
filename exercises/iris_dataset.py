import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()

# print(iris.keys())
# print(iris.data.shape)
# print(iris.data)
# print(iris.target)
# print(iris.target.shape)
# print(iris.target_names)

x_index = 0
y_index = 1


plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2])
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.show()

