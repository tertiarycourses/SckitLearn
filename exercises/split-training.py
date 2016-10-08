from sklearn.cross_validation import train_test_split

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# import matplotlib.pyplot as plt 
# plt.imshow(digits.images[0])
# plt.show()
