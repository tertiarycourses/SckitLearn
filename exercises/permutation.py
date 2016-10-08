import numpy as np 

a = np.array([1,2,3,4,5,6])

#print(a[[2,3]])

indices = np.random.permutation(len(a))

print(a[indices])
# from sklearn import datasets

# dia = datasets.load_diabetes()

# X,y = dia.data,dia.target

# X_train = X[:-20]
# y_train = y[:-20]
# X_test = X[-20:]
# y_test = y[-20:]

# from sklearn import linear_model

# model = linear_model.LinearRegression()
# model.fit(X_train,y_train)

# import matplotlib.pyplot as plt 

# plt.close()
# plt.scatter(X_train[:,4],y_train)
# plt.show()
# print(model.predict(X_test))
# print(y_test)
