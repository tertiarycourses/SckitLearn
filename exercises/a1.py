from sklearn import datasets

dia = datasets.load_iris()

X,y = dia.data,dia.target

X_train = X[:-20]
y_train = y[:-20]
X_test = X[-20:]
y_test = y[-20:]


import matplotlib.pyplot as plt 

plt.close()
plt.scatter(X_train[:,3],y_train)
plt.show()