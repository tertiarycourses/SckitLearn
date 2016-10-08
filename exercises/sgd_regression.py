import numpy as np
import matplotlib.pyplot as plt

X= np.linspace(0, 10, 200).reshape(-1,1)
y = X.ravel()+np.random.normal(0,1,200)

from sklearn import linear_model

lm = linear_model.SGDRegressor()

lm.fit(X, y)
plt.scatter(X,y)
plt.plot(X,lm.predict(X),'-r')
plt.show()
