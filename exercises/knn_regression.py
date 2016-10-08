import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

X= np.linspace(0, 10, 200).reshape(-1,1)
y = X.ravel()+np.random.normal(0,1,200)

lm = neighbors.KNeighborsRegressor()
lm.fit(X, y)
plt.scatter(X,y)
plt.plot(X,lm.predict(X),'-r')
plt.show()
