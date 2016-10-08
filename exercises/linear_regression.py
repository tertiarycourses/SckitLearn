import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 


X = np.linspace(1,20,100).reshape(-1,1)
y = X + np.random.normal(0,1,100).reshape(-1,1)

lm = linear_model.LinearRegression()
lm.fit(X, y) 

plt.scatter(X,y)
plt.plot(X,lm.predict(X),'-r')
plt.show()