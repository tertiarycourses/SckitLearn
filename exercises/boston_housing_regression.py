import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X)
scalery = StandardScaler().fit(y)

X = scalerX.transform(X)
y = scalery.transform(y)

from sklearn import linear_model

model = linear_model.SGDRegressor()
model.fit(X,y)

plt.plot(X[:,2],y,'or')
y_pred = model.predict(X)
plt.plot(X[:,2],y_pred,'og')
plt.show()