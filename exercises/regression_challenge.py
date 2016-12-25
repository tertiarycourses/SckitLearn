import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

boston = datasets.load_boston()
X,y = boston.data,boston.target


from sklearn import linear_model

lm = linear_model.LinearRegression()

lm.fit(X,y)

X1 = pd.DataFrame(X)
X1.columns = boston.feature_names

#plt.scatter(X1.RM,y)
plt.scatter(y,lm.predict(X))
plt.xlabel('Price')
plt.ylabel('Predict Price')
plt.show()
