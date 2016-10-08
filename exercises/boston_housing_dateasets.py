import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

boston = datasets.load_boston()
#print(boston.DESCR)
#print(boston.feature_names)

x_index = 2
plt.scatter(boston.data[:, x_index], boston.target)
plt.xlabel(boston.feature_names[x_index])
plt.ylabel("Housing Price")
plt.show()

