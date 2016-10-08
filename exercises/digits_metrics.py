from sklearn import datasets

digits = datasets.load_digits()

X = digits.data
y = digits.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# classification
from sklearn.svm import SVC
model = SVC()
model.fit(X, y)
predicted = model.predict(X_test)
expected = y_test

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6)) 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)


# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')
    
    # label the image with the target value
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(predicted[i]), color='red')

plt.show()


# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# import matplotlib.pyplot as plt 
# plt.imshow(digits.images[0])
# plt.show()
