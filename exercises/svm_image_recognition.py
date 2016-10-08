
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train,y_train)

y_pred= model.predict(X_test)


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

# import matplotlib.pyplot as plt

# # plt.imshow(faces.images[0])
# # plt.show()

# def print_faces(images, target, top_n):
#     # set up the figure size in inches
#     fig = plt.figure(figsize=(12, 12))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#     for i in range(top_n):
#         # plot the images in a matrix of 20x20
#         p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
#         p.imshow(images[i], cmap=plt.cm.bone)

#         # label the image with the target value
#         p.text(0, 14, str(target[i]))
#         p.text(0, 60, str(i))
#     plt.show()


# print_faces(faces.images, faces.target,200)