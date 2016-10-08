import matplotlib.pyplot as plt

from sklearn import datasets
digits = datasets.load_digits()

#print(digits.keys())
#print(digits.data.shape)
#print(digits.data)
#print(digits.target)
#print(digits.target.shape)
#print(digits.target_names)

# image = 1
# plt.imshow(digits.images[image],cmap=plt.cm.binary)
# plt.show()

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    #ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()