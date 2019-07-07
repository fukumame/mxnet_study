from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as plt
import time

mnist_train = datasets.FashionMNIST(train=True)
X, y = mnist_train[0]
print(X.shape)
print(y)


text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

X, y = mnist_train[0:10]

display.set_matplotlib_formats('svg')

_, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))

for f,x,yi in zip(figs, X,y):
    f.imshow(x.reshape((28, 28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[int(yi)])
    ax.title.set_fontsize(14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])

mnist_train = mnist_train.transform_first(transformer)


batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

for data, label in train_data:
    print(data.shape, label.shape)
    break