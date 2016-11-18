import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
# %matplotlib inline

import numpy as np
from skimage.io import imread
from sklearn.datasets import fetch_mldata
import numpy as np
from scipy import ndimage


from skimage.io import imread


mnist = fetch_mldata('MNIST original')

data = mnist.data / 255.0
labels = mnist.target.astype('int')

train_rank = 5000
test_rank = 100
#------- MNIST subset --------------------------
train_subset = np.random.choice(data.shape[0], train_rank)
test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
train_data = data[train_subset]
train_labels = labels[train_subset]

# test dataset
test_data = data[test_subset]
test_labels = labels[test_subset]

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

test = [3, 5, 9]
print to_categorical(test, 10)

train_out = to_categorical(train_labels, 10)
test_out = to_categorical(test_labels, 10)

#--------------- ANN ------------------
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

# prepare model
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))

# compile model with optimizer
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

# training
training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
print training.history['loss'][-1]
# evaluate on test data
scores = model.evaluate(test_data, test_out, verbose=1)
print 'test', scores
# evaluate on train data
scores = model.evaluate(train_data, train_out, verbose=1)
print 'train', scores

file= open("out.txt", "w")
file.write("RA 157/2013 Sasa Lalic\n")
file.write("file \t sum\n")
for num in range(0,100):
    img = imread('images/img-{}.png'.format(num))  # img je Numpy array
    print num
    plt.imshow(img)
    plt.show()
    (h, w, c) = img.shape

    imgw = img[:, :, 0]
    imgw = imgw > 50
    # plt.imshow(imgw, 'gray')  # imshow je funkcija za prikaz slike (u formatu Numpy array-a)
    # plt.show()
    from skimage.measure import regionprops

    labeled, nr_objects = ndimage.label(imgw)
    regions = regionprops(labeled)
    nr_objects = nr_objects - 1
    print nr_objects
    sum = 0
    for number in range(0, nr_objects):
        obj = regions[number]
        blok_size = (28, 28)
        blok_center = (int(obj.centroid[0]), int(obj.centroid[1]))
        blok_loc = (blok_center[0] - blok_size[0] / 2, blok_center[1] - blok_size[1] / 2)

        imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0],
               blok_loc[1]:blok_loc[1] + blok_size[1], 0]
        # plt.imshow(imgB, cmap="Greys")
        # plt.show()
        (h, w) = imgB.shape
        imgB_test = imgB.reshape(784)
        # print imgB_test
        imgB_test = imgB_test / 255.
        # print imgB_test.shape
        tt = model.predict(np.array([imgB_test]), verbose=1)
        print tt
        result = 0
        answer = np.argmax(tt)
        print answer
        sum += answer
    print sum
    file.write('images/img-{}.png\t{}\n'.format(num,sum))