import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
# %matplotlib inline

import numpy as np
from skimage.io import imread
from sklearn.datasets import fetch_mldata
import numpy as np
from scipy import ndimage


from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy import ndimage
import math
from sklearn.datasets import fetch_mldata
from scipy import ndimage
import cv2
import numpy as np
from vector import pnt2line
"""
mnist = fetch_mldata('MNIST original',data_home='scikit_learn_data')

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
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('th')
# prepare model
"""

"""
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))"""
"""
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

training = model.fit(train_data, train_out, nb_epoch=10, batch_size=200, verbose=2)
print training.history['loss'][-1]


# evaluate on test data
scores = model.evaluate(test_data, test_out, verbose=1)
print 'test', scores
# evaluate on train data
scores = model.evaluate(train_data, train_out, verbose=1)
print 'train', scores
"""
# compile model with optimizer
"""
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
# training


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
"""
"""
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
"""
"""
file= open("out.txt", "w")
file.write("RA 157/2013 Sasa Lalic\n")
file.write("file \t sum\n")
for num in range(0,100):
    img = imread('images/img-{}.png'.format(num))  # img je Numpy array
    print num
   # plt.imshow(img)
   # plt.show()
    (h, w, c) = img.shape

    imgw = img[:, :, 0]
    imgw = imgw > 50
    # plt.imshow(imgw, 'gray')  # imshow je funkcija za prikaz slike (u formatu Numpy array-a)
    # plt.show()
    from skimage.measure import regionprops

    from skimage.measure import regionprops
    from skimage.morphology import opening, closing
    from skimage.morphology import square, diamond, disk  # strukturni elementi

    str_elem = disk(15)
    from skimage.morphology import dilation

    from skimage.morphology import erosion
   # imgw = dilation(imgw, selem=str_elem)
   # imgw = erosion(imgw, selem=str_elem)
    imgw = closing(imgw, selem=str_elem)
  #  plt.imshow(imgw, 'gray')
   # plt.imshow(imgw, 'gray')


    labeled, nr_objects = ndimage.label(imgw)
    regions = regionprops(labeled)
    nr_objects = nr_objects
    print nr_objects
    sum = 0
    for number in range(0, nr_objects):
        obj = regions[number]
        blok_size = (28, 28)
        blok_center = (int(obj.centroid[0]), int(obj.centroid[1]))
        blok_loc = (blok_center[0] - blok_size[0] / 2, blok_center[1] - blok_size[1] / 2)

        imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0],
               blok_loc[1]:blok_loc[1] + blok_size[1],0]
        # plt.imshow(imgB, cmap="Greys")
        # plt.show()
        #(h, w) = imgB.shape

        imgB = imgB.reshape(1, 1, 28, 28).astype('float32')

        # normalize inputs from 0-255 to 0-1
        imgB = imgB / 255

        imgB_test = imgB.reshape(784)
        # print imgB_test
        imgB_test = imgB_test / 255.
        # print imgB_test.shape
        tt = model.predict(imgB, verbose=1)
        print tt
        result = 0
        answer = np.argmax(tt)
        print answer
        sum += answer
    print sum
    file.write('images/img-{}.png\t{}.0\n'.format(num,sum))
"""

line = [(100,450), (500, 100)]

cap = cv2.VideoCapture("videos/video-0.avi")

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

from vector import distance
def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal
# color filter
kernel = np.ones((3,3),np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])

#boundaries = [
#    ([230, 230, 230], [255, 255, 255])
#]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

elements = []
t =0
counter = 0
times = []

while not cap.isOpened():
    cap = cv2.VideoCapture("videos/video-0.avi")
    cv2.waitKey(1000)
    print "Wait for the header"

while (1):
    start_time = time.time()
    ret, img = cap.read()
    if not(ret):
        break
    # (lower, upper) = boundaries[0]
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    img0 = 1.0 * mask

    img0 = cv2.dilate(img0, kernel)
    #cv2.erode(img0,kernel)
    #img0 = cv2.dilate(img0, kernel)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    for i in range(nr_objects):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                    (loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = ((loc[1].stop - loc[1].start),
                      (loc[0].stop - loc[0].start))

        if (dxc > 11 or dyc > 11):
            cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
            elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
            # find in range
            lst = inRange(20, elem, elements)
            nn = len(lst)
            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t
                elem['pass'] = False
                elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                elem['future'] = []
                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                lst[0]['future'] = []

    for el in elements:
        tt = t - el['t']

        if (tt < 3):
            result = pnt2line(el['center'], line[0], line[1])
            dist=result[0]
            pnt=result[1]
            r=0
            if(len(result)>2):
                r=result[2]
            if r > 0:
                cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                c = (25, 25, 255)
                if (dist < 9):
                    c = (0, 255, 160)
                    if el['pass'] == False:
                        el['pass'] = True
                        counter += 1

                cv2.circle(img, el['center'], 16, c, 2)

            id = el['id']
            cv2.putText(img, str(el['id']),
                        (el['center'][0] + 10, el['center'][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            for hist in el['history']:
                ttt = t - hist['t']
                if (ttt < 100):
                    cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

            for fu in el['future']:
                ttt = fu[0] - t
                if (ttt < 100):
                    cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)


    elapsed_time = time.time() - start_time
    times.append(elapsed_time * 1000)
    cv2.putText(img, 'Counter: ' + str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

    # print nr_objects
    t += 1
    if t % 10 == 0:
        print t
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    out.write(img)
out.release()
cap.release()
cv2.destroyAllWindows()

et = np.array(times)
print 'mean %.2f ms' % (np.mean(et))
# print np.std(et)
"""
for j in range(0,1200):
    ret, frame = cap.read()
    #cv2.imshow('video', frame)
    file_name  = 'images/frame-'+str(j)+'.png'

    cv2.imwrite(file_name, frame)
cap.release()


img = imread('images/frame-1.png')  # img je Numpy array
#plt.imshow(img)
#plt.show()
"""
"""
imgw = img[:, :, 0]
imgw = imgw > 50
plt.imshow(imgw, 'gray')  # imshow je funkcija za prikaz slike (u formatu Numpy array-a)
plt.show()
from skimage.measure import regionprops

from skimage.morphology import square, diamond, disk  # strukturni elementi

str_elem = disk(15)
from skimage.morphology import dilation

imgw = dilation(imgw, selem=str_elem)
plt.imshow(imgw, 'gray')
from skimage.morphology import erosion

imgw = erosion(imgw, selem=str_elem)
plt.imshow(imgw, 'gray')
labeled, nr_objects = ndimage.label(imgw)
regions = regionprops(labeled)
nr_objects = nr_objects
print nr_objects
sum = 0
for number in range(nr_objects):
        obj = regions[number]
        blok_size = (28, 28)
        blok_center = (int(obj.centroid[0]), int(obj.centroid[1]))
        blok_loc = (blok_center[0] - blok_size[0] / 2, blok_center[1] - blok_size[1] / 2)

        imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0],
               blok_loc[1]:blok_loc[1] + blok_size[1],0]
        plt.imshow(imgB, cmap="Greys")
        plt.show()
        #(h, w) = imgB.shape

        imgB = imgB.reshape(1, 1, 28, 28).astype('float32')

        # normalize inputs from 0-255 to 0-1
        imgB = imgB / 255

        imgB_test = imgB.reshape(784)
        # print imgB_test
        imgB_test = imgB_test / 255.
        # print imgB_test.shape
        tt = model.predict(imgB, verbose=1)
        print tt
        answer = np.argmax(tt)
        print answer
        sum += answer
print sum"""