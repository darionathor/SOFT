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

line = [(100,450), (500, 100)]
line1 = [(100,450), (500, 100)]
line2 = [(100,450), (400, 100)]

for num in range(5,6):
    cap = cv2.VideoCapture("videos/video-{}.avi".format(num))

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
    kernel = np.ones((2,2),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    lowerl1 = np.array([0, 230, 0])
    upperl1 = np.array([155, 255, 155])
    lowerl2 = np.array([230, 0, 0])
    upperl2 = np.array([255, 155, 155])
    #boundaries = [
    #    ([230, 230, 230], [255, 255, 255])
    #]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

    elements = []
    t =0
    counter = 0
    times = []
    while (1):
        start_time = time.time()
        ret, img = cap.read()
        file_name = 'images/frame-' + str(t) + '.png'

        cv2.imwrite(file_name, img)
        if not(ret):
            break
        # (lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)

        maskl1= cv2.inRange(img,lowerl1,upperl1)
        maskl2= cv2.inRange(img,lowerl2,upperl2)

        img0 = 1.0 * mask


        #cv2.imshow('frame', imgl1)
        img0 = cv2.dilate(img0, kernel)

        #cv2.erode(img0,kernel)
        #img0 = cv2.dilate(img0, kernel)



        maxLineGap = 40
        minLineLength = 15
        lines = cv2.HoughLinesP(maskl1, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        #print lines
        line1[0]=(lines[0][0][0],lines[0][0][1])
        line1[1]=(lines[0][0][2],lines[0][0][3])
        lines = cv2.HoughLinesP(maskl2, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        #print lines
        line2[0]=(lines[0][0][0],lines[0][0][1])
        line2[1]=(lines[0][0][2],lines[0][0][3])

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
                    elem['pass1'] = False
                    elem['pass2'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    elem['vector'] = []
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']

            if (tt < 3):
                result1 = pnt2line(el['center'], line1[0], line1[1])
                result2 = pnt2line(el['center'], line2[0], line2[1])
                dist1=result1[0]
                dist2 = result2[0]

                pnt1=result1[1]
                pnt2=result2[1]
                r1=0
                r2=0
                if(len(result1)>2):
                    r1=result1[2]
                if(len(result2)>2):
                    r2=result2[2]

                #cv2.line(img, pnt1, el['center'], (0, 255, 25), 1)
                c = (25, 25, 255)
                if (dist1 < 9):
                    c = (0, 255, 160)
                    if el['pass1'] == False:
                        el['pass1'] = True
                        counter += 1

                #cv2.circle(img, el['center'], 16, c, 2)

               # cv2.line(img, pnt2, el['center'], (0, 255, 25), 1)
                #c = (25, 25, 255)
                if (dist2 < 9):
                    c = (0, 255, 160)
                    if el['pass2'] == False:
                        el['pass2'] = True
                        counter += 1

                cv2.circle(img, el['center'], 16, c, 2)

                id = el['id']
                cv2.putText(img, str(el['id']),
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

                er = []
                xn = []
                for hist in el['history']:
                    er.append(hist['center'][0])
                    xn.append(hist['center'][1])
                    ttt = t - hist['t']
                    if (ttt < 100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                if (len(er) > 5):
                    rez=np.polyfit(er,xn,1)
                    p=np.poly1d(rez)
                    el['vector']=rez
                    cv2.line(img, (0, int(p(0))), (640, int(p(640))), (155, 155, 25), 1)

        cv2.line(img,line1[0],line1[1],(0,25,255),1)
        cv2.line(img,line2[0],line2[1],(0,25,255),1)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Counter: ' + str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        # print nr_objects
        t += 1
        if t % 10 == 0:
            print t

       # if t == 200:
        #    break
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
    suma=0


    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)


    # Return true if line segments AB and CD intersect
    def intersect(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
        return ccw(Ax, Ay, Cx, Cy, Dx, Dy) != ccw(Bx, By, Cx, Cy, Dx, Dy) and ccw(Ax, Ay, Bx, By, Cx, Cy) != ccw(Ax, Ay,
                                                                                                                 Bx, By,
                                                                                                                 Dx, Dy)


    class Point:
        def __init__(self, x, y):
            self.x = x;
            self.y = y


    for el in elements:
        if not el['pass1'] or not el['pass2']:
            for el2 in elements:
                if el['pass1'] != el2['pass1'] or el['pass2'] != el2['pass2']:
                    if el['history'][-1]['t'] < el2['history'][0]['t']:
                        # compare vectors
                        v1 = np.poly1d(el['vector'])
                        v2 = np.poly1d(el2['vector'])
                        if abs(v1(0) - v2(0)) <5  and abs(v1(640) - v2(640)) < 10:
                            el['center'] = el2['center']
                            el['t'] = el2['t']
                            for d in el2['history']:
                                el['history'].append(d)
                            # el['history'].append(el2['history'])
                            # A = Point(el['history'][0]['center'][0],el['history'][0]['center'][1]);
                            # B=Point(el['center'][0],el['center']);
                            # C=Point(line1[0][0],line1[0][1])
                            # D=Point(line1[1][0],line1[1][1])
                            if intersect(el['history'][0]['center'][0], el['history'][0]['center'][1],
                                         el['center'][0], el['center'][1],
                                         line1[0][0], line1[0][1], line1[1][0], line1[1][1]):
                                el['pass1'] = True

                            # D = Point(line2[1][0], line2[1][1])
                            if intersect(el['history'][0]['center'][0], el['history'][0]['center'][1],
                                         el['center'][0], el['center'][1],
                                         line1[0][0], line1[0][1],
                                         line2[1][0], line2[1][1]):
                                el['pass2'] = True
                            print 'vector error correction success'

    for el in elements:
        if el['pass1'] or el['pass2']:
            x_move=[-1,0,1]
            y_move = [-1, 0, 1]
            tt=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for br in range(2,31):
                current=int((br-1)*len(el['history'])/30)
                img = imread('images/frame-{}.png'.format(el['history'][current]['t']))
                blok_size = (28, 28)
                blok_center = (int(el['history'][current]['center'][0]), int(el['history'][current]['center'][1]))

                for movex in x_move:
                    for movey in y_move:
                        blok_center2=(blok_center[0]+movex,blok_center[1]+movey)
                        blok_loc = (blok_center2[1] - blok_size[0] / 2, blok_center2[0] - blok_size[1] / 2)
                        if(blok_loc[0]<0 or blok_loc[1]<0 or blok_loc[0]>452 or blok_loc[1]>592):
                            continue
                        #print blok_loc
                        imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0],
                               blok_loc[1]:blok_loc[1] + blok_size[1],0]

                       # cv2.circle(img, blok_center, 16, c, 2)
                        #cv2.imshow('frame',img)
                        #print el
                       # print blok_loc
                       # plt.imshow(img)
                       # plt.show()
                       # if movex==0 and movey==0:
                        #    plt.imshow(imgB, cmap="Greys")
                        #    plt.show()
                        #(h, w) = imgB.shape

                        imgB = imgB.reshape(1, 1, 28, 28).astype('float32')

                        # normalize inputs from 0-255 to 0-1
                        imgB = imgB / 255

                        imgB_test = imgB.reshape(784)
                        # print imgB_test
                        imgB_test = imgB_test / 255.
                        # print imgB_test.shape
                        tt += model.predict(imgB, verbose=1)
            #tt=tt/10
            print tt
            result = 0
            answer = np.argmax(tt)
            print answer
            if(el['pass1']):
                suma-=answer
            if(el['pass2']):
                suma+=answer
    print suma



