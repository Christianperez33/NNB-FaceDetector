from __future__ import print_function
from math import floor
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
from PIL import Image
from random import *
from tensorflow.image import resize_images
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda,Cropping2D,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD
from keras.backend import tf as ktf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


import cv2,time,scipy,os, sys,keras, urllib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--image',
    default='./test_images/cara.jpg',
    help='Path of label prediction')
argparser.add_argument(
    '-m',
    '--model',
    default='NNB_Face_Detector.hdf5',
    help='hdf5 file of model')
argparser.add_argument(
    '-t',
    '--threshold',
    default=1.0,
    help='image threshold')
argparser.add_argument(
    '-ms',
    '--maxsupress',
    default=0.1,
    help='non maximun supression threshold')

args = argparser.parse_args()

def kcrop(img, random_crop_size,init):
    height, width = img.shape[0], img.shape[1]
    dx, dy = random_crop_size
    x,y = init  
    return img[x:(x+dx),y:(y+dy)]

def rgb2gray(rgb):
    res = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return res

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    return boxes[pick]

input_size = (32, 32,1)
batch_size = 2048

print("Building Model")
# Start building our model
model = Sequential()

#load model
model = load_model(args.model)

# load weights
filepath="weights.best.hdf5"
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

img = cv2.imread(args.image)
imgres = img
img = img / 255.
img = rgb2gray(img)
bounding_boxes = {}
images = []
i = 0
j = 0
pyramidTime = time.time()

while( (floor(img.shape[0]*0.9),floor(img.shape[1]*0.9)) >= input_size[:-1] ):
    input_image = cv2.resize(img,(floor(img.shape[1]*0.9),floor(img.shape[0]*0.9)))
    img = input_image
    for w in range(input_image.shape[1]-input_size[1]+1):
        for h in range(input_image.shape[0]-input_size[0]+1):
            imgcrop = np.asarray(kcrop(input_image, input_size[:-1],(h,w)))
            imgcrop = np.expand_dims(imgcrop, -1)
            images.append(imgcrop)
            bounding_boxes[str(i)] = [input_image.shape[0],input_image.shape[1],h,w]
            i += 1
    j += 1
    # print("Scale {} processed...".format(j))

images = np.asarray(images)

print("Pyramid image generator time passed : {}".format(time.time()-pyramidTime))
batch_size = min(2**floor(np.log2(images.shape[0])),2**11)
netout = model.predict(images, batch_size=batch_size, verbose=0)

##Threshold discard list 
toNMS =[]
thresholdTime =time.time()
for x,no in enumerate(netout):
    if no[1] >= args.threshold:
        scaley,scalex,y,x=bounding_boxes[str(x)]
        scalex = int(imgres.shape[1]/scalex)
        scaley = int(imgres.shape[0]/scaley)
        x = x *scalex
        y = y *scaley
        toNMS.append((x,y,x+input_size[0]*scalex,y+input_size[1]*scaley,float(no[1])))


toNMS = np.array(toNMS)
print("Threshold discard time passed : {}".format(time.time()-thresholdTime))
del thresholdTime

##Non-Maximun Supression Boxs
cajas = non_max_suppression_fast(toNMS,float(args.maxsupress))

for x0,y0,x1,y1,s in cajas:
    x0=int(x0)
    y0=int(y0)
    x1=int(x1)
    y1=int(y1)
    imgres = cv2.rectangle(imgres,(x0,y0),(x1,y1),(0,255,0),2)
    imgres = cv2.putText(imgres, 
                    'Face:' + "{0:.2f}".format(s*100), 
                    (x0,y0+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (0,255,0),
                    1)
print("Final time : {}".format(time.time()-pyramidTime))
plt.imshow(imgres)
plt.show()
