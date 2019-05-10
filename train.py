from __future__ import print_function
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
from PIL import Image
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
from random import *
from tensorflow.image import resize_images
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import *
from keras.backend import tf as ktf
from keras.callbacks import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from DataGenerator import *
from tqdm import tqdm


import random
import cv2
import os, sys
import numpy as np
import tensorflow as tf
import keras
import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    res = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return res
  
def resize_normalize(image, size=(32, 32)): 
    resized = ktf.image.resize_images(image, size)
    resized = Reshape(size + (1,), input_shape=size)(resized)
    return resized

def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx)]

def preproc(image):
    i = int(random.uniform(1, 3))
    return np.expand_dims(cv2.blur(image,(i,i)),-1)



print("Starting...")
# Load 
all_faces    = np.asarray([x for x in list(open('./faces.txt'))[0].split(';') if len(x)>0 ])
no_faces = np.asarray([x for x in list(open('./no_faces.txt'))[0].split(';') if len(x)>0 ])
faces =  np.asarray([])
if list(open('./fp_faces.txt')) != []:
    faces = np.asarray([x for x in list(open('./fp_faces.txt'))[0].split(';') if len(x)>0 ])

f_train,f_test = train_test_split(all_faces, test_size=0.2)
f_train,f_val = train_test_split(f_train, test_size=0.2)
nf_train,nf_test = train_test_split(no_faces, test_size=0.2)
nf_train,nf_val = train_test_split(nf_train, test_size=0.2)

f_train = np.concatenate((faces,f_train))

print("f_train:{}, nf_train:{}".format(f_train.shape,nf_train.shape))
print("f_test:{},nf_test:{}".format(f_test.shape,nf_test.shape))
print("f_val:{}, nf_val:{}".format(f_val.shape,nf_val.shape))

gn = 0.15
epochs = 100
params = {
        'dim': (32,32),
        'batch_size': min(len(f_val),4096),
        'n_channels': 1,
        'n_classes' : 2,
        'shuffle': True,
        'dir_faces': './face_images/',
        'dir_no_faces': './no_face_images/'
        }

training_generator   = DataGenerator(f_train, nf_train, **params)
test_generator       = DataGenerator(f_test , nf_test , **params)
validation_generator = DataGenerator(f_val  , nf_val  , **params)

print("Building Model")
# Start building our model
model = Sequential()
model.add(BN(input_shape=(32, 32,1)))
model.add(GN(gn))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BN())
model.add(GN(gn))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BN())
model.add(GN(gn*1.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BN())
model.add(GN(gn*2))
model.add(Conv2D(256, (3, 3) , padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BN())
model.add(GN(gn*2.5))
model.add(Conv2D(256, (3, 3) , padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()


#####TRAINING OF NNB

## OPTIM AND COMPILE
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
# load weights
print("Preload weights...")
# load weights
filepath="weights.best.hdf5"
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > epochs*0.8:
        lr *= 0.5e-3
    elif epoch > epochs*0.6:
        lr *= 1e-3
    elif epoch > epochs*0.4:
        lr *= 1e-2
    elif epoch > epochs*0.2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10,verbose=1)
# checkpoint
checkpoint = ModelCheckpoint(
    os.path.join('checkpoints','weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min')
# DEFINE A LEARNING RATE SCHEDULER
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',patience=4,verbose=1, factor=0.5,min_lr=0.00001)
Lr_reduction = LearningRateScheduler(lr_schedule)


## TRAINING with DA and LRA
history=model.fit_generator(training_generator,
                            steps_per_epoch=len(f_train)//params['batch_size'],
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=len(f_val)//params['batch_size'],
                            callbacks=[early_stop,learning_rate_reduction,checkpoint,Lr_reduction],
                            max_queue_size=3,
                            verbose=1)

model.save('NNB_Face_Detector.hdf5')
model.save_weights(os.path.join(".", filepath), overwrite=True)
score = model.evaluate_generator(test_generator, 
                                max_queue_size=3,
                                steps=len(f_test)//params['batch_size'],
                                verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', (score[1])*100,'%')

with open('fp_faces.txt','w') as f:
    for img in tqdm(all_faces):
        topredict = np.expand_dims(cv2.resize(cv2.imread('./face_images/'+img,0)/255,params['dim']),-1)
        predict = model.predict( np.expand_dims(topredict,0))
        if np.argmax(predict) == 0:
            f.write(img+';')

#OUTPUT
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc_train', 'acc_validation','loss_train', 'loss_validation'], loc='upper left')
plt.show()



