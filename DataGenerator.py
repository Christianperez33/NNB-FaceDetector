import numpy as np
import keras
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator


def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx)]

def preproc(image):
    i = int(random.uniform(1, 3))
    return np.expand_dims(cv2.blur(image,(i,i)),-1)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, faces, no_faces, batch_size=32, dim=(32,32),
                 n_channels=1, n_classes=2, shuffle=True, dir_faces='./face_images/', dir_no_faces='./no_face_images/'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.faces = faces
        self.no_faces = no_faces
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dir_faces = dir_faces
        self.dir_no_faces = dir_no_faces
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.faces) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_faces    = [self.faces[k] for k in indexes]
        no_face       = cv2.imread(self.dir_no_faces+self.no_faces[random.randint(0,len(self.no_faces)-1)],0)/255

        # Generate data
        X, y = self.__data_generation(list_faces,no_face)

        datagen = ImageDataGenerator(
                preprocessing_function=preproc,
                rotation_range=random.randint(5,12),
                width_shift_range=random.uniform(0.1,0.3),
                height_shift_range=random.uniform(0.1,0.3),
                horizontal_flip=True)
        datagen.fit(X)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.faces))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_faces,no_face):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        faces    = np.empty((self.batch_size//2, *self.dim, self.n_channels))
        no_faces = np.empty((self.batch_size//2, *self.dim, self.n_channels))
        # Generate data
        for i in range(self.batch_size//2):
            # Store sample
            faces[i,]    = np.expand_dims(cv2.resize(cv2.imread(self.dir_faces+list_faces[i],0)/255,self.dim),-1)
            no_faces[i,] = np.expand_dims(random_crop(no_face,self.dim),-1)

        X = np.concatenate((faces,no_faces))
        y = np.concatenate((np.repeat(1,len(faces)),np.repeat(0,len(no_faces))))
        indices = list(range(len(X)))
        np.random.shuffle(indices)

        X = np.asarray([X[i] for i in indices])
        y = np.asarray([y[i] for i in indices])

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)