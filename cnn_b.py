import numpy as np 
import pandas as pd 
from keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
import PIL
from PIL.ImageEnhance import Brightness 
from keras.utils.data_utils import get_file
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
import sys
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# %% [code]
def onehot(y):
    y = np.array(y)
    B = np.zeros((y.shape[0],10))
    for i in range(y.shape[0]):
        for j in range (10):
            t = y[i].astype(int)
            B[i,t] = 1
    return B

def pics(x):
    images = np.zeros((x.shape[0],32,32,3))
    for i in range(x.shape[0]):
        images[i,:,:,:] =np.transpose(x[i,:].reshape(3,32,32),(1,2,0))
    images = images.astype(np.uint8)
    return images

def cnn_a(train,test,op):
    # %% [code]
    train = pd.read_csv(train,delimiter=' ',header=None)
    test = pd.read_csv(test,delimiter=' ',header=None)
    print(train.shape)
    data = train.values
    testdata = test.values
    testdata = testdata[:,:testdata.shape[1]-1]
    print(data.shape,testdata.shape)
    [m,n] = data.shape
    ytrain = data[:,n-1]
    xtrain = data[:,:n-1]

    # %% [code]
    xtrain, ytrain = pics(xtrain), onehot(ytrain)
    print(xtrain.shape)

    # %% [code]
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))
    model.add(Conv2D(64, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(256, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))
    model.add(Conv2D(256, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(512, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))
    model.add(Conv2D(512, kernel_size=3,strides=(1,1), padding='same', activation='relu',input_shape=(32,32,3)))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
    model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    # fits the model on batches with real-time data augmentation:
    datagen1 = ImageDataGenerator(
        featurewise_center=True,brightness_range=[0.3,1.0],
	zoom_range=0.3,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
	vertical_flip=False)
    datagen1.fit(xtrain)
    history = model.fit_generator(datagen1.flow(xtrain, ytrain, batch_size=500),steps_per_epoch=len(xtrain)/500, epochs=10)
    history = model.fit(xtrain,ytrain,validation_split=0.01,epochs=10,batch_size=500)
    
    datagen2 = ImageDataGenerator(
        featurewise_center=True,brightness_range=[0.3,1.0],
	zoom_range=0.3,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
	vertical_flip=False)
    datagen2.fit(xtrain)
    history = model.fit_generator(datagen2.flow(xtrain, ytrain, batch_size=500),steps_per_epoch=len(xtrain)/500, epochs=10)
    history = model.fit(xtrain,ytrain,validation_split=0.01,epochs=10,batch_size=500)
    
    datagen3 = ImageDataGenerator(
        featurewise_center=True,
        brightness_range=[0.3,1.0],
	zoom_range=0.3,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.3,
        height_shift_range=0.0,
        horizontal_flip=True,
	vertical_flip=False)
    datagen3.fit(xtrain)
    history = model.fit_generator(datagen3.flow(xtrain, ytrain, batch_size=500),steps_per_epoch=len(xtrain)/500, epochs=10)
    history = model.fit(xtrain,ytrain,validation_split=0.01,epochs=10,batch_size=500)
    prediction = model.predict(pics(testdata))
    output = np.argmax(prediction,axis=1)
    for i in output:
        print(i,file=open(op,"a"))

if __name__ == '__main__':
    cnn_a(*sys.argv[1:])