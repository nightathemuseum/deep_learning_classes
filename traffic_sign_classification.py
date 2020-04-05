# -*- coding: utf-8 -*-
"""
nightathemuseum

Classifying German traffic signs using convolutional neural nets.

Part of the "The Complete Self-Driving
Car Course - Applied Deep Learning" on
Udemy.
"""
# original command: !git clone https://bitbucket.org/jadslim/german-traffic-signs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical #for multiclass datasets

np.random.seed(0)

#Use bitbucket repository to access traffic sign dataset

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f) #unpickling data

with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f) #unpickling data

with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f) #unpickling data
    
x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']


#do some quick checks for consistency
assert(x_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(x_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
assert(x_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(x_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32 RGB."
assert(x_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32 RGB."
assert(x_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32 RGB."

#import the data
data = pd.read_csv('german-traffic-signs/signnames.csv')
num_samples = []
cols = 5
num_classes = 43

#preprocess images with opencv
import cv2

plt.imshow(x_train[1000])
plt.axis('off')

#convert RGB image to grayscale
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

img = grayscale(x_train[1000])
plt.imshow(img, cmap=plt.get_cmap('Greys'))
plt.axis('off')
print(img.shape) #confirm that image is now 2D

#standardize image lighting using equalization
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

img = equalize(img)
plt.imshow(img, cmap=plt.get_cmap('Greys'))
plt.axis('off')

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    #normalize pixel intensity to between 0 & 1
    img = img/255
    return img

#preprocess training data set
x_train = np.array(list(map(preprocessing, x_train)))
#do the same with others
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))

plt.imshow(x_train[random.randint(0,len(x_train)-1)], cmap=plt.get_cmap('Greys'))
plt.axis('off')
#reshape data for model
x_train = x_train.reshape(34799, 32, 32, 1)
x_val = x_val.reshape(4410, 32, 32, 1)
x_test = x_test.reshape(12630, 32, 32, 1)

#one hot encode data
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

#define the LeNet model function
def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(Conv2D(60, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
    return model   

#build and train the model, print summary of model
model = modified_model()
history = model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=10, batch_size=400, verbose=1, shuffle=1)
print(model.summary())

#plot the loss
plt.figure(0)
plt.subplot(2,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #validation loss. If val_loss starts getting high, that means you're overfitting!
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('Epochs')

#plot the accuracy
plt.subplot(2,1,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.tight_layout()


#evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print(type(score))
print('Test score:', score[0])
print('Test accuracy:', score[1])

#import image
from PIL import Image


#make prediction using model
prediction = model.predict_classes(x_test[random.randint(0,len(x_test)-1)])
print("Predicted class: ",str(prediction))


