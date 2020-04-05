# -*- coding: utf-8 -*-
"""
nightathemuseum

Designing a convolutional neural net to
classify images in the MNIST dataset.

Part of the "The Complete Self-Driving
Car Course - Applied Deep Learning" on
Udemy.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical #for multiclass datasets
from keras.models import Model
import random

np.random.seed(0) #seed the rng to get repeatable results

(x_train, y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)

assert(x_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(x_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(x_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28."
assert(x_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28."

num_samples = []
cols = 5
num_classes = 10
num_pixels = 28*28

#print a sample of the images in the dataset
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10,10))
# fig.tight_layout()
# for i in range(cols):
#     for j in range(num_classes):
#         x_selected = x_train[y_train == j]
#         axs[j][i].imshow(x_selected[random.randint(0,len(x_selected-1)), :, :], cmap=plt.get_cmap("gray"))
#         axs[j][i].axis("off")
#         if i == 2:
#             axs[j][i].set_title(str(j))
        
#one hot encode the data labels 
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#now normalize data
x_train = x_train/255 #adjusts pixel intensity to be between 0-1, instead of 0-255
x_test = x_test/255


#define the LeNet model function
def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy',metrics=['accuracy'])
    return model   


#build and train the model, print summary of model
model = leNet_model()
history = model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=400, verbose=1, shuffle=1)
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


#This URL has a test image: 
#'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'

#import image
import requests
from PIL import Image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
print(response)

#convert image to array
#img = Image.open(response.raw) #uses online image
img = Image.open("test.png") #uses image file in folder
img_array = np.array(img)

#resize image to 28x28
import cv2
img_array = np.asarray(img)
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#invert image black & white, normalize
image = cv2.bitwise_not(gray_scale)/255
plt.figure(1)
plt.imshow(image, cmap=plt.get_cmap("gray"))
#reshape image for cnn
image = image.reshape(1,28,28,1)

#make prediction using model
prediction = model.predict_classes(image)
print("Predicted digit: ",str(prediction))

#Visualize the filters and classes of model
layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)

visual_layer1, visual_layer2 = layer1.predict(image), layer2.predict(image)

plt.figure(2)
plt.figure(figsize=(10,6))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(visual_layer1[0, :, :, n], cmap=plt.get_cmap('jet'))
    plt.axis('off')
    
plt.figure(3)
plt.figure(figsize=(10,6))
for n in range(15):
    plt.subplot(3,5,n+1)
    plt.imshow(visual_layer2[0, :, :, n], cmap=plt.get_cmap('jet'))
    plt.axis('off')