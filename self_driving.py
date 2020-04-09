# -*- coding: utf-8 -*-
"""
nightathemuseum

Creating a self driving car model using
Udacity's autonomous vehicle simulator.

Part of the "The Complete Self-Driving
Car Course - Applied Deep Learning" on
Udemy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd
import random
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datadir = 'driving_data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', None)



def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

print(data.head())

#create histogram of data values
num_bins = 25
sample_threshold = 200 #each bin can have max of 200 samples to reduce spikes
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:])*0.5
# plt.bar(center, hist, width=0.05)
# plt.plot((np.min(data['steering']), np.max(data['steering'])), (sample_threshold, sample_threshold))


#balance the data -> reduce amount of 0 deg steering angle data
print('Total data: ', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)    
    list_ = shuffle(list_)
    list_ = list_[sample_threshold:]
    remove_list.extend(list_)

print('removed', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining data: ', len(data))

hist, _ = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (sample_threshold, sample_threshold))

def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

#create training and validation datasets
image_paths, steerings = load_img_steering(datadir + '/IMG', data)
x_train, x_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(x_train), len(x_val)))

#preprocess images
def img_preprocess(img): #the imported object is just an img path
    img = mpimg.imread(img)
    #crop the image to remove unnecessary hood & scenery
    img = img[60:135, :, :]
    #Nvidia cnn creators recommend YUV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #apply gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3,3), 0)
    #resize image for faster computations & to match input size for Nvidia model
    img = cv2.resize(img, (200,66))
    #normalize
    img = img/255    

    return img

#preprocess data
x_train = np.array(list(map(img_preprocess, x_train)))
x_val = np.array(list(map(img_preprocess, x_val)))

#verify preprocessing of data
plt.figure(0)
plt.imshow(x_train[random.randint(0, len(x_train))])
plt.axis('off')
print(x_train.shape)

#build Nvidia model
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation = 'elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation = 'elu'))
    model.add(Conv2D(64, (3, 3), activation = 'elu'))
    model.add(Conv2D(64, (3, 3), activation = 'elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation = 'elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation = 'elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation = 'elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1)) #outputs the predicted steering angle for self driving car
    
    model.compile(loss='mse', optimizer=Adam(lr = 0.001))
    return model

model = nvidia_model()
print(model.summary())

#train the model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), batch_size=100, verbose=1, shuffle=1)

#plot loss values
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')

#save this model
model.save('model.h5')




