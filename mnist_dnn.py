# -*- coding: utf-8 -*-
"""
nightathemuseum

Implementing a deep neural net for image
recognition on MNIST dataset.

Part of the "The Complete Self-Driving
Car Course - Applied Deep Learning" on
Udemy.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.dataset import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical #for multiclass datasets
import random

np.random.seed(0) #seed the rng to get repeatable results

(x_train, y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)



