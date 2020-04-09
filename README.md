# deep_learning_classes
 


---

mnist_dnn.py: A simple deep neural net using Keras & Tensorflow. The model was trained on the MNIST dataset and classifies a black background, grayscale handwritten digit (28x28 pixels) with about 96% accuracy.


mnist_cnn.py: Very similar to the mnist_dnn above, but restructures inputs to use convolution layers. Based on the LeNet CNN architecture. Accuracy of 99%, but could still use some improvement (not very accurate for off center letters, 9s vs 4s, etc).


traffic_sign_classification: uses a dataset of German traffic signs from bitbucket to train a cnn for image classification. There are 43 different classes and the data is preprocessed to 32x32 grayscale images. Generated an augmented dataset from original. Trains in about 6-8 mins with gpu enabled tensorflow & a GTX950. Final model accuracy of around 97%.


self_driving.py: uses a nvidia model CNN to control steering angle of vehicle based on live image data. Used with Udacity Vehicle simulator.


polynomial_regression_ex.py: example from class of utilizing polynomial regression.

