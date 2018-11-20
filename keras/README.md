# Keras

## Introduction

This is a Keras implementation of a Convolutional Neural Network (CNN) for digit classification using the MNIST dataset.

The dataset is composed of 
	- 54,000 training samples
	- 10,000 testing samples
	- 6,000 validation samples

Using TensorFlow as the backend, Keras automatically runs the computations on as many cores as available.

## Architecture

The input image has a shape of 28x28x1 (only one color channel since images are black and white). 
The model begins with 3 convolutional layers alternated with max pooling layers. All of them use the ReLU activation function.
In each convolutional layer the number of filters is duplicated (16, 32, 64). The pool_size of every Max Pooling layer is 2. 
After the CNN, a flatten layer is used to give the input to a neural network layer. The 576 output neurons of the flatten layer are connected with 500 neurons. That is activated with ReLU and connected with a final, dense layer with 10 (number of classes) neurons, and a softmax activation function which is useful for classification.

_________________________________________________________________
Layer (type)                 Output Shape              Param number  
conv2d_1 (Conv2D)            (None, 28, 28, 16)        80        
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 16)        0         
conv2d_2 (Conv2D)            (None, 14, 14, 32)        2080      
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 32)          0         
conv2d_3 (Conv2D)            (None, 7, 7, 64)          8256      
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
flatten_1 (Flatten)          (None, 576)               0         
dense_1 (Dense)              (None, 500)               288500    
dense_2 (Dense)              (None, 10)                5010      
Total params: 303,926
_________________________________________________________________

## Hyperparameters
- Batch size: 100
- Epochs: 10
- Loss: Categorical Crossentropy
- Stochastic Gradient Descent

## Results

Total time: 6m 49s 
Validation Accuracy: 0.9807
Test Accuracy: 0.9799



