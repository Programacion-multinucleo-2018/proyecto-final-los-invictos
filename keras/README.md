# Keras

## Introduction

This is a Keras implementation of a Convolutional Neural Network (CNN) for digit classification using the MNIST dataset.

The dataset is composed of 

	- 54,000 training samples
	
	- 10,000 testing samples
	
	- 6,000 validation samples

Using TensorFlow as the backend, Keras automatically runs the computations on as many cores as available.
_________________________________________________________________

## Architecture

The input image has a shape of 28x28x1 (only one color channel since images are black and white). 
The model begins with 3 convolutional layers alternated with max pooling layers. All of them use the ReLU activation function.
In each convolutional layer the number of filters is duplicated (16, 32, 64). The pool_size of every Max Pooling layer is 2. 
After the CNN, a flatten layer is used to give the input to a neural network layer. The 576 output neurons of the flatten layer are connected with 500 neurons. That is activated with ReLU and connected with a final, dense layer with 10 (number of classes) neurons, and a softmax activation function which is useful for classification.

![keras architecture](https://github.com/Programacion-multinucleo-2018/proyecto-final-los-invictos/blob/master/keras/keras_arch.png "Keras architecture")
_________________________________________________________________

## Hyperparameters
- Batch size: 100
- Epochs: 10
- Loss: Categorical Crossentropy
- Stochastic Gradient Descent

_________________________________________________________________
## Results

Total time: 6m 49s

Validation Accuracy: 0.9807

Test Accuracy: 0.9799



