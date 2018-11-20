import keras
import numpy as np
from keras.utils import np_utils

def preprocess_features(images):
    # Convert pixel values from 0-255 to 0-1
    images = images.astype('float32')/255
    return images


def one_hot_label(labels, num_classes=10, log=False):
    labels = keras.utils.to_categorical(labels, num_classes)

    if log == True:
	    print('One-hot encoded labels.')
	    print('Example image label', labels[0])
    return labels
