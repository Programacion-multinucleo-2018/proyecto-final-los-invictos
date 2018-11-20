import sys
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

sys.path.append('../utils')

from extract import *

def build_model():
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(filters=16,
                    kernel_size=2, 
                    padding='same', 
                    activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32,
                    kernel_size=2,
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64,
                    kernel_size=2, 
                    padding='same', 
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Deep Neural Network
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    return model

if __name__ == "__main__":
    # Get data
    x_train, y_train, x_valid, y_valid, x_test, y_test = extract(reshape_input=True)

    # Build and compile model
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Fit data for training set
    print('fitting')
    hist = model.fit(x_train, y_train, batch_size=1, epochs=5,
          validation_data=(x_valid, y_valid), 
          verbose=1, shuffle=True)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])