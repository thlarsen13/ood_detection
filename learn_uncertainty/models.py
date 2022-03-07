import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D 
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications import *


def get_model(model_arch, verbose): 

    """

    Different model architectures to try
    """

    model = None 

    if model_arch == 'EfficientNetB0Transfer': 

        input_shape = (224, 224, 3)
        n_classes = 10

        efnb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)

        model = Sequential()
        model.add(efnb0)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(n_classes))

    elif model_arch == 'EfficientNetB0': 

        model = EfficientNetB0(weights=None, classes=10, input_shape=(32, 32, 3), classifier_activation=None)
    elif model_arch == 'EfficientNetB2': 
        model = EfficientNetB2(weights=None, classes=10, input_shape=(32, 32, 3), classifier_activation=None)
    elif model_arch == 'conv':
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation=tf.nn.softmax))
        # exit() 
    else: 
        print("no model arch or model specified, exiting")
        exit() 

    if verbose >= 3: 
        model.summary()

    return model 