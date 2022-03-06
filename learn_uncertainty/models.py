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

        model = EfficientNetB0(weights=None, classes=10, input_shape=input_shape, classifier_activation=None)
        # exit() 
    else: 
        print("no model arch or model specified, exiting")
        exit() 

    if verbose >= 3: 
        model.summary()

    return model 