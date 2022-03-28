# import tensorflow as tf

### BNN stuff
import os
import warnings

# Dependency imports
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

#Other models
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications import *


tf.enable_v2_behavior()


def get_bnn(): 

    tfd = tfp.distributions

    IMAGE_SHAPE = [32, 32, 3]
    NUM_TRAIN_EXAMPLES = 60000
    NUM_HELDOUT_EXAMPLES = 10000
    NUM_CLASSES = 10
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                        tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    model = tf.keras.models.Sequential([
            tfp.layers.Convolution2DFlipout(
                    6, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                    16, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                    120, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                    84, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                    NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.softmax)
    ])
    return model 


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
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        # exit() 
    elif model_arch == 'conv2': 
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(10, kernel_size=(3,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10,))

    elif model_arch == 'seq':
        inputs = keras.Input(shape=(3072), name="digits")
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
        model = keras.Model(inputs=inputs, outputs=outputs)
    elif model_arch == 'bnn':
        model = get_bnn()

    else: 
        print("no model arch or model specified, exiting")
        exit() 

    if verbose >= 3: 
        model.summary()

    return model 