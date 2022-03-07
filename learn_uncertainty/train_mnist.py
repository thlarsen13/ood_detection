

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from calibration_stats import ExpectedCalibrationError
import time 
from train_loop import TrainBuilder

from tensorflow.keras.models import Sequential

verbose = False

input_shape = (32,32)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

print(x_test.shape)
# x_train = np.reshape(x_train, (-1, input_dim))
# x_test = np.reshape(x_test, (-1, input_dim))
# print(x_test.shape)

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


def main(): 

    # weights = [10 **i for i in range(-3, 3)]
    # learning_rates = [10**i for i in range(-5, -1)]
    weights = [.1, 0]               
    learning_rates = [10**-4]
    overall_results = [['l/w']+weights]

    # weights = [1]
    # learning_rates = [10**-3]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 30
    for lr in learning_rates: 
        overall_results.append([f'lr = {lr}'])
        for w in weights:

            # inputs = keras.Input(shape=(input_dim,), name="digits")
            # x1 = layers.Dense(64, activation="relu")(inputs)
            # x2 = layers.Dense(64, activation="relu")(x1)
            # outputs = layers.Dense(10, name="predictions")(x2)
            # model = keras.Model(inputs=inputs, outputs=outputs)


            model_save_path = f'{prefix}saved_weights/mnist_calibrate/conv(lr={lr})(w={w})'
            graph_path = f'{prefix}training_plots/mnist_calibrate/conv(lr={lr})(w={w}).png'

            builder = TrainBuilder(input_shape=input_shape,
                                    lr=lr, w=w, epochs=epochs, 
                                    graph_path=graph_path,
                                    model_save_path=model_save_path,
                                    transform = None, 
                                    verbose=3, 
                                    model_arch='conv')
            acc, ece = builder.train_attempt(train_dataset, val_dataset, model=None) 
            overall_results[-1].append((acc, ece))

    s = [[str(e) for e in row] for row in overall_results]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

if __name__ == "__main__": 
    main()

