

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from calibration_stats import ExpectedCalibrationError
import time 
from train_loop import TrainBuilder
from helper import load_dataset_c
from tensorflow.keras.models import Sequential

verbose = False

input_shape = (32,32, 3)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

print(x_train.shape)

print(x_test.shape)
x_train = x_train.reshape((-1, 32, 32, 1)).repeat(3, axis=3)
x_test = x_test.reshape((-1, 32, 32, 1)).repeat(3, axis=3)

print(x_train.shape)


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


data, labels, sev = load_dataset_c('contrast', 'mnist_c')

train_dataset_shift = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset_shift = train_dataset_shift.batch(batch_size)
def flatten(input_imgs): 
    # print(input_imgs.shape)
    return tf.reshape(input_imgs, (-1, 3072))

def bnn_cast(imgs): 
    return tf.cast(imgs, dtype=tf.float32)

def main(): 

    # weights = [10 **i for i in range(-3, 3)]
    # learning_rates = [10**i for i in range(-5, -1)]
    weights = [0]               
    learning_rates = [10**-3]
    overall_results = [['l/w']+weights]
    model_arch = 'bnn'

    # weights = [1]
    # learning_rates = [10**-3]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 5
    for lr in learning_rates: 
        overall_results.append([f'lr = {lr}'])
        for w in weights:
            transform = None
            if model_arch == 'bnn': 
                model_save_path = f'{prefix}saved_weights/mnist_calibrate/bnn(lr={lr})(w={w})'
                transform = bnn_cast
            elif model_arch == 'seq?':
                model_save_path = f'{prefix}saved_weights/mnist_calibrate/sh_cal(lr={lr})(w={w})'
                transform = flatten
            builder = TrainBuilder(lr=lr, w=w, epochs=epochs, 
                                    graph_path=None,
                                    model_save_path=model_save_path,
                                    transform = transform, 
                                    verbose=2, 
                                    model_arch=model_arch)
            # acc, ece = builder.shift_train_attempt(train_dataset, train_dataset_shift, val_dataset, model=None) 

            acc, ece = builder.train_attempt(train_dataset, val_dataset, model=None) 
            overall_results[-1].append((acc, ece))

    s = [[str(e) for e in row] for row in overall_results]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

if __name__ == "__main__": 
    main()

