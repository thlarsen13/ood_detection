import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
import time 
from datetime import datetime
from tensorflow.keras.applications import *
from tqdm import tqdm
from train_loop import trainBuilder
# from shift_train_loop import train_attempt_shift

from helper import load_dataset_c

verbose = False

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
input_shape = x_train[0].shape

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

data, labels, sev = load_dataset_c('contrast', 'cifar_c')


train_dataset_shift = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset_shift = train_dataset_shift.batch(batch_size)


def main(): 
    shift = False 

    # weights = [10 **i for i in range(-3, 1)]
    # learning_rates = [10**i for i in range(-5, -2)]

    weights = [0]
    learning_rates = [10**-4]
    # weights = [.1]
    # learning_rates = [10**-3]

    overall_results = [['l/w']+ weights]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 50
    acc, ece = None, None 
    with tqdm(total=len(learning_rates) * len(weights)) as pbar:
        for lr in learning_rates: 
            overall_results.append([lr])
            for w in weights:
                model = None
               
                if shift: 
                    model_save_path = f'{prefix}saved_weights/cifar_calibrate/2_cal(lr={lr})(w={w}).h5'
                    graph_path = f'{prefix}training_plots/cifar_calibrate/2_cal(lr={lr})(w={w}).png'

                    acc, ece = train_attempt_shift(model_save_path, train_dataset, train_dataset_shift, val_dataset, 
                                        input_shape=input_shape,
                                        lr=lr, w=w, epochs=epochs, 
                                        graph_path=graph_path,
                                        verbose=True)
                else: 
                    model_save_path = f'{prefix}saved_weights/cifar_calibrate/cal(lr={lr})(w={w}).h5'
                    graph_path = f'{prefix}training_plots/cifar_calibrate/cal(lr={lr})(w={w}).png'

                    acc, ece = train_attempt(input_shape, train_dataset, val_dataset, 
                                        lr=lr, w=w, epochs=epochs, 
                                        graph_path=graph_path,
                                        verbose=True, 
                                        model_save_path=model_save_path)

                overall_results[-1].append((acc, ece))
                pbar.update(1)

    #array nonsense to make it print the array in a readable format
    s = [[str(e) for e in row] for row in overall_results]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
if __name__ == "__main__": 
    main()







