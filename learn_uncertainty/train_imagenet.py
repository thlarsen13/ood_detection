import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from cal_error import ExpectedCalibrationError
import time 
from datetime import datetime
from tensorflow.keras.applications import *
from tqdm import tqdm
# from train_ece_shifted import train_attempt
from train_ece_loop import train_attempt

from helper import load_cifar_c

verbose = False

#Load imagenet 
imagenet_path = "/home/thlarsen/tensorflow_datasets/imagenet/"
batch_size = 128

# Prepare the training dataset.
imagenet = tfds.image_classification.Imagenet2012(imagenet_path)
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

data, labels, sev = load_cifar_c('contrast')
train_dataset_shift = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset_shift = train_dataset_shift.batch(batch_size)

def main(): 

    # weights = [10 **i for i in range(-3, 1)]
    # learning_rates = [10**i for i in range(-5, -2)]

    weights = [0, 10**-2, .1]
    # learning_rates = [10**-3, 10**-2]
    # weights = [.1]
    learning_rates = [10**-3]

    overall_results = [['l/w']+ weights]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 60

    with tqdm(total=len(learning_rates) * len(weights)) as pbar:
        for lr in learning_rates: 
            overall_results.append([lr])
            for w in weights:
                model = EfficientNetB2(weights=None, classes=10, input_shape=input_shape, classifier_activation=None)
                model_save_path = f'{prefix}saved_weights/imagenet_calibrate/2_cal(lr={lr})(w={w})'
                graph_path = f'{prefix}training_plots/imagenet_calibrate/2_cal(lr={lr})(w={w}).png'
                acc, ece = train_attempt(model, train_dataset, train_dataset_shift, val_dataset, 
                                        lr=lr, w=w, epochs=epochs, 
                                        graph_path=graph_path,
                                        model_save_path=model_save_path, 
                                        verbose=True)
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







