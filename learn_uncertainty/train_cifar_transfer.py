#!/usr/bin/env python
# coding: utf-8

# # CIFAR-100 Image Classification

# ## Importing the Libraries

# In[2]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D 
from tensorflow.keras.layers import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.transform import resize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import seaborn as sns
import cv2

from tqdm import tqdm 
from tensorflow.keras.applications import *
from calibration_stats import ExpectedCalibrationError
from train_loop import TrainBuilder


# In[3]:


## Loading the CIFAR-10 Dataset

batch_size = 64
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# input_shape = x_train[0].shape
N = x_train.shape[0]

#resizing the images as per EfficientNetB0 to size (224, 224)
height = 224
width = 224
channels = 3

n_classes = 10
input_shape = (height, width, channels)


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

    
def resize_img(img):
    return cv2.resize(img.numpy(), (height, width), interpolation=cv2.INTER_CUBIC)

def resize_imgs(imgs): 
    new_imgs = np.empty((imgs.shape[0], height, width, channels))
    for i in range(imgs.shape[0]): 
        new_imgs[i] = resize_img(imgs[i])
    return new_imgs


def main(): 
    shift = False 

    # weights = [10 **i for i in range(-3, 1)]
    # learning_rates = [10**i for i in range(-5, -2)]

    weights = [0, .1, .01]
    learning_rates = [10**-4]
    # weights = [.1]
    # learning_rates = [10**-3]

    overall_results = [['l/w']+ weights]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 1
    acc, ece = None, None 
    with tqdm(total=len(learning_rates) * len(weights)) as pbar:
        for lr in learning_rates: 
            overall_results.append([lr])
            for w in weights:
                model = None
               
                model_save_path = f'{prefix}saved_weights/cifar_calibrate/trn_cal(lr={lr})(w={w}).h5'
                graph_path = f'{prefix}training_plots/cifar_calibrate/trn_cal(lr={lr})(w={w}).png'

                builder = TrainBuilder(input_shape=input_shape,
                                    lr=lr, w=w, epochs=epochs, 
                                    graph_path=graph_path,
                                    model_save_path=model_save_path,
                                    transform = resize_imgs, 
                                    verbose=3)
                acc, ece = builder.train_attempt(train_dataset, val_dataset) 


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




# efnb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)

# model = Sequential()
# model.add(efnb0)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.5))
# model.add(Dense(n_classes))

# model.summary()

# w = .1


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# verbose = False
# cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# ese_loss_fn = ExpectedCalibrationError(weight=w)


# epochs = 15

# ECE_dict = { i : [] for i in range(epochs)}
# ACC = []



# def loss_fn(y_batch_train, logits, verbose=False): 
#     # print(y_batch_train.shape)
#     # print(logits.shape)
#     # exit()
#     cce = cce_loss_fn(y_batch_train, logits) 
#     ese = tf.constant(0)
#     if w != 0: #avoid evaluating this if w != 0
#         ese = ese_loss_fn(y_batch_train, logits)
#     if verbose:
#         # print(f'w={w}')
#         print(f'Training cce, ese, loss (for one batch): {cce:.4f}, {ese:.4f}, {cce+ese:.4f}')

#     ECE_dict[epoch].append(ese.numpy())
#     return tf.add(cce, ese)

# # Prepare the metrics.
# train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# for epoch in tqdm(range(epochs)):
#     if verbose: 
#         print("\nStart of epoch %d" % (epoch,))
# #     start_time = time.time()

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

#         with tf.GradientTape() as tape:
#             logits = model(resize_imgs(x_batch_train), training=True)
#             loss_value = loss_fn(y_batch_train, logits, verbose=verbose)
#         grads = tape.gradient(loss_value, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # Update training metric.
#         train_acc_metric.update_state(y_batch_train, logits)

#     # Display metrics at the end of each epoch.
#     train_acc = train_acc_metric.result()
#     if verbose: 
#         print("Training acc over epoch: %.4f" % (float(train_acc),))
#     ACC.append(train_acc.numpy())
#     # Reset training metrics at the end of each epoch
#     train_acc_metric.reset_states()

#     # Run a validation loop at the end of each epoch.
#     for x_batch_val, y_batch_val in val_dataset:
#         val_logits = model(resize_imgs(x_batch_val), training=False)
#         # Update val metrics
#         val_acc_metric.update_state(y_batch_val, val_logits) 
#         #TODO: write custom update state for ece 

#     val_acc = val_acc_metric.result()
#     val_acc_metric.reset_states()
#     if verbose: 
#         print("Validation acc: %.4f" % (float(val_acc),))
# #         print("Time taken: %.2fs" % (time.time() - start_time))


# if graph_path is not None: 
#     ECE = [np.mean(ECE_dict[i]) for i in range(epochs)]
#     print(f"\n\n\n@@@ {graph_path}\n ECE ({len(ECE)}): {ECE} \nACC ({len(ACC)}): {ACC}")
# if model_save_path is not None: 
#     # model.compile(optimizer="Adam", loss=tf.keras.losses.CategoricalCrossentropy)
#     model.save(model_save_path)
#     np.save(model_save_path + 'opt_weights.npy', optimizer.get_weights())



