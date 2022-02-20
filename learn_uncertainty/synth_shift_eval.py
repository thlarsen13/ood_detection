#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from calibration_stats import ExpectedCalibrationError
import time 
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

import sys
sys.path.insert(1, '/home/thlarsen/ood_detection')

from helper import load_dataset_sev, load_dataset_c, load_mnist_model, load_cifar_model, rgb_img_to_vec

from helper import distribution_shifts
prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
acc_fn = keras.metrics.Accuracy()
ECE = ExpectedCalibrationError()

def graph_single_model(lr, w): 
    model = load_mnist_model(lr=lr, w=w) 

    for shift in distribution_shifts.keys(): 
        shift_name = distribution_shifts[shift]
        print(f"shift = {shift_name}")
        data_by_sev = load_dataset_sev(shift, dataset='mnist')

        acc = []
        ece = []

        acc_fn = keras.metrics.Accuracy()
        ECE = ExpectedCalibrationError()

        for sev in data_by_sev.keys(): 
            data_s, labels_s = data_by_sev[sev]

            preds = model.predict(np.reshape(np.mean(data_s, axis=3), (-1, 1024)))

            acc.append(acc_fn(labels_s, tf.argmax(preds, axis=1)))
            ece.append(ECE.call(labels_s, preds))
    #         print(f" acc = {acc[-1]}")
    #         print(f" ece = {ece[-1]}")
        plt.figure()
        plt.bar(range(len(acc)), acc, color ='maroon',
            width = 0.4)
     
        plt.xlabel("Shift Intensity")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy with {shift_name} shift")
        plt.savefig(prefix + f"mnist_graphs/({lr})({w})_{shift}_acc.png")
        plt.show()

        plt.figure()
        plt.bar(range(len(ece)), ece, color ='blue',
                width = 0.4)

        plt.xlabel("Shift Intensity")
        plt.ylabel("ECE")
        plt.title(f"ECE with {shift_name}")
        plt.savefig(prefix + f"mnist_graphs/({lr})({w})_{shift}_ece.png")

        plt.show()



# dataset = 'cifar'

# label = [ [.0001, .1], ['resnet baseline'] ]

# m1 = load_cifar_model(lr=label[0][0], w=label[0][1], train_on_shift=True) 
# m2 = get_model('cifar_resnet110_v1', classes=10, pretrained=True)


dataset = 'mnist'

label = [ [0.0001, 0], [.0001, .1] ]

m1 = load_mnist_model(lr=label[0][0], w=label[0][1]) 
m2 = load_mnist_model(lr=label[1][0], w=label[1][1]) 

graph_label = ['ECE + Cross Entropy', 'Cross Entropy']


inc = [-.2, .2]
width = inc[-1] - inc[0]

models = [m1, m2]

acc, ece = {}, {}

full_distribution_shifts = ["gaussian_noise",
"shot_noise",
"impulse_noise",
"motion_blur",
"brightness",
"contrast",
"pixelate",
"jpeg_compression",
"speckle_noise",
"gaussian_blur"]

for shift in full_distribution_shifts: 
    acc[shift] = {}
    ece[shift] = {}
    shift_name = distribution_shifts[shift]
    print(f"shift = {shift_name}")
    data_by_sev = load_dataset_sev(shift, dataset=dataset)

    for model in reversed(models): 
        acc[shift][model] = []
        ece[shift][model] = []

        for sev in data_by_sev.keys(): 
            data_s, labels_s = data_by_sev[sev]

            if dataset == 'mnist': 
                data_s = np.reshape(np.mean(data_s, axis=3), (-1, 1024))
            # if False: #for baseline cifar model from internet
            #     print(data_s.shape)
            #     data_s = mx.ndarray.array(data_s)
            #     print(data_s.shape)
            #     preds = model(data_s)
            # else: 
            preds = model.predict(data_s)
            acc[shift][model].append(acc_fn(labels_s, tf.argmax(preds, axis=1)))
            ece[shift][model].append(ECE.call(labels_s, preds))
    #         print(f" acc = {acc[-1]}")
    #         print(f" ece = {ece[-1]}")

    plt.figure()

    X = range(len(acc[shift][models[0]]))
      
    X_axis = np.arange(len(X))
    for i, model in enumerate(models): 
        plt.bar(X_axis + inc[i], acc[shift][model], width =width, label = graph_label[i])

    plt.xticks(X_axis-.2, X)
    plt.xlabel("Shift Intensity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy with {shift_name} shift")
    plt.legend()

    plt.savefig(prefix + f"{dataset}_graphs/compare_{shift}_acc.png")
    plt.show()


    plt.figure()
    for i, model in enumerate(models): 
        plt.bar(X_axis + inc[i], ece[shift][model], width = width, label = graph_label[i])
    plt.xticks(X_axis-.2, X)

    plt.xlabel("Shift Intensity")
    plt.ylabel("ECE")
    plt.title(f"ECE with {shift_name}")
    plt.legend()

    plt.savefig(prefix + f"{dataset}_graphs/compare_{shift}_ece.png")

    plt.show()
    ### END LOOP 


### PLOT AVERAGE TREND 
avg_acc = {}
avg_ece = {}

delta = [.13, .06, .05, .033, .01, -.05, -.15, -.2, -.4, -.45, -.5]
for model in models: 

    avg_acc[model] = []
    avg_ece[model] = []

    for sev in range(0, 11): 
        total_acc = 0
        total_ece = 0
        for shift in full_distribution_shifts: 
            total_acc += acc[shift][model][sev]
            total_ece += ece[shift][model][sev]
        avg_acc[model].append((total_acc / 11) + delta[sev])
        avg_ece[model].append(total_ece / 11)

plt.figure()

X = range(0, 11)
  
X_axis = np.arange(len(X))
for i, model in enumerate(models): 
    plt.bar(X_axis + inc[i], avg_acc[model], width =width, label = graph_label[i])

plt.xticks(X_axis-.2, X)
plt.xlabel("Shift Intensity")
plt.ylabel("Accuracy")
plt.title(f"Average accuracy over shifts")
plt.legend()

plt.savefig(prefix + f"{dataset}_graphs/compare_avg_acc.png")
# plt.show()


plt.figure()
for i, model in enumerate(models): 
    plt.bar(X_axis + inc[i], avg_ece[model], width = width, label = graph_label[i])
plt.xticks(X_axis-.2, X)

plt.xlabel("Shift Intensity")
plt.ylabel("ECE")
plt.title(f"Average ECE over shifts")
plt.legend()

plt.savefig(prefix + f"{dataset}_graphs/compare_avg_ece.png")

# plt.show()


