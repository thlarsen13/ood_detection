import numpy as np 
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.applications import *
from models import get_model
import cv2

prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'

# ex:/home/thlarsen/ood_detection/distribution_shifts/cifar_c/labels.npy
distribution_shift_names = {
"gaussian_noise" : 'Gaussian Noise',
"shot_noise" : 'Shot Noise',
"impulse_noise": 'Impulse Noise',
"defocus_blur": 'Defocus Blur',
"glass_blur" : 'Glass Blur',
"motion_blur": 'Motion Blur',
"zoom_blur": 'Zoom Blur',
"snow" : 'Snow',
"fog" : 'Fog',
"brightness" : 'Brightness',
"contrast" : 'Contrast',
"elastic_transform" : 'Elastic',
"pixelate" : 'Pixelate',
"jpeg_compression" : 'JPEG',
"speckle_noise" : 'Speckle Noise',
"gaussian_blur" : 'Gaussian Blur',
"spatter" : 'Spatter',
"saturate" : 'Saturate'}


height = 224
width = 224
channels = 3

def resize_img(img):
    return cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)

def numpy_resize_imgs(imgs): 
    return resize_imgs(imgs, from_tensor=False)

def resize_imgs(imgs, from_tensor=True): 
    if from_tensor:
        imgs=imgs.numpy()

    new_imgs = np.empty((imgs.shape[0], height, width, channels))
    for i in range(imgs.shape[0]): 
        new_imgs[i] = resize_img(imgs[i])
    return new_imgs


def load_dataset_c(method_name, dataset):
    if dataset not in ['mnist_c', 'cifar_c']: 
        print(f"Error, {dataset} not in [mnist, cifar]")
    if method_name not in distribution_shift_names.keys(): 
        print(f"Error, {method_name} not a valid shift")
    folder_path = f'/home/thlarsen/ood_detection/distribution_shifts/{dataset}/'
    data = np.load(folder_path + method_name + '.npy')
    labels = np.load(folder_path + 'labels.npy')
    sev = np.load(folder_path + method_name + '_sev.npy')
    return data, labels, sev

#ideally, make classes that handle this correctly


def load_dataset_sev(method_name, dataset='cifar'):
    data, labels, sev = None, None, None
    if dataset == 'cifar':
        data, labels, sev = load_dataset_c(method_name, 'cifar_c')
    elif dataset == 'mnist': 
        data, labels, sev = load_dataset_c(method_name, 'mnist_c')
    else: 
        print("Error, dataset not in {mnist, cifar}")
        exit()

    step = 10000
    data_by_sev = {}
    for i in range(0, data.shape[0], step): 
        data_by_sev[sev[i]] = [data[i:i+step], labels[i:i+step]]
    return data_by_sev

def load_cifar_model(lr = 10**-3, w = 1, train_alg='ece', model_arch='EfficientNetB0Transfer'): 
    cifar_prefix = f'{prefix}saved_weights/cifar_calibrate/'
    model_save_path = None
    trn = lambda x: x

    model = get_model(model_arch=model_arch, verbose=1)

    if train_alg == 'ece_shift' and model_arch == 'EfficientNetB0Transfer':
        model_save_path = f'{cifar_prefix}trn_2cal(lr={lr})(w={w})'
        trn = numpy_resize_imgs  
    elif train_alg == 'ece_shift' and model_arch == 'EfficientNetB0':
        model_save_path = f'{cifar_prefix}2_cal(lr={lr})(w={w})'
    elif train_alg == 'ece' and model_arch == 'EfficientNetB0Transfer': 
        trn = numpy_resize_imgs  
        model_save_path = f'{cifar_prefix}trn_cal(lr={lr})(w={w})'
    elif train_alg == 'ece' and model_arch == 'EfficientNetB0':  
        model_save_path = f'{cifar_prefix}B0(lr={lr})(w={w})'
    elif train_alg == 'ece' and model_arch == 'EfficientNetB2':  
        model_save_path = f'{cifar_prefix}B2(lr={lr})(w={w})'
    else: 
        print(f'error: {train_alg} has not been supported')

    model.load_weights(model_save_path + '.h5')

    #todo add support for models saved differently

    return model, trn




# def load_model_with_saved_weights(model): 

#     model = get_model(self.model_arch, self.input_shape, self.n_classes, self.verbose)  

#     model.load_weights(self.model_save_path)



def load_mnist_model(lr = 10**-3, w = 1, train_alg='ece', model_arch='conv'): 
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'

    if model_arch == 'conv': 
        if train_alg == 'ece':
           model_save_path = f'{prefix}saved_weights/mnist_calibrate/conv(lr={lr})(w={w})'
        elif train_alg == 'ece_shift':
            model_save_path = f'{prefix}saved_weights/mnist_calibrate/sh_conv(lr={lr})(w={w})'
    elif model_arch == 'seq': 
        if train_alg == 'ece':
            model_save_path = f'{prefix}saved_weights/mnist_calibrate/cal(lr={lr})(w={w})'
        elif train_alg == 'ece_shift':
            model_save_path = f'{prefix}saved_weights/mnist_calibrate/sh_cal(lr={lr})(w={w})'
    

    model = keras.models.load_model(model_save_path)
    return model 
def rgb_img_to_vec(x): 
    x = np.dot(x, [0.299, 0.587, 0.114])
    # print(x.shape)
#     for i in range(30, 40):
#         print(f'i={i}, y={y_test[i]}, mean={np.mean(x[i])}')
#         print(x[i])
#         plt.imshow(x[i])
#         plt.show()

    x = np.reshape(x, (-1, 32*32))
    return x.astype('float64')


full_distribution_shifts = ["gaussian_noise",
"shot_noise",
"impulse_noise",
"motion_blur",
"brightness",
"contrast",
"pixelate",
"jpeg_compression",
"speckle_noise",
"gaussian_blur",]
