import numpy as np 
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.applications import *

prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'

# ex:/home/thlarsen/ood_detection/distribution_shifts/cifar_c/labels.npy
distribution_shifts = {
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


def load_dataset_c(method_name, dataset):
    if dataset not in ['mnist_c', 'cifar_c']: 
        print(f"Error, {dataset} not in [mnist, cifar]")
    if method_name not in distribution_shifts.keys(): 
        print(f"Error, {method_name} not a valid shift")
    folder_path = f'/home/thlarsen/ood_detection/distribution_shifts/{dataset}/'
    data = np.load(folder_path + method_name + '.npy')
    labels = np.load(folder_path + 'labels.npy')
    sev = np.load(folder_path + method_name + '_sev.npy')
    return data, labels, sev

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

def load_cifar_model(lr = 10**-3, w = 1, train_on_shift=False): 
    model_save_path = f'{prefix}saved_weights/cifar_calibrate/cal(lr={lr})(w={w})'
    if train_on_shift: 
        model_save_path = f'{prefix}saved_weights/cifar_calibrate/2_cal(lr={lr})(w={w})'
    model = None 
    try: 
        model = keras.models.load_model(model_save_path)
    except: 
        model = EfficientNetB2(weights=None, classes=10, input_shape=(32, 32, 3), classifier_activation=None)
        model.load_weights(model_save_path + '.h5')

    return model 


def load_mnist_model(lr = 10**-3, w = 1): 
	prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
	model_save_path = f'{prefix}saved_weights/mnist_calibrate/cal(lr={lr})(w={w})'

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
