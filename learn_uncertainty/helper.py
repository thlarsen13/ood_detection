import numpy as np 
from tensorflow import keras
import tensorflow as tf

def foo(): 
    return "foo" 

def load_mnist_c(method_name): 
    return load_dataset_c(method_name, 'mnist_c')
def load_cifar_c(method_name): 
    return load_dataset_c(method_name, 'cifar_c')


# ex: /home/thlarsen/ood_detection/distribution_shifts/cifar_c/labels.npy

def load_dataset_c(method_name, dataset):
    folder_path = f'/home/thlarsen/ood_detection/distribution_shifts/{dataset}/'
    data = np.load(folder_path + method_name + '.npy')
    labels = np.load(folder_path + 'labels.npy')
    sev = np.load(folder_path + method_name + '_sev.npy')
    return data, labels, sev

def load_cifar_c_sev(method_name):
    data, labels, sev = load_cifar_c(method_name)
    step = 10000
    data_by_sev = {}
    for i in range(0, data.shape[0], step): 
        data_by_sev[sev[i]] = [data[i:i+step], labels[i:i+step]]
    return data_by_sev

def load_cifar_model(lr = 10**-3, w = 1, train_on_shift=False): 
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    model_save_path = f'{prefix}saved_weights/cifar_calibrate/cal(lr={lr})(w={w})'
    if train_on_shift: 
        model_save_path = f'{prefix}saved_weights/cifar_calibrate/2_cal(lr={lr})(w={w})'

    model = keras.models.load_model(model_save_path)
    return model 


def load_mnist_by_sev(method_name): 
    data, labels, sev = load_mnist_c(method_name)
    step = 10000
    mnist_by_sev = {}
    for i in range(0, data.shape[0], step): 
    	mnist_by_sev[sev[i]] = [data[i:i+step], labels[i:i+step]]
    return mnist_by_sev

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

distribution_shifts = [
"gaussian_noise",
"shot_noise",
"impulse_noise",
"defocus_blur",
"glass_blur",
"motion_blur",
"zoom_blur",
"snow",
"fog",
"brightness",
"contrast",
"elastic_transform",
"pixelate",
"jpeg_compression",
"speckle_noise",
"gaussian_blur",
"spatter",
"saturate"]

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
