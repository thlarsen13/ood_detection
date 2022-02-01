import numpy as np 
from tensorflow import keras
import tensorflow as tf

folder_path = '/home/thlarsen/ood_detection/distribution_shifts/mnist_c/'

def load_mnist_c(method_name): 
    data = np.load(folder_path + method_name + '.npy')
    labels = np.load(folder_path + 'labels.npy')
    sev = np.load(folder_path + method_name + '_sev.npy')
    return data, labels, sev
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

	

