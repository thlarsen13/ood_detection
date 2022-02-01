import numpy as np 


folder_path = '/home/thlarsen/ood_detection/distribution_shifts/mnist_c/'

def load_mnist_c(method_name): 
	data = np.load(folder_path + d[method_name].__name__ + '.npy')
	labels = np.load(folder_path + 'labels.npy')
    	sev = np.load(folder_path + d[method_name].__name__ + '_sev.npy')
	return data, labels, sev
