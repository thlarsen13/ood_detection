

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
from tqdm import tqdm
import pickle 
# from model import get_model

import sys
sys.path.insert(1, '/home/thlarsen/ood_detection')

from helper import load_dataset_sev, load_dataset_c, load_mnist_model, load_cifar_model, rgb_img_to_vec

from helper import distribution_shift_names
prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'

sevs = 11

inc = [-.3,0,.3]
width = .3

def dump_eval_to_pickle(models, model_ids, transforms, full_distribution_shifts,dataset,  verbose=3):


    num_batches = 100
    num_test = 10000 
    m = num_test // num_batches

    # acc_fn = keras.metrics.Accuracy()
    # ECE = ExpectedCalibrationError()
    acc = {}
    ece = {}

    #10 shift severities + unshifted = 11 total shift magnitudes
    with tqdm(total=len(full_distribution_shifts) * len(models) * sevs) as pbar:

        for shift in full_distribution_shifts: 
            acc[shift] = {}
            ece[shift] = {}


            shift_name = distribution_shift_names[shift]
            print(f"shift = {shift_name}")
            data_by_sev = load_dataset_sev(shift, dataset=dataset)

            for idx, model in enumerate(models): 
                model_id = model_ids[idx]
                acc[shift][model_id] = []
                ece[shift][model_id] = []

                for sev in data_by_sev.keys(): 
                    # if sev > 1: 
                    #     continue
                    batch_acc, batch_ece = [], []
                    for batch in range(num_batches): #attempted fix for OOM errors
                        
                        data_s, labels_s = data_by_sev[sev]

                        data_s = data_s[batch*m:(batch+1)*m]
                        labels_s = labels_s[batch*m:(batch+1)*m]
                        tran = transforms[model_id]

                        preds = model.predict(tran(data_s))

                        acc_fn = keras.metrics.Accuracy()
                        ECE = ExpectedCalibrationError()
                        batch_acc.append(acc_fn(labels_s, tf.argmax(preds, axis=1)))
                        batch_ece.append(ECE.call(labels_s, preds))

                    acc[shift][model_id].append(np.mean(np.array(batch_acc)))
                    ece[shift][model_id].append(np.mean(np.array(batch_ece)))
                    if verbose >= 2:
                        print(f"sev = {sev}")
                        print(f" acc = {acc[shift][model_id][-1]}")
                        print(f" ece = {ece[shift][model_id][-1]}")
                    pbar.update(1)
    # generate_shift_graph(acc, ece, shift)
    with open(f'{prefix}{dataset}_eval_save/{model_ids}ece.pickle', 'wb') as handle:
        pickle.dump(ece, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{prefix}{dataset}_eval_save/{model_ids}acc.pickle', 'wb') as handle:
        pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f'dumped data to {prefix}{dataset}_eval_save/ece.pickle and {prefix}{dataset}_eval_save/acc.pickle')
        # plt.show()
        ### END LOOP 

def load_eval_from_pickle(model_ids, dataset): 
    
    with open(f'{prefix}{dataset}_eval_save/{model_ids}acc.pickle', 'rb') as handle:
        acc = pickle.load(handle)

    with open(f'{prefix}{dataset}_eval_save/{model_ids}ece.pickle', 'rb') as handle:
        ece = pickle.load(handle)
    # print(f'read data from {prefix}{dataset}_eval_save/ece.pickle and {prefix}{dataset}_eval_save/acc.pickle')

    return acc, ece  

def merge_pickles(model_ids_to_add, model_ids, dataset, distribution_shifts):
    with open(f'{prefix}{dataset}_eval_save/acc.pickle', 'rb') as handle:
        acc = pickle.load(handle)

    with open(f'{prefix}{dataset}_eval_save/ece.pickle', 'rb') as handle:
        ece = pickle.load(handle)
    
    with open(f'{prefix}{dataset}_eval_save/{model_ids_to_add}acc.pickle', 'rb') as handle:
        acc_new = pickle.load(handle)

    with open(f'{prefix}{dataset}_eval_save/{model_ids_to_add}ece.pickle', 'rb') as handle:
        ece_new = pickle.load(handle)

    # print(acc.keys())
    print(acc_new.keys())
    print(ece_new.keys())
    # print(acc['gaussian_noise'].keys())
    # print(acc_new['gaussian_noise'].keys())
    # exit()
    for shift in distribution_shifts: 
        for model_id in model_ids_to_add: 
            if model_id in ece[shift].keys():
                print(ece[shift].keys())
                print(model_id)
                print("would overwrite data, exiting")
                exit() 
            # print(acc_new[shift])
            # print(ece_new[shift])

            ece[shift][model_id] = ece_new[shift][model_id]
            acc[shift][model_id] = acc_new[shift][model_id]

    with open(f'{prefix}{dataset}_eval_save/{model_ids}ece.pickle', 'wb') as handle:
        pickle.dump(ece, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{prefix}{dataset}_eval_save/{model_ids}acc.pickle', 'wb') as handle:
        pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    print('dumped to ' + f'{prefix}{dataset}_eval_save/{model_ids}ece.pickle')  


def generate_shift_graph(model_ids, acc, ece, shift, dataset):

    shift_name = distribution_shift_names[shift]

    plt.figure()
    X = range(len(acc[shift][model_ids[0]]))
      
    X_axis = np.arange(len(X))
    for i, model_id in enumerate(model_ids): 
        plt.bar(X_axis + inc[i], acc[shift][model_id], width=width, label=model_ids[model_id])

    plt.xticks(X_axis, X)
    plt.xlabel("Shift Intensity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy with {shift_name} shift")
    plt.legend()

    plt.savefig(prefix + f"{dataset}_graphs/compare_{shift}_acc.png")
    # plt.show()

    plt.figure()
    for i, model_id in enumerate(model_ids): 
        plt.bar(X_axis + inc[i], ece[shift][model_id], width = width, label = model_ids[model_id])
    plt.xticks(X_axis, X)

    plt.xlabel("Shift Intensity")
    plt.ylabel("ECE")
    plt.title(f"ECE with {shift_name}")
    plt.legend()

    plt.savefig(prefix + f"{dataset}_graphs/compare_{shift}_ece.png")
    plt.close()

def generate_average_graph(model_ids, acc, ece, distribution_shifts, dataset, baseline_id=None): 
    ### PLOT AVERAGE TREND 
    avg_acc = {}
    avg_ece = {}

    X = range(sevs)

    X_axis = np.arange(len(X))

    ece_max = {} 
    for shift in distribution_shifts: 
        ece_max[shift] = max([ece[shift][model_id][sev] for model_id in model_ids for sev in range(sevs)])
    for model_id in model_ids:
        avg_ece[model_id] = []
        for sev in range(sevs): 
            eces = []
            for shift in distribution_shifts: 
                eces.append(ece[shift][model_id][sev] / ece_max[shift])
            avg_ece[model_id].append(np.mean(eces))

    plt.figure()
    for i, model_id in enumerate(model_ids): 
        plt.bar(X_axis + inc[i], avg_ece[model_id], width = width, label = model_id)
    plt.xticks(X_axis, X)

    plt.xlabel("Shift Intensity")
    plt.ylabel("Normalized ECE")
    plt.title(f"Average normalized ECE over shifts")
    plt.legend()

    plt.savefig(prefix + f"{dataset}_graphs/compare_norm_avg_ece.png")

def collate_results(distribution_shifts, metric, metric_name, model_ids, dataset): 

    # if dataset == 'mnist':
    y_max = .7
    ece_y = [.0, .1, .2, .3, .4, .5, .6]
    ax_font = 30

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(50, 25))

    i, j = 0, 0

    for shift in distribution_shifts:

        ax = axes[i,j]
        # print(i, j)
        shift_name = distribution_shift_names[shift]

        X = range(sevs)  
        X_axis = np.arange(len(X))


        axes[i,j].set_xticks(X_axis, X)
        axes[i,j].set_xticklabels(labels=range(-2, len(X), 2), fontsize=ax_font)


        if metric_name == 'ECE':
            axes[i,j].set_ylim(0, y_max)
            axes[i,j].set_yticks(ece_y)
            axes[i,j].set_yticklabels(labels=ece_y, fontsize=ax_font)
        elif metric_name == 'Accuracy': 
            axes[i,j].set_ylim(0, 1.05)
            axes[i,j].set_yticks(np.arange(0, 1.2, .2))
            axes[i,j].set_yticklabels(labels=[0.0, .2, .4, .6, .8, 1.0], fontsize=ax_font)
        else: 
            print(f"oops: metric_name: {metric_name}")
            exit() 

        for idx, model_id in enumerate(model_ids.keys()): 
            axes[i,j].bar(X_axis + inc[idx], metric[shift][model_id], width=width, label=model_ids[model_id])

        if i == 1: 
            axes[i,j].set_xlabel("Shift Intensity", fontsize=35)
        if j == 0: 
            axes[i,j].set_ylabel(f"{metric_name}", fontsize=35)
        axes[i,j].set_title(f"{shift_name}", fontsize=45)

        i += 1 
        if i % 2 == 0: 
            i = 0
            j += 1 

    avg_metric = {}
    for model_id in model_ids.keys(): 

        print(avg_metric)
        avg_metric[model_id] = []

        for sev in range(0, sevs): 
            total_metric = 0
            for shift in distribution_shifts: 
                total_metric += metric[shift][model_id][sev]
            avg_metric[model_id].append((total_metric / len(distribution_shifts)))

        total_avg_metric = np.mean(avg_metric[model_id])
        print(f'model: {model_id} avg {metric_name}: {total_avg_metric}')



    X = range(sevs)
      
    X_axis = np.arange(len(X))
    for i, model_id in enumerate(model_ids): 
        axes[1, 4].bar(X_axis + inc[i], avg_metric[model_id], width =width, label = model_ids[model_id])

    axes[1, 4].set_xlabel("Shift Intensity", fontsize=35)
    # axes[1, 4].set_ylabel(f"{metric_name}", fontsize=35)
    axes[1, 4].set_title(f"Average {metric_name} over shifts", fontsize=45)
    # axes[1, 4].legend()

    X = range(sevs)  
    X_axis = np.arange(len(X))
    axes[1, 4].set_xticks(X_axis, X)
    axes[1, 4].set_xticklabels(labels=range(-2, len(X), 2), fontsize=ax_font)

    if metric_name == 'ECE':
        axes[1, 4].set_ylim(0, y_max)
        axes[1, 4].set_yticks(ece_y)
        axes[1, 4].set_yticklabels(labels=ece_y, fontsize=ax_font)
    elif metric_name == 'Accuracy': 
        axes[1, 4].set_ylim(0, 1.05)
        axes[1, 4].set_yticks(np.arange(0, 1.2, .2))
        axes[1, 4].set_yticklabels(labels=[0.0, .2, .4, .6, .8, 1.0], fontsize=ax_font)
    else: 
        print(f"oops: metric_name: {metric_name}")
        exit() 

    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(f"{prefix}{dataset}_graphs/{dataset}_collate_{metric_name.lower()}.png")
    print(f'figure saved to {prefix}{dataset}_graphs/{dataset}_collate_{metric_name.lower()}.png')


def flatten(input_imgs): 
    # print(input_imgs.shape)
    return tf.reshape(input_imgs, (-1, 3072))


def mnist_main(): 
   
    dataset = 'mnist'
    distribution_shifts = ['gaussian_noise',
    "shot_noise",
    "impulse_noise",
    "motion_blur",
    # "brightness",
    "contrast",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur"]

    # full_distribution_shifts = ['gaussian_noise']
    gen_pickle = False 
    # model_ids = ['Baseline', 'ECE Loss', 'OE ECE Loss']
    file_model_ids = ['Baseline', 'ECE Loss', 'OE ECE Loss']
    model_ids = {'Baseline': 'Baseline', 'ECE Loss': 'ECE Loss', 'OE ECE Loss': 'OE ECE Loss'}

    if gen_pickle: 

        m1 = load_mnist_model(lr=.0001, w=0, train_alg='ece', model_arch='seq') 
        m2 = load_mnist_model(lr=.0001, w=.1, train_alg='ece', model_arch='seq') 
        m3 = load_mnist_model(lr=.0001, w=.1, train_alg='ece_shift', model_arch='seq') 
        ident = lambda id_input:id_input
        transforms = {file_model_ids[0]: flatten, file_model_ids[1]:flatten, file_model_ids[2]:flatten}

        models = [m1, m2, m3]
        # models = [m1]

        dump_eval_to_pickle(models, file_model_ids, transforms, distribution_shifts, dataset, verbose=2)

    acc, ece = load_eval_from_pickle(file_model_ids, dataset)

    # collate_results(distribution_shifts, acc, "Accuracy", model_ids, dataset)
    # collate_results(distribution_shifts, ece, "ECE", model_ids, dataset)

    generate_average_graph(file_model_ids, acc, ece, distribution_shifts, dataset, baseline_id=None)

def main(): 

    distribution_shifts = ['gaussian_noise',
    "shot_noise",
    "impulse_noise",
    "motion_blur",
    # "brightness",
    "contrast",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur"]

    dataset='cifar'

    # full_distribution_shifts = ['gaussian_noise']
    gen_pickle = False 

    file_model_ids = ['Baseline', 'ECE Loss', 'OE ECE Loss', 'BaselineTransfer']
    model_ids = {'Baseline': 'Baseline', 'ECE Loss': 'ECE Loss', 'OE ECE Loss': 'OE ECE Loss'}

    # model_ids_to_add = ['BaselineTransfer']

    if gen_pickle: 

        m1, trn1 = load_cifar_model(lr=.0001, w=0, train_alg='ece', model_arch='EfficientNetB0Transfer') 
        m2, trn2 = load_cifar_model(lr=.0001, w=.1, train_alg='ece', model_arch='EfficientNetB0Transfer') 
        m3, trn3 = load_cifar_model(lr=.0001, w=.1, train_alg='ece_shift', model_arch='EfficientNetB0Transfer') 
        transforms = {model_ids[0]: trn1, model_ids[1]:trn2, model_ids[2]:trn3}
        transforms = {model_ids[0]: trn1}

        models = [m1, m2, m3]
        # models = [m1]

        dump_eval_to_pickle(models, file_model_ids, transforms, distribution_shifts, dataset, verbose=2)


    # merge_pickles(model_ids_to_add, model_ids, dataset, distribution_shifts)
    acc, ece = load_eval_from_pickle(file_model_ids, dataset)

    # collate_results(distribution_shifts, acc, "Accuracy", model_ids, dataset)
    collate_results(distribution_shifts, ece, "ECE", model_ids, dataset)

    # generate_average_graph(model_ids, acc, ece, distribution_shifts, dataset, baseline_id=None)

    # generate_average_graph(model_ids, acc, ece, distribution_shifts, dataset, baseline_id=model_ids[0])

    # for shift in distribution_shifts: 
    #     generate_shift_graph(model_ids, acc, ece, shift, dataset)


if __name__ == '__main__': 
    main() 
    # mnist_main()
