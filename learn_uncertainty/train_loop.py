import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from calibration_stats import ExpectedCalibrationError
import time 
from datetime import datetime
from tensorflow.keras.applications import *
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D 
from tensorflow.keras.layers import BatchNormalization
import cv2

import os 


class TrainBuilder(): 

    def __init__(self, input_shape, lr=1e-3, w=1, epochs=20, graph_path=None, model_save_path=None, verbose=False, transform=None): 
        self.input_shape = input_shape
        self.lr = lr 
        self.w = w
        self.epochs = epochs 
        self.graph_path = graph_path
        self.model_save_path = model_save_path
        self.verbose = verbose
        self.transform = transform
        
        if transform == None: 
            transform = lambda identity: identity

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        self.cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.ese_loss_fn = ExpectedCalibrationError(weight=self.w)

        self.ECE_dict = { i : [] for i in range(epochs)}
        self.ACC = []
        self.model_arch = 'EfficientNetB0Transfer'

        self.n_classes = 10

        self.epoch = None # hacky to save epoch number for loss function to update state

    def loss_fn(self, y_batch_train, logits): 
        # print(y_batch_train.shape)
        # print(logits.shape)
        # exit()
        cce = self.cce_loss_fn(y_batch_train, logits) 
        ece = tf.constant(0, dtype=tf.float32)
        if self.w != 0: #avoid evaluating this if w != 0
            ese = self.ese_loss_fn(y_batch_train, logits)

        self.ECE_dict[self.epoch].append(ece.numpy())

        loss_val =  tf.add(cce, ece)
        if self.verbose:
            # print(f'w={w}')
            print(f'Training cce, ese, loss (for one batch): {cce:.4f}, {ece:.4f}, {loss_val:.4f}')

        return loss_val

    def load_saved_weights(self, model): 
        opt_weights = np.load(self.model_save_path + 'opt_weights.npy', allow_pickle=True)

        grad_vars = model.trainable_weights
        # This need not be model.trainable_weights; it must be a correctly-ordered list of 
        # grad_vars corresponding to how you usually call the optimizer.

        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        # Apply gradients which don't do nothing with Adam
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights of the optimizer
        self.optimizer.set_weights(opt_weights)

        # NOW set the trainable weights of the model
        # model_weights = np.load('/path/to/saved/model/weights.npy', allow_pickle=True)
        # model.set_weights(model_weights)
        model.load_weights(self.model_save_path)

        # model._make_train_function()
        # with open('optimizer.pkl', 'rb') as f:
        #     weight_values = pickle.load(f)
        # model.optimizer.set_weights(weight_values)
    def get_model(self): 

        model = None 

        if self.model_arch == 'EfficientNetB0Transfer': 
            efnb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape, classes=self.n_classes)

            model = Sequential()
            model.add(efnb0)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(self.n_classes))

            model.summary()
        elif self.model_arch == 'EfficientNetB0': 

            self.model = EfficientNetB0(weights=None, classes=10, input_shape=input_shape, classifier_activation=None)
            # exit() 
        else: 
            print("no model arch or model specified, exiting")
            exit() 

        if os.path.exists(self.model_save_path):
            try: 
                self.load_saved_weights()
                print("Succefully loaded model with optimizer info")

            except: 
                print("Error finding pretrained model and optimizer, training from scratch instead")
        else:
            print('starting to train new model')
        return model 

    def train_attempt(self, train_dataset, val_dataset, model=None): 
        if model == None: 
            model = self.get_model() 

        # Prepare the metrics.
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        """
        Here's our training & evaluation loop:
        """

        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            if self.verbose: 
                print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # print(x_batch_train.shape)
                # print(y_batch_train.shape)
                # exit()
                with tf.GradientTape() as tape:
                    logits = model(self.transform(x_batch_train), training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            if self.verbose: 
                print("Training acc over epoch: %.4f" % (float(train_acc),))
            self.ACC.append(train_acc.numpy())
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = model(self.transform(x_batch_val), training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits) 
                #TODO: write custom update state for ece 

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            if self.verbose: 
                print("Validation acc: %.4f" % (float(val_acc),))
                print("Time taken: %.2fs" % (time.time() - start_time))
        #TODO Add EVAL
        if graph_path is not None: 
            ECE = [np.mean(ECE_dict[i]) for i in range(epochs)]
            print(f"\n\n\n@@@ {graph_path}\n ECE ({len(ECE)}): {ECE} \nACC ({len(ACC)}): {ACC}")
        if model_save_path is not None: 
            # model.compile(optimizer="Adam", loss=tf.keras.losses.CategoricalCrossentropy)
            model.save(model_save_path)
            np.save(model_save_path + 'opt_weights.npy', optimizer.get_weights())        

        return round(ACC[-1], 4), round(ECE[-1], 4)








