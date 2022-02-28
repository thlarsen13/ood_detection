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

import os 

def train_attempt(input_shape, train_dataset, val_dataset, lr=1e-3, w=1, epochs=20, graph_path=None, model_save_path=None, verbose=False): 

    model = EfficientNetB0(weights=None, classes=10, input_shape=input_shape, classifier_activation=None)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Instantiate a loss function.
    cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ese_loss_fn = ExpectedCalibrationError(weight=w)

    if os.path.exists(model_save_path):
        try: 
                    # Get saved weights
            opt_weights = np.load(model_save_path + 'opt_weights.npy', allow_pickle=True)

            grad_vars = model.trainable_weights
            # This need not be model.trainable_weights; it must be a correctly-ordered list of 
            # grad_vars corresponding to how you usually call the optimizer.

            zero_grads = [tf.zeros_like(w) for w in grad_vars]

            # Apply gradients which don't do nothing with Adam
            optimizer.apply_gradients(zip(zero_grads, grad_vars))

            # Set the weights of the optimizer
            optimizer.set_weights(opt_weights)

            # NOW set the trainable weights of the model
            # model_weights = np.load('/path/to/saved/model/weights.npy', allow_pickle=True)
            # model.set_weights(model_weights)
            model.load_weights(model_save_path)

            # model._make_train_function()
            # with open('optimizer.pkl', 'rb') as f:
            #     weight_values = pickle.load(f)
            # model.optimizer.set_weights(weight_values)
            print("Succefully loaded model with optimizer info")
        except: 
            print("Error finding pretrained model and optimizer, training from scratch instead")


    ECE_dict = { i : [] for i in range(epochs)}
    ACC = []

    def loss_fn(y_batch_train, logits, verbose=False): 
        # print(y_batch_train.shape)
        # print(logits.shape)
        # exit()
        cce = cce_loss_fn(y_batch_train, logits) 
        ese = ese_loss_fn(y_batch_train, logits)
        if verbose:
            # print(f'w={w}')
            print(f'Training cce, ese, loss (for one batch): {cce:.4f}, {ese:.4f}, {cce+ese:.4f}')

        ECE_dict[epoch].append(ese.numpy())
        return tf.add(cce, ese)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    """
    Here's our training & evaluation loop:
    """

    for epoch in tqdm(range(epochs)):
        if verbose: 
            print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # print(x_batch_train.shape)
            # print(y_batch_train.shape)
            # exit()
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits, verbose=False)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        if verbose: 
            print("Training acc over epoch: %.4f" % (float(train_acc),))
        ACC.append(train_acc.numpy())
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits) 
            #TODO: write custom update state for ece 

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        if verbose: 
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








