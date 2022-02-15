import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from train_stats import ExpectedCalibrationError
import time 
from datetime import datetime
from tensorflow.keras.applications import *
from tqdm import tqdm


def train_attempt(model_path, train_dataset, val_dataset, lr=1e-3, w=1, epochs=20, graph_path=None, model_save_path=None, verbose=False): 

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Instantiate a loss function.
    cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ese_loss_fn = ExpectedCalibrationError(weight=w)

    ECE_dict = { i : [] for i in range(epochs)}
    ACC = []

    def loss_fn(y_batch_train, logits, verbose=False): 
        # print(y_batch_train.shape)
        # print(logits.shape)
        # exit()
        cce = cce_loss_fn(y_batch_train, logits) 
        ese = ese_loss_fn(y_batch_train, logits)
        if verbose:
            print(f'Training cce, ese, loss (for one batch): {cce:.4f}, {ese:.4f}, {cce+ese:.4f}')

        ECE_dict[epoch].append(ese.numpy())
        return tf.add(cce, ese)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    """
    Here's our training & evaluation loop:
    """

    for epoch in range(epochs):
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
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if verbose and step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

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
    return round(ACC[-1], 4), round(ECE[-1], 4)








