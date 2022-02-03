import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from cal_error import ExpectedCalibrationError
import time 
from datetime import datetime
from tensorflow.keras.applications import *
from tqdm import tqdm

verbose = True



# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
input_shape = x_train[0].shape

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

def train_attempt(lr=1e-3, w=1, epochs=20, graph_path=None, model_save_path=None): 

    model = EfficientNetB2(weights=None, classes=10, input_shape=input_shape, classifier_activation=None)


    optimizer = keras.optimizers.SGD(learning_rate=lr)
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

def main(): 

    weights = [10 **i for i in range(-3, 1)]
    learning_rates = [10**i for i in range(-5, -2)]

    # weights = [1]
    # learning_rates = [10**-3]

    overall_results = [['l/w']+ weights]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 20

    with tqdm(total=len(learning_rates) * len(weights)) as pbar:
        for lr in learning_rates: 
            overall_results.append([lr])
            for w in weights:
                model_save_path = f'{prefix}saved_weights/cifar_calibrate/cal(lr={lr})(w={w})'
                graph_path = f'{prefix}training_plots/cifar_calibrate/cal(lr={lr})(w={w}).png'
                acc, ece = train_attempt(lr=lr, w=w, epochs=epochs, graph_path=graph_path, model_save_path=model_save_path)
                overall_results[-1].append((acc, ece))
                pbar.update(1)

    #array nonsense to make it print the array in a readable format
    s = [[str(e) for e in row] for row in overall_results]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
if __name__ == "__main__": 
    main()







