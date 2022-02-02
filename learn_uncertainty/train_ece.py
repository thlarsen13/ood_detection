

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.losses import Loss
from cal_error import ExpectedCalibrationError
import time 

verbose = False

input_dim = 32*32

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

print(x_test.shape)
x_train = np.reshape(x_train, (-1, input_dim))
x_test = np.reshape(x_test, (-1, input_dim))
print(x_test.shape)

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

    inputs = keras.Input(shape=(input_dim,), name="digits")
    x1 = layers.Dense(64, activation="relu")(inputs)
    x2 = layers.Dense(64, activation="relu")(x1)
    outputs = layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.SGD(learning_rate=lr)
    # Instantiate a loss function.
    cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ese_loss_fn = ExpectedCalibrationError(weight=w)

    ECE_dict = { i : [] for i in range(epochs)}
    ACC = []

    def loss_fn(y_batch_train, logits, verbose=False): 
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
    return ACC[-1], ECE[-1]

def main(): 

    weights = [10 **i for i in range(-3, 3)]
    learning_rates = [10**i for i in range(-5, -1)]
    overall_results = [['lrs']+weights]

    # weights = [1]
    # learning_rates = [10**-3]
    prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
    epochs = 20
    for lr in learning_rates: 
        overall_results.append([f'lr = {lr}'])
        for w in weights:
            model_save_path = f'{prefix}saved_weights/mnist_calibrate/cal(lr={lr})(w={w})'
            graph_path = f'{prefix}training_plots/mnist_calibrate/cal(lr={lr})(w={w}).png'
            acc, ece = train_attempt(lr=lr, w=w, epochs=epochs, graph_path=graph_path, model_save_path=model_save_path)
            overall_results[-1].append((acc, ece))

    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in overall_results]))
if __name__ == "__main__": 
    main()

