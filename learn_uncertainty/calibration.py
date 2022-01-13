#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.stats import quantiles as quantiles_lib


# In[13]:



inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[14]:




# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

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


# In[15]:


def _compute_calibration_bin_statistics(
    num_bins, logits=None, labels_true=None):
    """ Compute binning statistics required for calibration measures.
  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
  Returns:
    bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
      correct (row 1) predictions in each of the `num_bins` probability bins.
    pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
      probabilities in each probability bin.
    """
    
    # We take the label with the maximum probability
    # decision.  This corresponds to the optimal expected minimum loss decision
    # under 0/1 loss.
#         pred_y = tf.argmax(logits, axis=1, output_type=labels_true.dtype)
    pred_y = tf.argmax(logits, axis=1)


    correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

    # Collect predicted probabilities of decisions
    pred = tf.nn.softmax(logits, axis=1)
    prob_y = tf.gather(
      pred, pred_y[:, tf.newaxis], batch_dims=1)  # p(pred_y | x)
    prob_y = tf.reshape(prob_y, (ps.size(prob_y),))

    # Compute b/z histogram statistics:
    # bz[0,bin] contains counts of incorrect predictions in the probability bin.
    # bz[1,bin] contains counts of correct predictions in the probability bin.
    bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=num_bins)
    event_bin_counts = tf.math.bincount(
      correct * num_bins + bins,
      minlength=2 * num_bins,
      maxlength=2 * num_bins)
    event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))

    # Compute mean predicted probability value in each of the `num_bins` bins
    pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
    tiny = np.finfo(dtype_util.as_numpy_dtype(logits.dtype)).tiny
    pmean_observed = pmean_observed / (
      tf.cast(tf.reduce_sum(event_bin_counts, axis=0), logits.dtype) + tiny)

    return event_bin_counts, pmean_observed


# In[16]:


def computeESE(num_bins, logits=None, labels_true=None): 
    """
    Args:
        num_bins: int, number of probability bins to compute oberved likelyhoods of correctness, e.g. 10.
        logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
        labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
          ground truth class labels in the range [0,nlabels].
    Returns:
        ece: Tensor, scalar, tf.float32.
    """
    with tf.name_scope('expected_calibration_error'):
        logits = tf.convert_to_tensor(logits)
        labels_true = tf.convert_to_tensor(labels_true)
        labels_true = tf.cast(labels_true, dtype=tf.int64)

        # Compute empirical counts over the events defined by the sets
        # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
        # of predicted probabilities in each probability bin.
        event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
            num_bins, logits=logits, labels_true=labels_true)

        # Compute the marginal probability of observing a probability bin.
        event_bin_counts = tf.cast(event_bin_counts, tf.float32)
        bin_n = tf.reduce_sum(event_bin_counts, axis=0)
        pbins = bin_n / tf.reduce_sum(bin_n)  # Compute the marginal bin probability

        # Compute the marginal probability of making a correct decision given an
        # observed probability bin.
        tiny = np.finfo(np.float32).tiny
        pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

        # Compute the ESE statistic which is supposed to be analagous to brier score
        ese = tf.reduce_sum(pbins * tf.square(pcorrect - pmean_observed))
    return ese


# In[21]:


# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-2)
# cal_optimizer = keras.optimizers.SGD(learning_rate=1e-8)

# Instantiate a loss function.
cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

num_bins=10
gamma = 10**0
def loss_fn(y_batch_train, logits, verbose=False): 
    cce = cce_loss_fn(y_batch_train, logits) 
    ese = tf.multiply(gamma, computeESE(num_bins, logits=logits, labels_true=y_batch_train))
    if verbose:
        print(f"Training cce, ese, loss (for one batch): {cce:.4f}, {ese:.4f}, {cce+ese:.4f}")

#     return tf.add(cce, ese)
    return ese

epochs = 5


# In[22]:


for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True)  # Logits for this minibatch
            loss_value = loss_fn(y_batch_train, logits, verbose =(step % 200 == 0)) #verbose every 200 batches

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    


# In[ ]:




