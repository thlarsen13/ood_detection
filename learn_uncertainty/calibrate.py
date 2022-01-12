"""
Title: Writing a training loop from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/01
Last modified: 2020/04/15
Description: Complete guide to writing low-level training & evaluation loops.
"""
"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
## Introduction

Keras provides default training and evaluation loops, `fit()` and `evaluate()`.
Their usage is covered in the guide
[Training & evaluation with the built-in methods](/guides/training_with_built_in_methods/).

If you want to customize the learning algorithm of your model while still leveraging
the convenience of `fit()`
(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and
implement your own `train_step()` method, which
is called repeatedly during `fit()`. This is covered in the guide
[Customizing what happens in `fit()`](/guides/customizing_what_happens_in_fit/).

Now, if you want very low-level control over training & evaluation, you should write
your own training & evaluation loops from scratch. This is what this guide is about.
"""

"""
## Using the `GradientTape`: a first end-to-end example

Calling a model inside a `GradientTape` scope enables you to retrieve the gradients of
the trainable weights of the layer with respect to a loss value. Using an optimizer
instance, you can use these gradients to update these variables (which you can
retrieve using `model.trainable_weights`).

Let's consider a simple MNIST model:

"""

inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

"""
Let's train it using mini-batch gradient with a custom training loop.

First, we're going to need an optimizer, a loss function, and a dataset:
"""

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

"""
Here's our training loop:

- We open a `for` loop that iterates over epochs
- For each epoch, we open a `for` loop that iterates over the dataset, in batches
- For each batch, we open a `GradientTape()` scope
- Inside this scope, we call the model (forward pass) and compute the loss
- Outside the scope, we retrieve the gradients of the weights
of the model with regard to the loss
- Finally, we use the optimizer to update the weights of the model based on the
gradients
"""

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))

"""
## Low-level handling of metrics

Let's add metrics monitoring to this basic loop.

You can readily reuse the built-in metrics (or custom ones you wrote) in such training
loops written from scratch. Here's the flow:

- Instantiate the metric at the start of the loop
- Call `metric.update_state()` after each batch
- Call `metric.result()` when you need to display the current value of the metric
- Call `metric.reset_states()` when you need to clear the state of the metric
(typically at the end of an epoch)

Let's use this knowledge to compute `SparseCategoricalAccuracy` on validation data at
the end of each epoch:
"""

# Get model
# inputs = keras.Input(shape=(784,), name="digits")
# x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x = layers.Dense(64, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(10, name="predictions")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)

# # Instantiate an optimizer to train the model.
# optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# # Instantiate a loss function.
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# # Prepare the metrics.
# train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# """
# Here's our training & evaluation loop:
# """

# import time

# epochs = 2
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#     start_time = time.time()

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         with tf.GradientTape() as tape:
#             logits = model(x_batch_train, training=True)
#             loss_value = loss_fn(y_batch_train, logits)
#         grads = tape.gradient(loss_value, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # Update training metric.
#         train_acc_metric.update_state(y_batch_train, logits)

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %d samples" % ((step + 1) * batch_size))

#     # Display metrics at the end of each epoch.
#     train_acc = train_acc_metric.result()
#     print("Training acc over epoch: %.4f" % (float(train_acc),))

#     # Reset training metrics at the end of each epoch
#     train_acc_metric.reset_states()

#     # Run a validation loop at the end of each epoch.
#     for x_batch_val, y_batch_val in val_dataset:
#         val_logits = model(x_batch_val, training=False)
#         # Update val metrics
#         val_acc_metric.update_state(y_batch_val, val_logits)
#     val_acc = val_acc_metric.result()
#     val_acc_metric.reset_states()
#     print("Validation acc: %.4f" % (float(val_acc),))
#     print("Time taken: %.2fs" % (time.time() - start_time))

# """
# ## Speeding-up your training step with `tf.function`

# The default runtime in TensorFlow 2 is
# [eager execution](https://www.tensorflow.org/guide/eager).
# As such, our training loop above executes eagerly.

# This is great for debugging, but graph compilation has a definite performance
# advantage. Describing your computation as a static graph enables the framework
# to apply global performance optimizations. This is impossible when
# the framework is constrained to greedly execute one operation after another,
# with no knowledge of what comes next.

# You can compile into a static graph any function that takes tensors as input.
# Just add a `@tf.function` decorator on it, like this:
# """


# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss_value = loss_fn(y, logits)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     train_acc_metric.update_state(y, logits)
#     return loss_value


# """
# Let's do the same with the evaluation step:
# """


# @tf.function
# def test_step(x, y):
#     val_logits = model(x, training=False)
#     val_acc_metric.update_state(y, val_logits)


# """
# Now, let's re-run our training loop with this compiled training step:
# """

# import time

# epochs = 2
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#     start_time = time.time()

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         loss_value = train_step(x_batch_train, y_batch_train)

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %d samples" % ((step + 1) * batch_size))

#     # Display metrics at the end of each epoch.
#     train_acc = train_acc_metric.result()
#     print("Training acc over epoch: %.4f" % (float(train_acc),))

#     # Reset training metrics at the end of each epoch
#     train_acc_metric.reset_states()

#     # Run a validation loop at the end of each epoch.
#     for x_batch_val, y_batch_val in val_dataset:
#         test_step(x_batch_val, y_batch_val)

#     val_acc = val_acc_metric.result()
#     val_acc_metric.reset_states()
#     print("Validation acc: %.4f" % (float(val_acc),))
#     print("Time taken: %.2fs" % (time.time() - start_time))

# """
# Much faster, isn't it?
# """

# """
# ## Low-level handling of losses tracked by the model

# Layers & models recursively track any losses created during the forward pass
# by layers that call `self.add_loss(value)`. The resulting list of scalar loss
# values are available via the property `model.losses`
# at the end of the forward pass.

# If you want to be using these loss components, you should sum them
# and add them to the main loss in your training step.

# Consider this layer, that creates an activity regularization loss:

# """


# class ActivityRegularizationLayer(layers.Layer):
#     def call(self, inputs):
#         self.add_loss(1e-2 * tf.reduce_sum(inputs))
#         return inputs


# """
# Let's build a really simple model that uses it:
# """

# inputs = keras.Input(shape=(784,), name="digits")
# x = layers.Dense(64, activation="relu")(inputs)
# # Insert activity regularization as a layer
# x = ActivityRegularizationLayer()(x)
# x = layers.Dense(64, activation="relu")(x)
# outputs = layers.Dense(10, name="predictions")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)


# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss_value = loss_fn(y, logits)
#         # Add any extra losses created during the forward pass.
#         loss_value += sum(model.losses)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     train_acc_metric.update_state(y, logits)
#     return loss_value


