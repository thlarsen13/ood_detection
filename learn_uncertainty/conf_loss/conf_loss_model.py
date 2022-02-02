import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as kb

import numpy as np

C = 10

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  print(label.shape)
  return tf.cast(image, tf.float32) / 255., label

"""
explicitly learn a representation of distributoins in the output layer in order
to model the uncertainty. mathematically, distribution should be invariant under some 
transformations. e.g. background shift. 
under noisy transformations, mean should stay the same, but std should be allowed to change 


need to define a loss function that wants to have low error/high confidence, 
but punishes high confidence errors significantly
"""
def old_confidence_loss(y_true, y_pred): 
  gamma = 1

  y_hat = y_pred[:C]
  conf_hat = y_pred[C:]

  # true_label = tf.argmax(y_true)
  # true_label = y_true[:, 0]
  # print(true_label)
  # print(y_pred.shape, y_hat.shape, conf_hat.shape)
  # exit()
  cross_entropy = tf.reduce_sum()
  # cross_entropy = -conf_hat[:, y_true[:, 0]] * tf.math.log(y_hat[:, y_true[:, 0]])
  confidence_penalty = - gamma * tf.math.log(tf.reduce_sum(conf_hat, axis=1))

  # return -np.dot(conf_hat, np.log(y_hat[y_true])) + gamma*np.sum(1 - conf_hat)
  return cross_entropy + confidence_penalty
# def confidence_loss(y_true, y_pred): 
  # gamma = 1
  # y_pred_ = tf.reshape(y_pred, (-1, 2))
  # log_preds = tf.math.log(y_pred_)
  # print(y_true.shape)
  # # print(y_true)
  # exit()
  # T = tf.tensordot(tf.one_hot(y_true, C), log_preds, axes=1)
  
  # cross_entropy = tf.reduce_mean(-tf.reduce_sum(T, axis=1))
  # confidence_penalty = - gamma * tf.math.log(tf.reduce_sum(y_pred_, axis=1)[0])

  # return tf.add(cross_entropy, confidence_penalty)
  # return kb.square(y_true-y_pred)
confidence_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# class CustomAccuracy(tf.keras.losses.Loss):
#   def __init__(self):
#     super().__init__()
#   def call(self, y_true, y_pred):
#     mse = tf.reduce_mean(tf.square(y_pred-y_true))
#     rmse = tf.math.sqrt(mse)
#     return rmse / tf.reduce_mean(tf.square(y_true)) - 1
 
    
def confidence_acc(y_true, y_pred): 

  return tf.reduce_mean(tf.math.multiply(tf.one_hot(y_true, int(2*C)), y_pred), axis=1)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=confidence_loss,
    metrics=[confidence_loss],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

