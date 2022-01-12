import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import tensorflow_datasets as tfds
import tensorflow.keras.backend as kb
import keras
'''
Simplest possible instatiation of eager execution bug with uninitialized tensors 
running throuhg the model. 
'''

BATCH_SIZE = 128
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
cce = tf.keras.losses.CategoricalCrossentropy() 


### broken implementation of sparse categorical cross entropy. 
# To fix, add epsilon to each entry
def custom_cce_loss(y_true, y_hat): 
    y_hat_softmax = tf.nn.softmax(y_hat)
    y_true_one_hot = tf.squeeze(tf.one_hot(y_true, depth=10))
    # print(y_hat_softmax.shape, y_true_one_hot.shape)
    custom_loss = -tf.reduce_sum( 
        tf.math.multiply(y_true_one_hot,tf.math.log(y_hat_softmax)), axis=1) 
    return custom_loss

"""
computing cross entropy loss yourself is numerically unstable, it results in a lot of nans while training 
instead find out how to use tensorflow's crossentropy. 
1. split up the important indices (keep the Tensors on GPU/don't copy too much memory)
2. apply categorical cross entropy from logits to the working one
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
3. compute confidence penalty from the rest of the NN output. 

"""
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def conf_cce_loss(y_true, y_hat): 
    gamma = 1e-3 #make this into a class probably, this becomes a member var
    C = 10 #number of output classes
    y_hat = y_hat[:, :C]
    conf_hat = y_hat[:, C:]
    conf = gamma * tf.math.log(tf.reduce_sum(conf_hat, axis=1))
    cce_val = cce(y_true, y_hat)
    return conf + cce_val

def divergence_prevention(target, ouput):
    if not from_logits: #assuming already softmaxed probs. 
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                              reduction_indices=len(output.get_shape()) - 1)

# def custom_loss(y_actual,y_pred): 
#     # tf.print(y_actual)
#     # tf.print(y_pred)
#     # custom_loss=kb.square(y_actual-y_pred)
#     # custom_loss = tf.nn.softmax_cross_entropy_with_logits( labels=y_actual, logits=y_pred) works 

#     # custom_loss = cce(y_actual, y_pred)
#     # exit()
#     return custom_loss


class LossV2(keras.losses.Loss):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="confidence_loss"):
        # print("here")
        super().__init__(reduction=reduction, name=name)
        # self.noisy_signal = noisy_signal

 
    def call(self, y_true, y_pred):
        # noise_true = self.noisy_signal - y_true
        # noise_pred = self.noisy_signal - y_pred
        # alpha = (tf.reduce_mean(tf.square(y_true)) /
        #          tf.reduce_mean(tf.square(y_true) + tf.square(self.noisy_signal - y_pred)))
        # return alpha * self.sdr_loss(y_true, y_pred) + (1 - alpha) * self.sdr_loss(noise_true, noise_pred)
        return kb.square(y_true - y_pred)

confidence_loss = tf.keras.losses.CategoricalCrossentropy()


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(28, 28, 1)), 
    MaxPooling2D((2, 2)), 
    Conv2D(64, 
                 kernel_size=(3, 3), 
                 activation='relu'), 
    MaxPooling2D((2, 2)), 
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(20)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(10**-3),

    # loss=tf.nn.softmax_cross_entropy_with_logits
        # loss=tf.nn.sparse_softmax_cross_entropy_with_logits
#below works: 
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 

    loss = conf_cce_loss
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=60,
    validation_data=ds_test,
)