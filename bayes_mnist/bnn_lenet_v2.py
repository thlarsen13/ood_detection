# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify MNIST digits.

The architecture is LeNet-5 [1].

#### References

[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
         Gradient-based learning applied to document recognition.
         _Proceedings of the IEEE_, 1998.
         http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()

# TODO(b/78137893): Integration tests currently fail with seaborn imports.
warnings.simplefilter(action='ignore')

# try:
#     import seaborn as sns  # pylint: disable=g-import-not-at-top
#     HAS_SEABORN = True
# except ImportError:
#     HAS_SEABORN = False
HAS_SEABORN = False

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60000
NUM_HELDOUT_EXAMPLES = 10000
NUM_CLASSES = 10

flags.DEFINE_float('learning_rate',
                                     default=0.001,
                                     help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                                         default=10,
                                         help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                                         default=128,
                                         help='Batch size.')
flags.DEFINE_string('data_dir',
                                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                                                                 'bayesian_neural_network/data'),
                                        help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
        'model_dir',
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                                 'bayesian_neural_network/'),
        help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                                         default=400,
                                         help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                                         default=50,
                                         help='Network draws to compute predictive probabilities.')
flags.DEFINE_bool('fake_data',
                                    default=False,
                                    help='If true, uses fake data. Defaults to real data.')

FLAGS = flags.FLAGS


def create_model():
    """Creates a Keras model using the LeNet-5 architecture.

    Returns:
            model: Compiled Keras model.
    """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                        tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    model = tf.keras.models.Sequential([
            tfp.layers.Convolution2DFlipout(
                    6, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                    16, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                    pool_size=[2, 2], strides=[2, 2],
                    padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                    120, kernel_size=5, padding='SAME',
                    kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                    84, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                    NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
                    activation=tf.nn.softmax)
    ])

    # Model compilation.
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    # We use the categorical_crossentropy loss since the MNIST dataset contains
    # ten labels. The Keras API will then automatically add the
    # Kullback-Leibler divergence (contained on the individual layers of
    # the model), to the cross entropy loss, effectively
    # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
    model.compile(optimizer, loss='categorical_crossentropy',
                                metrics=['accuracy'], experimental_run_tf_function=False)
    return model


def train_model(path, train_seq, heldout_seq): 

    model = create_model()
    # TODO(b/149259388): understand why Keras does not automatically build the
    # model correctly.
    model.build(input_shape=[None, 28, 28, 1])

    print(' ... Training convolutional neural network')
    for epoch in range(FLAGS.num_epochs):
        epoch_accuracy, epoch_loss = [], []
        for step, (batch_x, batch_y) in enumerate(train_seq):
            batch_loss, batch_accuracy = model.train_on_batch(
                    batch_x, batch_y)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)

            if step % 100 == 0:
                print('Epoch: {}, Batch index: {}, '
                            'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                                    epoch, step,
                                    tf.reduce_mean(epoch_loss),
                                    tf.reduce_mean(epoch_accuracy)))

    model.save(path)
    return model

def main():


    train_set, heldout_set = tf.keras.datasets.mnist.load_data()

    train = True
    path = '/home/thlarsen/ood_detection/bayes_mnist/lenet_save/'
    model = None 
    if train: 
        model = train_model(path, train_set, heldout_set)
    else: 
        model = load_model(path)

    # model.summary()
    # exit()
    print('---Predicting---')
    for _ in range(FLAGS.num_monte_carlo):
        print(model.predict(heldout_seq, verbose=1))

if __name__ == '__main__':
    main() 
