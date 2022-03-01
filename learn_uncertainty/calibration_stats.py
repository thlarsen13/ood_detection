import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.stats import quantiles as quantiles_lib
import numpy as np 
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import dtype_util

class ExpectedCalibrationError(Loss):
    def __init__(self, weight=1, num_bins=10):
        super().__init__()
        self.weight = weight
        self.num_bins = num_bins
    
    def _compute_calibration_bin_statistics(self, 
        logits=None, labels_true=None):
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
        pred_y = tf.argmax(logits, axis=1)


        correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

        # Collect predicted probabilities of decisions
        pred = tf.nn.softmax(logits, axis=1)
        prob_y = tf.gather(
          pred, pred_y[:, tf.newaxis], batch_dims=1)  # p(pred_y | x)
        prob_y = tf.reshape(prob_y, (ps.size(prob_y),))

        # Compute b/z? histogram statistics:
        # bz[0,bin] contains counts of incorrect predictions in the probability bin.
        # bz[1,bin] contains counts of correct predictions in the probability bin.
        bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=self.num_bins)
        event_bin_counts = tf.math.bincount(
          correct * self.num_bins + bins,
          minlength=2 * self.num_bins,
          maxlength=2 * self.num_bins)
        event_bin_counts = tf.reshape(event_bin_counts, (2, self.num_bins))

        # Compute mean predicted probability value in each of the `num_bins` bins
        pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, self.num_bins)
        tiny = np.finfo(dtype_util.as_numpy_dtype(logits.dtype)).tiny
        pmean_observed = pmean_observed / (
          tf.cast(tf.reduce_sum(event_bin_counts, axis=0), logits.dtype) + tiny)

        return event_bin_counts, pmean_observed


    def call(self, labels_true=None, logits=None): 
        """
        Args:
            num_bins: int, number of probability bins to compute oberved likelyhoods of correctness, e.g. 10.
            logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
            labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
              ground truth class labels in the range [0,nlabels].
        Returns:
            ece: Tensor, scalar, tf.float32.
        """
        logits = tf.convert_to_tensor(logits)
        labels_true = tf.convert_to_tensor(labels_true)
        labels_true = tf.cast(labels_true, dtype=tf.int64)

        # Compute empirical counts over the events defined by the sets
        # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
        # of predicted probabilities in each probability bin.
        event_bin_counts, pmean_observed = self._compute_calibration_bin_statistics(
            logits=logits, labels_true=labels_true)

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
        return tf.multiply(self.weight, ese)
        

