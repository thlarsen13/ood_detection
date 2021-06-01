import tensorflow_probability as tfp
import tensorflow as tf 
import numpy as np 

from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib


dist = normal_lib.Normal(loc=[1, 2], scale=[.5, .5])


input = tf.keras.layers.Input(shape=(1,))
layer = tfp.layers.DenseReparameterization(2)

x = layer(input)

model = tf.keras.models.Model(inputs=input, outputs=x)

inp = np.array([[1]])
print(model.predict(inp))

print("weights =", layer.get_weights())

print("layer.kernel_posterior.mean() =", layer.kernel_posterior.mean())
print("layer.kernel_posterior.stddev() =", layer.kernel_posterior.stddev())

print("layer.kernel_prior.mean() =", layer.kernel_prior.mean())
print("layer.kernel_prior.stddev() =", layer.kernel_prior.stddev())

print("layer.bias_posterior.mean() =", layer.bias_posterior.mean())
print("layer.bias_posterior.stddev() =", layer.bias_posterior.stddev())