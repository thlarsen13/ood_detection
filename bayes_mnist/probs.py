import tensorflow_probability as tfp
import tensorflow as tf 
import numpy as np 

from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib


dist = normal_lib.Normal(loc=[1, 2], scale=[.5, .5])

print(dist.sample([4]))