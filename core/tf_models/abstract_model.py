import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import os
from tensorflow.python.ops import math_ops
import random
import matplotlib.pyplot as plt
from keras import backend as k
from keras.layers import Dense
from sklearn.model_selection import train_test_split

seed_value = 2020
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.set_random_seed(seed_value)



# %%
class AbstractModel():

    def __init__(self, K):
        self.K = K # number of Mixtures

    def architecture_def(self, X):
        raise NotImplementedError()

    def log_prob(self, y):
        X, y = xs
        self.mapping(X)
        result = tf.exp(norm.logpdf(y, self.mus, self.sigmas))
        result = tf.mul(result, self.pi)
        result = tf.reduce_sum(result, 1)
        result = tf.log(result)
        return tf.reduce_sum(result)


class FFMDN(AbstractModel):
    def __init__(self, K):
        super(FFMDN, self).__init__(K)

    def architecture_def(self, X):
        """pi, mu, sigma = NN(x; theta)"""
        hidden1 = Dense(15, activation='relu')(X)  # fully-connected layer with 15 hidden units
        hidden2 = Dense(15, activation='relu')(hidden1)
        self.mus = Dense(self.K)(hidden2) # the means
        self.sigmas = Dense(self.K, activation=K.exp)(hidden2) # the variance
        self.pi = Dense(self.K, activation=K.softmax)(hidden2) # the mixture components


class GRUMDN(AbstractModel):
    pass

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

X_train, X_test, y_train, y_test = build_toy_dataset()

plt.scatter(X_train, y_train)

model = FFMDN(20)
