import tensorflow as tf
import os
from tensorflow.python.ops import math_ops
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import TensorBoard

from datetime import datetime

# %%

class AbstractModel(tf.keras.Model):
    log_dir = './models/learned_controllers/exp001/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self, config, n_components):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.optimizer = tf.optimizers.Adam()
        self.n_components = n_components # number of Mixtures
        self.callback = TensorBoard(log_dir=self.log_dir, write_graph=True)

    def architecture_def(self, X):
        raise NotImplementedError()

class FFMDN(AbstractModel):
    def __init__(self, n_components):
        super(FFMDN, self).__init__(n_components)
        self.n_hidden_neurons = 20
        self.n_hidden_layers = 2
        self.architecture_def()

    def architecture_def(self):
        """pi, mu, sigma = NN(x; theta)"""
        # for n in range(self.n_hidden_layers):
        self.h1 = Dense(self.n_hidden_neurons, activation='relu', name="h1")
        self.h2 = Dense(self.n_hidden_neurons, activation='relu', name="h2")
        self.mus = Dense(self.n_components, name="mus")
        self.sigmas = Dense(self.n_components, activation=K.exp, name="sigmas")
        self.alphas = Dense(self.n_components, activation=K.softmax, name="pi")
        self.pvector = Concatenate(name="output") # parameter vector

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)

        return self.pvector([alpha_v, mu_v, sigma_v])

class GRUMDN(AbstractModel):
    pass
