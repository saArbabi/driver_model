import tensorflow as tf
import os
from tensorflow.python.ops import math_ops
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import TensorBoard

from datetime import datetime

# %%

class AbstractModel(tf.keras.Model):

    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.exp_dir = self.get_expDir(config)
        self.optimizer = tf.optimizers.Adam()
        self.components_n = self.config['components_n'] # number of Mixtures
        self.callback = self.callback_def()

    def architecture_def(self, X):
        raise NotImplementedError()

    def get_expDir(self, config):
        exp_dir = './models/'
        if config['experiment_type']['model'] == 'controller':
            exp_dir += 'controller/'

        elif config['experiment_type']['model'] == 'driver_model':
            exp_dir += 'driver_model/'
        else:
            raise Exception("Unknown experiment type")

        return exp_dir + config['experiment_name']

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = self.exp_dir+'/logs/'+current_time
        print("logs saved at "+log_dir)
        return TensorBoard(log_dir=log_dir)

class FFMDN(AbstractModel):
    def __init__(self, config):
        super(FFMDN, self).__init__(config)
        self.hidden_size = 20
        self.n_hidden_layers = 2
        self.architecture_def()

    def architecture_def(self):
        """pi, mu, sigma = NN(x; theta)"""
        # for n in range(self.n_hidden_layers):
        self.h1 = Dense(self.hidden_size, activation='relu', name="h1")
        self.h2 = Dense(self.hidden_size, activation='relu', name="h2")
        self.mus = Dense(self.components_n, name="mus")
        self.sigmas = Dense(self.components_n, activation=K.exp, name="sigmas")
        self.alphas = Dense(self.components_n, activation=K.softmax, name="pi")
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
