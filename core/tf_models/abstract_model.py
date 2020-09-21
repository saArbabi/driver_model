from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)

import os
from tensorflow.python.ops import math_ops
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime



# %%

class AbstractModel(tf.keras.Model):
    """
    See https://www.tensorflow.org/guide/intro_to_graphs if you want to wrap the model
    in a graph using tf.function decorator.
    """

    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.model_type = config['model_type']

        self.exp_dir = './models/experiments/'+config['exp_id']
        self.learning_rate = self.config['learning_rate']
        self.neurons_n = self.config['neurons_n']
        self.layers_n = self.config['layers_n']
        self.epochs_n = self.config['epochs_n']
        self.batch_n = self.config['batch_n']
        self.components_n = self.config['components_n'] # number of Mixtures
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.callback = self.callback_def()

    def architecture_def(self, X):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'+current_time
        return TensorBoard(log_dir=log_dir, write_graph=True)

class FFMDN(AbstractModel):
    def __init__(self, config):
        super(FFMDN, self).__init__(config)
        self.architecture_def(config)

    def architecture_def(self, config):
        """pi, mu, sigma = NN(x; theta)"""
        # for n in range(self.layers_n):
        self.hidden_layers =  [Dense(self.neurons_n, activation='relu') for _
                                                in range(self.config['layers_n'])]
        self.alphas = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long = Dense(self.components_n, name="mus_long")
        self.sigmas_long = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        if self.model_type == 'merge_policy':
            self.mus_lat = Dense(self.components_n, name="mus_lat")
            self.sigmas_lat = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
            self.rhos = Dense(self.components_n, activation=K.tanh, name="rhos")

        self.pvector = Concatenate(name="output") # parameter vector

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        x = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            x = layer(x)

        alphas = self.alphas(x)
        mus_long = self.mus_long(x)
        sigmas_long = self.sigmas_long(x)
        if self.model_type == 'merge_policy':
            mus_lat = self.mus_lat(x)
            sigmas_lat = self.sigmas_lat(x)
            rhos = self.rhos(x)
            return self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
        return self.pvector([alphas, mus_long, sigmas_long])


class GRUMDN(AbstractModel):
    pass
