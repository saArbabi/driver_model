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

    @tf.function
    def tracemodel(self, x):
        """Trace model execution - use for writing model graph
        :param: A sample input
        """
        return self(x)

    def saveGraph(self, x):
        writer = tf.summary.create_file_writer(self.exp_dir+'/graph')
        tf.summary.trace_on(graph=True, profiler=True)
        # Forward pass
        z = self.tracemodel(x.reshape(-1,1))
        with writer.as_default():
            tf.summary.trace_export(name='model_trace',
                                step=0, profiler_outdir=self.exp_dir+'/graph')
        writer.close()

class FFMDN(AbstractModel):
    def __init__(self, config):
        super(FFMDN, self).__init__(config)
        self.architecture_def(config)

    def architecture_def(self, config):
        """pi, mu, sigma = NN(x; theta)"""
        # for n in range(self.layers_n):
        self.h1 = Dense(self.neurons_n, activation='relu', name="h1")
        self.h2 = Dense(self.neurons_n, activation='relu', name="h2")
        self.alphas = Dense(self.components_n, activation=K.softmax, name="pi_long")
        self.mus_long = Dense(self.components_n, name="mus_long")
        self.sigmas_long = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        if config['exp_type']['model'] == 'driver_model':
            self.mus_lat = Dense(self.components_n, name="mus_lat")
            self.sigmas_lat = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
            self.rhos = Dense(self.components_n, activation=K.exp, name="rhos")

        self.pvector = Concatenate(name="output") # parameter vector

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        x = self.h1(inputs)
        x = self.h2(x)
        alpha_v = self.alphas(x)
        mu_v = self.mus_long(x)
        sigma_v = self.sigmas_long(x)

        return self.pvector([alpha_v, mu_v, sigma_v])


class GRUMDN(AbstractModel):
    pass
