from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)

import os
from tensorflow.python.ops import math_ops
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow_probability import distributions as tfd
from datetime import datetime
# from models.core.tf_models.utils import self.nll_loss, varMin

# %%
class AbstractModel(tf.keras.Model):
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
        self.callback = self.callback_def()
        self.nll_loss = lambda y, p_y: -p_y.log_prob(tf.reshape(y, (tf.shape(y)[0], 10)))

    def architecture_def(self, X):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'variances')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, targets, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            predictions = self([states, conditions], training=True)
            loss = self.nll_loss(targets, predictions)

            if self.model_type == 'merge_policy':
                var_long_min, var_lat_min = varMin(predictions, self.model_type)
                tf.summary.scalar('var_long_min', var_long_min, step=epoch)
                tf.summary.scalar('var_lat_min', var_lat_min, step=epoch)

            elif self.model_type == '///':
                var_long_min = varMin(predictions, self.model_type)
                tf.summary.scalar('var_long_min', var_long_min, step=epoch)
        self.writer_2.flush()

    @tf.function
    def train_step(self, states, targets, conditions, optimizer):
        with tf.GradientTape() as tape:
            predictions = self([states, conditions], training=True)
            loss = self.nll_loss(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function
    def test_step(self, states, targets, conditions):
        predictions = self([states, conditions], training=False)
        loss = self.nll_loss(targets, predictions)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, states, targets, conditions):
        dataset = tf.data.Dataset.from_tensor_slices((states.astype("float32"),
                    targets.astype("float32"), conditions.astype("float32"))).batch(self.batch_n)
        return dataset

class FFMDN(AbstractModel):
    def __init__(self, config):
        super(FFMDN, self).__init__(config)
        self.architecture_def(config)

    def architecture_def(self, config):
        """pi, mu, sigma = NN(x; theta)"""
        # for n in range(self.layers_n):
        self.net_layers =  [Dense(self.neurons_n, activation='relu') for _
                                                in range(self.config['layers_n'])]
        self.dropout_layers = Dropout(0.25)

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
        x = self.net_layers[0](inputs)
        # x = self.dropout_layers(x)
        for layer in self.net_layers[1:]:
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

class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.latent_dim = 20
        self.architecture_def(config)

    def architecture_def(self, config):
        self.lstm_layers = LSTM(self.latent_dim, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.latent_dim = 20
        self.architecture_def(config)

    def architecture_def(self, config):
        self.lstm_layers = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.mus = Dense(1)
        self.sigmas = Dense(1, activation=K.exp)
        # self.pvector = Concatenate(name="output") # parameter vector

    def call(self, inputs):
        # input[0] = conditions
        # input[1] = encoder states
        with tf.name_scope("Decoder") as scope:
            decoder_outputs, state_h, state_c = self.lstm_layers(inputs[0], initial_state=inputs[1])
            mu = self.mus(decoder_outputs)
            sigma = self.sigmas(decoder_outputs)
            self.state_h = state_h
            self.state_c = state_c
            self.decoder_outputs = decoder_outputs

        return tfd.Normal(loc=mu[..., 0], scale=sigma[..., 0])

class CAE(AbstractModel):
    def __init__(self, encoder_model, decoder_model, config):
        super(CAE, self).__init__(config)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def architecture_def(self, X):
        pass

    def call(self, inputs):
        with tf.name_scope("CAE") as scope:
            # Defines the computation from inputs to outputs
            # input[0] = state obs
            # input[1] = conditions
            encoder_states = self.encoder_model(inputs[0])
        return self.decoder_model([inputs[1], encoder_states])
