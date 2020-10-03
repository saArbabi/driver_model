
from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM

class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['enc_units']
        self.architecture_def(config)

    def architecture_def(self, config):
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.dec_units = config['dec_units']
        self.architecture_def(config)

    def architecture_def(self, config):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layers = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.alphas = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long = Dense(self.components_n, name="mus_long")
        self.sigmas_long = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        if self.model_type == 'merge_policy':
            self.mus_lat = Dense(self.components_n, name="mus_lat")
            self.sigmas_lat = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
            self.rhos = Dense(self.components_n, activation=K.tanh, name="rhos")

    def call(self, inputs):
        # input[0] = conditions
        # input[1] = encoder states
        outputs, state_h, state_c = self.lstm_layers(inputs[0], initial_state=inputs[1])
        self.state = [state_h, state_c]
        alphas = self.alphas(outputs)
        mus_long = self.mus_long(outputs)
        sigmas_long = self.sigmas_long(outputs)
        if self.model_type == 'merge_policy':
            mus_lat = self.mus_lat(outputs)
            sigmas_lat = self.sigmas_lat(outputs)
            rhos = self.rhos(outputs)
            parameter_vector = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
        else:
            parameter_vector = self.pvector([alphas, mus_long, sigmas_long])

        return parameter_vector

class CAE(AbstractModel):
    def __init__(self, encoder_model, decoder_model, config):
        super(CAE, self).__init__(config)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def architecture_def(self, X):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.encoder_model(inputs[0])
        return self.decoder_model([inputs[1], encoder_states])
