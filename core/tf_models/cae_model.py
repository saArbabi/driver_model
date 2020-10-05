from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM, Masking
from models.core.tf_models.utils import get_pdf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from models.core.tf_models.abstract_model import  AbstractModel


class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['model_config']['enc_units']
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.pred_horizon = config['data_config']['pred_horizon']

        self.architecture_def()

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layers = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.masking = Masking(mask_value=0., input_shape=(self.pred_horizon, None))
        """Merger vehicle
        """
        self.alphas_m = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_m = Dense(self.components_n, name="mus_long")
        self.sigmas_long_m = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_m = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_m = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
        self.rhos_m = Dense(self.components_n, activation=K.tanh, name="rhos")
        """Yielder vehicle
        """
        self.alphas_y = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_y = Dense(self.components_n, name="mus_long")
        self.sigmas_long_y = Dense(self.components_n, activation=K.exp, name="sigmas_long")

    def call(self, inputs):
        # input[0] = conditions
        # input[1] = encoder states
        conditions = self.masking(inputs[0])
        outputs, state_h, state_c = self.lstm_layers(conditions, initial_state=inputs[1])
        self.state = [state_h, state_c]
        """Merger vehicle
        """
        alphas = self.alphas_m(outputs)
        mus_long = self.mus_long_m(outputs)
        sigmas_long = self.sigmas_long_m(outputs)
        mus_lat = self.mus_lat_m(outputs)
        sigmas_lat = self.sigmas_lat_m(outputs)
        rhos = self.rhos_m(outputs)
        param_vec_m = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
        gmm_m = get_pdf(param_vec_m, 'merge_vehicle')
        """Yielder vehicle
        """
        alphas = self.alphas_y(outputs)
        mus_long = self.mus_long_y(outputs)
        sigmas_long = self.sigmas_long_y(outputs)
        param_vec_y = self.pvector([alphas, mus_long, sigmas_long])
        gmm_y = get_pdf(param_vec_y, 'yield_vehicle')

        return gmm_m, gmm_y

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
