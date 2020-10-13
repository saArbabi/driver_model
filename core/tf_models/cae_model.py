from numpy.random import seed # keep this at top
seed(2020)
import numpy as np
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM, Masking, TimeDistributed
from models.core.tf_models.utils import get_pdf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from models.core.tf_models.abstract_model import  AbstractModel


class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['model_config']['enc_units']
        # self.enc_emb_units = config['model_config']['enc_emb_units']
        self.enc_emb_units = 15
        self.architecture_def()

    def fc_embedding(self, inputs):
        output = self.embedding_layer_1(inputs)
        output = self.embedding_layer_2(output)
        return output


    def architecture_def(self):
        self.embedding_layer_1 = TimeDistributed(Dense(self.enc_emb_units))
        self.embedding_layer_2 = TimeDistributed(Dense(self.enc_emb_units))
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        inputs = self.fc_embedding(inputs)
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.dec_emb_units = 3
        self.time_stamp = np.zeros([1,1,self.pred_horizon], dtype='float32')

        self.architecture_def()

    def fc_embedding(self, inputs):
        output = self.embedding_layer_1(inputs)
        output = self.embedding_layer_2(output)
        return output

    def get_timeStamp(self, step, batch_size):
        if step == 0:
            self.time_stamp[0,0,0] = 1
        else:
            self.time_stamp[0,0,step-1] = 0
            self.time_stamp[0,0,step] = 1

        time_stamp = tf.repeat(self.time_stamp, batch_size, axis=0)
        return time_stamp

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.embedding_layer_1 = Dense(self.dec_emb_units)
        self.embedding_layer_2 = Dense(self.dec_emb_units)
        self.lstm_layer = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.masking = Masking(mask_value=0., batch_size=(self.pred_horizon, None))
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
        """F vehicle
        """
        self.alphas_f = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_f = Dense(self.components_n, name="mus_long")
        self.sigmas_long_f = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        """Fadj vehicle
        """
        self.alphas_fadj = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_fadj = Dense(self.components_n, name="mus_long")
        self.sigmas_long_fadj = Dense(self.components_n, activation=K.exp, name="sigmas_long")

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        vec_ms = [] # parameter vectors for all the steps
        vec_ys = []
        vec_fs = []
        vec_fadjs = []

        self.state = inputs[1]
        batch_size = inputs[0].shape[0]
        enc_h = tf.reshape(self.state[0], [batch_size, 1, self.dec_units]) # encoder hidden state
        conditions = self.masking(inputs[0])
        step_condition = tf.expand_dims(conditions[:, 0, :], axis=1)

        for i in range(self.pred_horizon):
        # for i in range(5):
            step_condition = self.fc_embedding(step_condition)
            time_stamp = self.get_timeStamp(i, batch_size)
            contex_vector = tf.concat([step_condition, enc_h, time_stamp], axis=2)

            outputs, state_h, state_c = self.lstm_layer(contex_vector, initial_state=self.state)
            self.state_m = [state_h, state_c]
            """Merger vehicle
            """
            alphas = self.alphas_m(outputs)
            mus_long = self.mus_long_m(outputs)
            sigmas_long = self.sigmas_long_m(outputs)
            mus_lat = self.mus_lat_m(outputs)
            sigmas_lat = self.sigmas_lat_m(outputs)
            rhos = self.rhos_m(outputs)
            param_vec_m = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
            vec_ms.append(param_vec_m)
            """Yielder vehicle
            """
            alphas = self.alphas_y(outputs)
            mus_long = self.mus_long_y(outputs)
            sigmas_long = self.sigmas_long_y(outputs)
            param_vec_y = self.pvector([alphas, mus_long, sigmas_long])
            vec_ys.append(param_vec_y)
            """F vehicle
            """
            alphas = self.alphas_f(outputs)
            mus_long = self.mus_long_f(outputs)
            sigmas_long = self.sigmas_long_f(outputs)
            param_vec_f = self.pvector([alphas, mus_long, sigmas_long])
            vec_fs.append(param_vec_f)
            """Fadj vehicle
            """
            alphas = self.alphas_fadj(outputs)
            mus_long = self.mus_long_fadj(outputs)
            sigmas_long = self.sigmas_long_fadj(outputs)
            param_vec_fadj = self.pvector([alphas, mus_long, sigmas_long])
            vec_fadjs.append(param_vec_fadj)

            if i != (self.pred_horizon - 1):
                gmm_m = get_pdf(param_vec_m, 'merge_vehicle')
                gmm_y = get_pdf(param_vec_y, 'other_vehicle')
                gmm_f = get_pdf(param_vec_f, 'other_vehicle')
                gmm_fadj = get_pdf(param_vec_fadj, 'other_vehicle')

                sample_m = tf.reshape(gmm_m.sample(1), [batch_size, 1, 2])
                sample_y = tf.reshape(gmm_y.sample(1), [batch_size, 1, 1])
                sample_f = tf.reshape(gmm_f.sample(1), [batch_size, 1, 1])
                sample_fadj = tf.reshape(gmm_fadj.sample(1), [batch_size, 1, 1])
                step_condition = tf.concat([sample_m, sample_y, sample_f, sample_fadj], axis=-1)

        vec_ms = tf.concat(vec_ms, axis=1)
        vec_ys = tf.concat(vec_ys, axis=1)
        vec_fs = tf.concat(vec_fs, axis=1)
        vec_fadjs = tf.concat(vec_fadjs, axis=1)
        gmm_m = get_pdf(vec_ms, 'merge_vehicle')
        gmm_y = get_pdf(vec_ys, 'other_vehicle')
        gmm_f = get_pdf(vec_fs, 'other_vehicle')
        gmm_fadj = get_pdf(vec_fadjs, 'other_vehicle')

        return gmm_m, gmm_y, gmm_f, gmm_fadj

class CAE(AbstractModel):
    def __init__(self, config):
        super(CAE, self).__init__(config)
        self.enc_model = Encoder(config)
        self.dec_model = Decoder(config)

    def architecture_def(self, X):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.enc_model(inputs[0])
        return self.dec_model([inputs[1], encoder_states])
