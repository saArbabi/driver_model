from numpy.random import seed # keep this at top
seed(2020)
import numpy as np
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM, TimeDistributed
from models.core.tf_models.utils import get_pdf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from models.core.tf_models import  abstract_model
from importlib import reload
reload(abstract_model)


class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['model_config']['enc_units']
        # self.enc_emb_units = config['model_config']['enc_emb_units']
        self.enc_emb_units = config['model_config']['enc_emb_units']
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
    def __init__(self, config, model_use):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.dec_emb_units = config['model_config']['dec_emb_units']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.steps_n = None # note self.steps_n =< self.pred_horizon
        self.model_use = model_use # can be training or inference
        self.architecture_def()
        self.create_tf_time_stamp(self.pred_horizon)

    def fc_embedding(self, inputs):
        output = self.embedding_layer_1(inputs)
        output = self.embedding_layer_2(output)
        return output

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.embedding_layer_1 = Dense(self.dec_emb_units)
        self.embedding_layer_2 = Dense(self.dec_emb_units)
        self.lstm_layer = LSTM(self.dec_units, return_sequences=True, return_state=True)
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

    def create_tf_time_stamp(self, pred_horizon):
        ts = np.zeros([1, pred_horizon, pred_horizon])
        for i in range(pred_horizon):
            ts[0, i, :i+1] = 1

        self.time_stamp = tf.constant(ts, dtype='float32')

    def create_context_vec(self, enc_h, step_condition, batch_size, step):
        ts = self.time_stamp[:, step:step+1, :]
        step_condition = self.fc_embedding(step_condition)
        contex_vector = tf.concat([step_condition, enc_h, \
                                    tf.repeat(ts, batch_size, axis=0)], axis=2)

        return contex_vector

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        vec_ms = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # parameter vectors for all the steps
        vec_ys = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        vec_fs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        vec_fadjs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        conditions = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training':
            batch_size = tf.shape(conditions)[0] # dynamiclaly assigned
            self.steps_n = tf.shape(conditions)[1] # dynamiclaly assigned

        elif self.model_use == 'inference' and not self.steps_n:
            raise AttributeError("The prediciton horizon must be set.")

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state
        step_condition = conditions[:, 0:1, :]

        for step in tf.range(self.steps_n):
            contex_vector = self.create_context_vec(enc_h, step_condition, batch_size, step)
            outputs, state_h, state_c = self.lstm_layer(contex_vector, \
                                                            initial_state=[state_h, state_c])
            """Merger vehicle
            """
            alphas = self.alphas_m(outputs)
            mus_long = self.mus_long_m(outputs)
            sigmas_long = self.sigmas_long_m(outputs)
            mus_lat = self.mus_lat_m(outputs)
            sigmas_lat = self.sigmas_lat_m(outputs)
            rhos = self.rhos_m(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
            gmm = get_pdf(param_vec, 'merge_vehicle')
            sample_m = tf.reshape(gmm.sample(1), [batch_size, 1, 2])

            vec_ms = vec_ms.write(vec_ms.size(), param_vec)
            """Yielder vehicle
            """
            alphas = self.alphas_y(outputs)
            mus_long = self.mus_long_y(outputs)
            sigmas_long = self.sigmas_long_y(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])

            vec_ys = vec_ys.write(vec_ys.size(), param_vec)
            """F vehicle
            """
            alphas = self.alphas_f(outputs)
            mus_long = self.mus_long_f(outputs)
            sigmas_long = self.sigmas_long_f(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])

            vec_fs = vec_fs.write(vec_fs.size(), param_vec)
            """Fadj vehicle
            """
            alphas = self.alphas_fadj(outputs)
            mus_long = self.mus_long_fadj(outputs)
            sigmas_long = self.sigmas_long_fadj(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])

            vec_fadjs = vec_fadjs.write(vec_fadjs.size(), param_vec)
            """Conditioning
            """
            step_condition = tf.concat([sample_m, sample_y, sample_f, sample_fadj], axis=-1)

        gmm_m = get_pdf(vec_ms.stack()[0], 'merge_vehicle')
        gmm_y = get_pdf(vec_ys.stack()[0], 'other_vehicle')
        gmm_f = get_pdf(vec_fs.stack()[0], 'other_vehicle')
        gmm_fadj = get_pdf(vec_fadjs.stack()[0], 'other_vehicle')

        return gmm_m, gmm_y, gmm_f, gmm_fadj

class CAE(abstract_model.AbstractModel):
    def __init__(self, config, model_use):
        super(CAE, self).__init__(config)
        self.enc_model = Encoder(config)
        self.dec_model = Decoder(config, model_use)

    def architecture_def(self):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.enc_model(inputs[0])
        return self.dec_model([inputs[1], encoder_states])
