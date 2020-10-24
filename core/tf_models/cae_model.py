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
        self.enc_in_linear_units = config['model_config']['enc_in_linear_units']
        self.architecture_def()

    def in_linear(self, inputs):
        output = self.linear_layer_1(inputs)
        return self.linear_layer_2(output)

    def architecture_def(self):
        self.linear_layer_1 = TimeDistributed(Dense(self.enc_in_linear_units))
        self.linear_layer_2 = TimeDistributed(Dense(30))
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(self.in_linear(inputs))
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config, model_use):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.dec_in_linear_units = config['model_config']['dec_in_linear_units']
        self.dec_out_linear_units = config['model_config']['dec_out_linear_units']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.steps_n = None # note self.steps_n =< self.pred_horizon
        self.model_use = model_use # can be training or inference
        self.architecture_def()
        self.create_tf_time_stamp(self.pred_horizon)

    def in_linear(self, inputs):
        output = self.in_linear_layer(inputs)
        return output

    def dec_lstms(self, inputs, initial_state1, initial_state2):
        output1, state_h1, state_c1 = self.lstm_layer_1(inputs, initial_state=initial_state1)
        output2, state_h2, state_c2 = self.lstm_layer_2(output1, initial_state=initial_state2)
        return output2, [state_h1, state_c1], [state_h2, state_c2]

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.in_linear_layer = Dense(self.dec_in_linear_units)
        self.out_linear_layer = Dense(self.dec_out_linear_units)
        self.lstm_layer_1 = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_2 = LSTM(self.dec_units, return_sequences=True, return_state=True)

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

    def create_context_vec(self, enc_h, step_condition, ts):
        contex_vector = tf.concat([enc_h, step_condition, ts], axis=2)
        return self.in_linear(contex_vector)

    def concat_param_vecs(self, step_param_vec, veh_param_vec, step):
        """Use for concatinating gmm parameters across time-steps
        """
        if step == 0:
            return step_param_vec
        else:
            veh_param_vec = tf.concat([veh_param_vec, step_param_vec], axis=1)
            return veh_param_vec

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        conditions = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training':
            batch_size = tf.shape(conditions)[0] # dynamiclaly assigned
            steps_n = tf.shape(conditions)[1] # dynamiclaly assigned

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)

        # Initialize param vector
        param_m = tf.zeros([batch_size,0,30])
        param_y = tf.zeros([batch_size,0,15])
        param_f = tf.zeros([batch_size,0,15])
        param_fadj = tf.zeros([batch_size,0,15])

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state
        state_c_fadj = state_c
        step_condition = conditions[:, 0:1, :]

        state1 = state2 = inputs[1]
        for step in tf.range(steps_n):
        # for step in tf.range(3):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                        (param_m, tf.TensorShape([None,None,None])),
                        (param_y, tf.TensorShape([None,None,None])),
                        (param_f, tf.TensorShape([None,None,None])),
                        (param_fadj, tf.TensorShape([None,None,None])),
                        (step_condition, tf.TensorShape([None,None,5])),
                        ])

            ts = tf.repeat(self.time_stamp[:, step:step+1, :], batch_size, axis=0)
            contex_vector = self.create_context_vec(enc_h, step_condition, ts)
            outputs, state1, state2 = self.dec_lstms(contex_vector, state1, state2)



            outputs = self.out_linear_layer(tf.concat([contex_vector, outputs], axis=2))
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
            param_m = self.concat_param_vecs(param_vec, param_m, step)
            """Yielder vehicle
            """
            alphas = self.alphas_y(outputs)
            mus_long = self.mus_long_y(outputs)
            sigmas_long = self.sigmas_long_y(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_y = self.concat_param_vecs(param_vec, param_y, step)
            """F vehicle
            """
            alphas = self.alphas_f(outputs)
            mus_long = self.mus_long_f(outputs)
            sigmas_long = self.sigmas_long_f(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_f = self.concat_param_vecs(param_vec, param_f, step)
            """Fadj vehicle
            """
            alphas = self.alphas_fadj(outputs)
            mus_long = self.mus_long_fadj(outputs)
            sigmas_long = self.sigmas_long_fadj(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_fadj = self.concat_param_vecs(param_vec, param_fadj, step)
            """Conditioning
            """
            step_condition = tf.concat([sample_m, sample_y, sample_f, sample_fadj], axis=-1)

        gmm_m = get_pdf(param_m, 'merge_vehicle')
        gmm_y = get_pdf(param_y, 'other_vehicle')
        gmm_f = get_pdf(param_f, 'other_vehicle')
        gmm_fadj = get_pdf(param_fadj, 'other_vehicle')

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
