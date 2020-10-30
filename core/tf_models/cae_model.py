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
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config, model_use):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.teacher_percent = config['model_config']['teacher_percent']
        self.steps_n = None # note self.steps_n =< self.pred_horizon
        self.model_use = model_use # can be training or inference
        self.architecture_def()
        self.create_tf_time_stamp(self.pred_horizon)

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        """Merger vehicle
        """
        self.lstm_layer_m = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.alphas_m = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_m = Dense(self.components_n, name="mus_long")
        self.sigmas_long_m = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_m = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_m = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
        self.rhos_m = Dense(self.components_n, activation=K.tanh, name="rhos")
        """Yielder vehicle
        """
        self.lstm_layer_y = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.alphas_y = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_y = Dense(self.components_n, name="mus_long")
        self.sigmas_long_y = Dense(self.components_n, activation=K.exp, name="sigmas_long")

    def create_tf_time_stamp(self, pred_horizon):
        ts = np.zeros([1, pred_horizon, pred_horizon])
        for i in range(pred_horizon):
            ts[0, i, :i+1] = 1

        self.time_stamp = tf.constant(ts, dtype='float32')

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
            zeros_pad_m = tf.zeros([batch_size, 1, 2])
            zeros_pad_y = tf.zeros([batch_size, 1, 1])

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)
            """Delete these later
            """
            zeros_pad_m = tf.zeros([batch_size, 1, 2])
            zeros_pad_y = tf.zeros([batch_size, 1, 1])

        # Initialize param vector
        param_m = tf.zeros([batch_size,0,30], dtype=tf.float32)
        param_y = tf.zeros([batch_size,0,15], dtype=tf.float32)

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state


        step_condition_m = tf.slice(conditions, [0, 0, 0], [batch_size, 1, 3])
        step_condition_y = step_condition_m

        state_h_m = state_h
        state_h_y = state_h

        state_c_m = state_c
        state_c_y = state_c

        for step in tf.range(steps_n):
        # for step in tf.range(3):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                        (param_m, tf.TensorShape([None,None,None])),
                        (param_y, tf.TensorShape([None,None,None])),
                        (step_condition_m, tf.TensorShape([None,None,3])),
                        (step_condition_y, tf.TensorShape([None,None,3])),
                        ])

            ts = tf.repeat(self.time_stamp[:, step:step+1, :], batch_size, axis=0)
            """Merger vehicle
            """
            outputs, state_h_m, state_c_m = self.lstm_layer_m(tf.concat([enc_h, step_condition_m], axis=2), \
                                                            initial_state=[state_h_m, state_c_m])
            outputs = tf.concat([outputs, ts], axis=2)
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
            outputs, state_h_y, state_c_y = self.lstm_layer_y(tf.concat([enc_h, step_condition_y], axis=2), \
                                                            initial_state=[state_h_y, state_c_y])
            outputs = tf.concat([outputs, ts], axis=2)
            alphas = self.alphas_y(outputs)
            mus_long = self.mus_long_y(outputs)
            sigmas_long = self.sigmas_long_y(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_y = self.concat_param_vecs(param_vec, param_y, step)
            """Conditioning
            """
            if step < steps_n-1:
                if self.model_use == 'training':
                    ################################
                    """Merger vehicle conditional
                    """
                    coin_flip = tf.random.uniform([1])
                    if coin_flip < self.teacher_percent:
                        # feed truth - Teacher forcing
                        act_m = tf.slice(conditions, [0,  step+1, 0], [batch_size, 1, 2])
                        act_y = tf.slice(conditions, [0,  step+1, 2], [batch_size, 1, 1])

                    else:
                        # feed zero
                        act_m = zeros_pad_m
                        act_y = tf.slice(conditions, [0,  step+1, 2], [batch_size, 1, 1])

                    step_condition_m = tf.concat([act_m, act_y], axis=2)
                    ################################
                    """Yielder vehicle conditional
                    """
                    coin_flip = tf.random.uniform([1])
                    if coin_flip < self.teacher_percent:
                        # feed truth - Teacher forcing
                        act_m = tf.slice(conditions, [0,  step+1, 0], [batch_size, 1, 2])
                        act_y = tf.slice(conditions, [0,  step+1, 2], [batch_size, 1, 1])

                    else:
                        # feed zero
                        act_m = tf.slice(conditions, [0,  step+1, 0], [batch_size, 1, 2])
                        act_y = zeros_pad_y

                    step_condition_y = tf.concat([act_m, act_y], axis=2)

                elif self.model_use == 'inference':
                    step_condition = tf.concat([sample_m, sample_y], axis=-1)

        gmm_m = get_pdf(param_m, 'merge_vehicle')
        gmm_y = get_pdf(param_y, 'other_vehicle')
        return gmm_m, gmm_y

class CAE(abstract_model.AbstractModel):
    def __init__(self, config, model_use):
        super(CAE, self).__init__(config)
        self.enc_model = Encoder(config)
        self.dec_model = Decoder(config, model_use)
        self.dec_model.total_batch_count = None

    def architecture_def(self):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.enc_model(inputs[0])
        return self.dec_model([inputs[1], encoder_states])
