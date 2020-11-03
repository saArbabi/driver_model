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
        self.lstm_layer_m = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_y = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_ffadj = LSTM(self.dec_units, return_sequences=True, return_state=True)
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

    def concat_param_vecs(self, step_param_vec, veh_param_vec, step):
        """Use for concatinating gmm parameters across time-steps
        """
        if step == 0:
            return step_param_vec
        else:
            veh_param_vec = tf.concat([veh_param_vec, step_param_vec], axis=1)
            return veh_param_vec

    def axis2_conc(self, items_list):
        """concats tensor along the time-step axis(2)"""
        return tf.concat(items_list, axis=2)

    def mask_action(self, action, vehicle_type):
        coin_flip = tf.random.uniform([1])
        if coin_flip < self.teacher_percent:
            # feed truth - Teacher forcing
            return action
        else:
            # feed zero
            if vehicle_type == 'yield_vehicle':
                return self.zeros_pad_1

            else:
                return self.zeros_pad_2

    def flip_action(self, true_action, sampled_action):
        coin_flip = tf.random.uniform([1])
        if coin_flip < self.teacher_percent:
            # feed truth - Teacher forcing
            return true_action
        else:
            return sampled_action


    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        conditions = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training':
            batch_size = tf.shape(conditions)[0] # dynamiclaly assigned
            steps_n = tf.shape(conditions)[1] # dynamiclaly assigned
            self.zeros_pad_2 = tf.zeros([batch_size, 1, 2])
            self.zeros_pad_1 = tf.zeros([batch_size, 1, 1]) # for other single action cars

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)
            self.zeros_pad_2 = tf.zeros([batch_size, 1, 2])
            self.zeros_pad_1 = tf.zeros([batch_size, 1, 1]) # for other single action cars

        # Initialize param vector
        param_m = tf.zeros([batch_size,0,30], dtype=tf.float32)
        param_y = tf.zeros([batch_size,0,15], dtype=tf.float32)
        param_f = tf.zeros([batch_size,0,15], dtype=tf.float32)
        param_fadj = tf.zeros([batch_size,0,15], dtype=tf.float32)

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state

        # first step conditional
        act_m = tf.slice(conditions, [0,  0, 0], [batch_size, 1, 2])
        act_y = tf.slice(conditions, [0,  0, 2], [batch_size, 1, 1])
        act_ffadj = tf.slice(conditions, [0,  0, 3], [batch_size, 1, 2])
        step_cond_m = self.axis2_conc([act_m, act_y, act_ffadj])
        step_cond_y = self.axis2_conc([act_m, act_y, act_ffadj])
        step_cond_ffadj = act_ffadj

        # first step's LSTM cell and hidden state
        state_h_m = state_h
        state_c_m = state_c

        state_h_y = state_h
        state_c_y = state_c

        state_h_ffadj = state_h
        state_c_ffadj = state_c

        for step in tf.range(steps_n):
        # for step in tf.range(3):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                        (param_m, tf.TensorShape([None,None,None])),
                        (param_y, tf.TensorShape([None,None,None])),
                        (param_f, tf.TensorShape([None,None,None])),
                        (param_fadj, tf.TensorShape([None,None,None])),
                        (step_cond_m, tf.TensorShape([None,None,5])),
                        (step_cond_y, tf.TensorShape([None,None,5])),
                        (step_cond_ffadj, tf.TensorShape([None,None,2])),
                        ])

            # ts = tf.repeat(self.time_stamp[:, step:step+1, :], batch_size, axis=0)
            """Merger vehicle
            """
            outputs, state_h_m, state_c_m = self.lstm_layer_m(self.axis2_conc([enc_h, step_cond_m]), \
                                                            initial_state=[state_h_m, state_c_m])
            # outputs = self.axis2_conc([outputs, ts])
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
            outputs, state_h_y, state_c_y = self.lstm_layer_y(self.axis2_conc([enc_h, step_cond_y]), \
                                                            initial_state=[state_h_y, state_c_y])
            # outputs = self.axis2_conc([outputs, ts])
            alphas = self.alphas_y(outputs)
            mus_long = self.mus_long_y(outputs)
            sigmas_long = self.sigmas_long_y(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_y = self.concat_param_vecs(param_vec, param_y, step)
            """F vehicle
            """
            outputs, state_h_ffadj, state_c_ffadj = self.lstm_layer_ffadj(self.axis2_conc([enc_h, step_cond_ffadj]), \
                                                            initial_state=[state_h_ffadj, state_c_ffadj])
            # outputs = self.axis2_conc([outputs, ts])
            alphas = self.alphas_f(outputs)
            mus_long = self.mus_long_f(outputs)
            sigmas_long = self.sigmas_long_f(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_f = self.concat_param_vecs(param_vec, param_f, step)
            """Fadj vehicle
            """
            # outputs = self.axis2_conc([outputs, ts])
            alphas = self.alphas_fadj(outputs)
            mus_long = self.mus_long_fadj(outputs)
            sigmas_long = self.sigmas_long_fadj(outputs)
            param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(param_vec, 'other_vehicle')
            sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            param_fadj = self.concat_param_vecs(param_vec, param_fadj, step)

            """Conditioning
            """
            if step < steps_n-1:
                if self.model_use == 'training' and self.teacher_percent != 0:
                    ################################
                    act_m = tf.slice(conditions, [0, step+1, 0], [batch_size, 1, 2])
                    act_y = tf.slice(conditions, [0, step+1, 2], [batch_size, 1, 1])
                    act_f = tf.slice(conditions, [0, step+1, 3], [batch_size, 1, 1])
                    act_fadj = tf.slice(conditions, [0, step+1, 4], [batch_size, 1, 1])

                    # act_m_masked = self.mask_action(act_m, 'merge_vehicle')
                    # act_y_masked = self.mask_action(act_y, 'yield_vehicle')
                    # act_ffadj_masked = self.mask_action(act_ffadj, 'other_vehicle')
                    act_m_flipped = self.flip_action(act_m, sample_m)
                    act_y_flipped = self.flip_action(act_y, sample_y)
                    act_f_flipped = self.flip_action(act_f, sample_f)
                    act_fadj_flipped = self.flip_action(act_fadj, sample_fadj)

                    step_cond_ffadj = self.axis2_conc([act_f_flipped, act_fadj_flipped])
                    step_cond_m = self.axis2_conc([act_m_flipped, act_y, act_f, act_fadj])
                    step_cond_y = self.axis2_conc([act_m, act_y_flipped, act_f, act_fadj])

                elif self.teacher_percent == 0:
                    step_cond_ffadj = self.zeros_pad_2
                    step_cond_m = tf.zeros([batch_size, 1, 5])
                    step_cond_y = tf.zeros([batch_size, 1, 5])

                elif self.model_use == 'inference':
                    step_cond_ffadj = self.axis2_conc([sample_f, sample_fadj])
                    step_cond_m = self.axis2_conc([sample_m, sample_y, step_cond_ffadj])
                    step_cond_y = self.axis2_conc([sample_m, sample_y, step_cond_ffadj])

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
        self.dec_model.total_batch_count = None

    def architecture_def(self):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.enc_model(inputs[0])
        return self.dec_model([inputs[1], encoder_states])
