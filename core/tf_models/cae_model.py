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

    def stacked_lstms(self, inputs):
        output1 = self.lstm_layers_1(inputs)
        _, h_s2, c_s2 = self.lstm_layers_2(output1)
        return [h_s2, c_s2]

    def architecture_def(self):
        self.lstm_layers_1 = LSTM(self.enc_units, return_sequences=True)
        self.lstm_layers_2 = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # _, state_h, state_c = self.lstm_layers(inputs)
        # _, state_h, state_c = self.lstm_layers(inputs)
        return self.stacked_lstms(inputs)

class Decoder(tf.keras.Model):
    def __init__(self, config, model_use):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.pred_h = config['data_config']['pred_h']
        self.allowed_error = config['model_config']['allowed_error']
        self.steps_n = None # note self.steps_n =< self.pred_h
        self.model_use = model_use # can be training or inference
        self.architecture_def()

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layer_my = LSTM(self.dec_units, return_sequences=True, return_state=True)
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

    def concat_gauss_param_vecs(self, step_gauss_param_vec, veh_gauss_param_vec, step):
        """Use for concatinating gmm parameters across time-steps
        """
        if step == 0:
            return step_gauss_param_vec
        else:
            veh_gauss_param_vec = tf.concat([veh_gauss_param_vec, step_gauss_param_vec], axis=1)
            return veh_gauss_param_vec

    def axis2_conc(self, items_list):
        """concats tensor along the time-step axis(2)"""
        return tf.concat(items_list, axis=2)

    def teacher_check(self, true, sample, vehicle_type):
        if vehicle_type == 'merge_vehicle':
            allowed_error = self.allowed_error
        elif vehicle_type == 'other_vehicle':
            allowed_error = self.allowed_error[0]

        error = tf.math.abs(tf.math.subtract(sample, true))
        less = tf.cast(tf.math.less(error, allowed_error), dtype='float')
        greater = tf.cast(tf.math.greater_equal(error, allowed_error), dtype='float')
        return  tf.math.add(tf.multiply(greater, true), tf.multiply(less, sample))

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        conditions = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training':
            batch_size = tf.shape(conditions[0])[0] # dynamiclaly assigned
            steps_n = tf.shape(conditions[0])[1] # dynamiclaly assigned

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state
        state_h_my = state_h
        state_c_my = state_c
        state_h_ffadj = state_h
        state_c_ffadj = state_c

        # Initialize param vector
        gauss_param_m = tf.zeros([batch_size,0,30], dtype=tf.float32)
        gauss_param_y = tf.zeros([batch_size,0,15], dtype=tf.float32)
        gauss_param_f = tf.zeros([batch_size,0,15], dtype=tf.float32)
        gauss_param_fadj = tf.zeros([batch_size,0,15], dtype=tf.float32)

        # first step conditional
        act_m = tf.slice(conditions[0], [0, 0, 0], [batch_size, 1, 2])
        act_y = tf.slice(conditions[1], [0, 0, 0], [batch_size, 1, 1])
        act_f = tf.slice(conditions[2], [0, 0, 0], [batch_size, 1, 1])
        act_fadj = tf.slice(conditions[3], [0, 0, 0], [batch_size, 1, 1])

        step_cond_ffadj = self.axis2_conc([act_f, act_fadj])
        step_cond_my = self.axis2_conc([act_m, act_y, step_cond_ffadj])

        for step in tf.range(steps_n):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (gauss_param_m, tf.TensorShape([None,None,None])),
                            (gauss_param_y, tf.TensorShape([None,None,None])),
                            (gauss_param_f, tf.TensorShape([None,None,None])),
                            (gauss_param_fadj, tf.TensorShape([None,None,None])),
                            (step_cond_my, tf.TensorShape([None,None,5])),
                            (step_cond_ffadj, tf.TensorShape([None,None,2]))])

            outputs_my, state_h_my, state_c_my = self.lstm_layer_my(\
                                    self.axis2_conc([enc_h, step_cond_my]), \
                                    initial_state=[state_h_my, state_c_my])

            outputs_ffadj, state_h_ffadj, state_c_ffadj = self.lstm_layer_ffadj(\
                                    self.axis2_conc([enc_h, step_cond_ffadj]), \
                                    initial_state=[state_h_ffadj, state_c_ffadj])

            """Merger vehicle
            """
            alphas = self.alphas_m(outputs_my)
            mus_long = self.mus_long_m(outputs_my)
            sigmas_long = self.sigmas_long_m(outputs_my)
            mus_lat = self.mus_lat_m(outputs_my)
            sigmas_lat = self.sigmas_lat_m(outputs_my)
            rhos = self.rhos_m(outputs_my)
            gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
            gmm = get_pdf(gauss_param_vec, 'merge_vehicle')
            sample_m = tf.reshape(gmm.sample(1), [batch_size, 1, 2])
            gauss_param_m = self.concat_gauss_param_vecs(gauss_param_vec, gauss_param_m, step)
            """Yielder vehicle
            """
            alphas = self.alphas_y(outputs_my)
            mus_long = self.mus_long_y(outputs_my)
            sigmas_long = self.sigmas_long_y(outputs_my)
            gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            gauss_param_y = self.concat_gauss_param_vecs(gauss_param_vec, gauss_param_y, step)
            """F vehicle
            """
            alphas = self.alphas_f(outputs_ffadj)
            mus_long = self.mus_long_f(outputs_ffadj)
            sigmas_long = self.sigmas_long_f(outputs_ffadj)
            gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            gauss_param_f = self.concat_gauss_param_vecs(gauss_param_vec, gauss_param_f, step)
            """Fadj vehicle
            """
            alphas = self.alphas_fadj(outputs_ffadj)
            mus_long = self.mus_long_fadj(outputs_ffadj)
            sigmas_long = self.sigmas_long_fadj(outputs_ffadj)
            gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
            gauss_param_fadj = self.concat_gauss_param_vecs(gauss_param_vec, gauss_param_fadj, step)
            """Conditioning
            """
            if step < steps_n-1:
                if self.model_use == 'training':
                    ################################
                    act_m = tf.slice(conditions[0], [0, step+1, 0], [batch_size, 1, 2])
                    act_y = tf.slice(conditions[1], [0, step+1, 0], [batch_size, 1, 1])
                    act_f = tf.slice(conditions[2], [0, step+1, 0], [batch_size, 1, 1])
                    act_fadj = tf.slice(conditions[3], [0, step+1, 0], [batch_size, 1, 1])

                    if self.allowed_error != [0, 0]:
                        act_m_checked = self.teacher_check(act_m, sample_m, 'merge_vehicle')
                        act_y_checked = self.teacher_check(act_y, sample_y, 'other_vehicle')
                        act_f_checked = self.teacher_check(act_f, sample_f, 'other_vehicle')
                        act_fadj_checked = self.teacher_check(act_fadj, sample_fadj, 'other_vehicle')

                        step_cond_ffadj = self.axis2_conc([act_f_checked, act_fadj_checked])
                        step_cond_my = self.axis2_conc([act_m_checked, act_y_checked, \
                                                                        step_cond_ffadj])
                    else:
                        step_cond_ffadj = self.axis2_conc([act_f, act_fadj])
                        step_cond_my = self.axis2_conc([act_m, act_y, step_cond_ffadj])

                elif self.model_use == 'inference':
                    step_cond_ffadj = self.axis2_conc([sample_f, sample_fadj])
                    step_cond_my = self.axis2_conc([sample_m, sample_y, step_cond_ffadj])

        gmm_m = get_pdf(gauss_param_m, 'merge_vehicle')
        gmm_y = get_pdf(gauss_param_y, 'other_vehicle')
        gmm_f = get_pdf(gauss_param_f, 'other_vehicle')
        gmm_fadj = get_pdf(gauss_param_fadj, 'other_vehicle')

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
