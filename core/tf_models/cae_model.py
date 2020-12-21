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

    def none_linear_enc(self, inputs):
        inputs = self.linear_enc_1(inputs)
        inputs = self.linear_enc_2(inputs)
        inputs = self.linear_enc_3(inputs)
        return self.linear_enc_4(inputs)

    def architecture_def(self):
        self.lstm_layers_1 = LSTM(self.enc_units, return_sequences=True)
        self.lstm_layers_2 = LSTM(self.enc_units, return_state=True)
        self.linear_enc_1 = TimeDistributed(Dense(200, activation='relu'))
        self.linear_enc_2 = TimeDistributed(Dense(200, activation='relu'))
        self.linear_enc_3 = TimeDistributed(Dense(200, activation='relu'))
        self.linear_enc_4 = TimeDistributed(Dense(20))

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # _, state_h, state_c = self.lstm_layers(inputs)
        # _, state_h, state_c = self.lstm_layers(inputs)
        # return self.stacked_lstms(inputs)
        return self.stacked_lstms(self.none_linear_enc(inputs))

class Decoder(tf.keras.Model):
    def __init__(self, config, model_use):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.pred_step_n = config['data_config']['pred_step_n']
        self.steps_n = None # note self.steps_n =< self.pred_step_n
        self.model_use = model_use # can be training or inference
        self.architecture_def()

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layer_m = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_y = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_f = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_fadj = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.linear_layer_m = TimeDistributed(Dense(self.components_n*6))
        self.linear_layer_y = TimeDistributed(Dense(self.components_n*3))
        self.linear_layer_f = TimeDistributed(Dense(self.components_n*3))
        self.linear_layer_fadj = TimeDistributed(Dense(self.components_n*3))

        """Merger vehicle
        """
        self.alphas_mlon = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_mlon = Dense(self.components_n, name="mus")
        self.sigmas_mlon = Dense(self.components_n, activation=K.exp, name="sigmas")

        self.alphas_mlat = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_mlat = Dense(self.components_n, name="mus")
        self.sigmas_mlat = Dense(self.components_n, activation=K.exp, name="sigmas")
        """Yielder vehicle
        """
        self.alphas_y = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_y = Dense(self.components_n, name="mus")
        self.sigmas_y = Dense(self.components_n, activation=K.exp, name="sigmas")
        """F vehicle
        """
        self.alphas_f = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_f = Dense(self.components_n, name="mus")
        self.sigmas_f = Dense(self.components_n, activation=K.exp, name="sigmas")
        """Fadj vehicle
        """
        self.alphas_fadj = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_fadj = Dense(self.components_n, name="mus")
        self.sigmas_fadj = Dense(self.components_n, activation=K.exp, name="sigmas")

    def concat_vecs(self, step_gauss_param_vec, veh_gauss_param_vec, step):
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

    def teacher_force(self, true, sample):
        if tf.random.uniform([1]) > self.teacher_percent:
            return sample
        else:
            return true

    def sample_action(self, gmm, batch_size):
        """Also trim the actions so avoid errors cascating
        """
        action = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
        return tf.clip_by_value(action, -3.0, 3.0)

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        conditions = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training' or self.model_use == 'validating':
            batch_size = tf.shape(conditions[0])[0] # dynamiclaly assigned
            steps_n = tf.shape(conditions[0])[1] # dynamiclaly assigned

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)


        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state
        state_h_m = state_h
        state_c_m = state_c
        state_h_y = state_h
        state_c_y = state_c
        state_h_f = state_h
        state_c_f = state_c
        state_h_fadj = state_h
        state_c_fadj = state_c

        # Initialize param vector
        gauss_param_mlon = tf.zeros([batch_size,0, self.components_n*3], dtype=tf.float32)
        gauss_param_mlat = tf.zeros([batch_size,0, self.components_n*3], dtype=tf.float32)
        gauss_param_y = tf.zeros([batch_size,0, self.components_n*3], dtype=tf.float32)
        gauss_param_f = tf.zeros([batch_size,0, self.components_n*3], dtype=tf.float32)
        gauss_param_fadj = tf.zeros([batch_size,0, self.components_n*3], dtype=tf.float32)

        # Initialize action vector
        pred_act_mlon = tf.zeros([batch_size, 1, 1], dtype=tf.float32)
        pred_act_mlat = tf.zeros([batch_size, 1, 1], dtype=tf.float32)
        pred_act_y = tf.zeros([batch_size, 1, 1], dtype=tf.float32)
        pred_act_f = tf.zeros([batch_size, 1, 1], dtype=tf.float32)
        pred_act_fadj = tf.zeros([batch_size, 1, 1], dtype=tf.float32)

        # First step conditional
        act_mlon = tf.slice(conditions[0], [0, 0, 0], [batch_size, 1, 1])
        act_mlat = tf.slice(conditions[1], [0, 0, 0], [batch_size, 1, 1])
        act_y = tf.slice(conditions[2], [0, 0, 0], [batch_size, 1, 1])
        act_f = tf.slice(conditions[3], [0, 0, 0], [batch_size, 1, 1])
        act_fadj = tf.slice(conditions[4], [0, 0  , 0], [batch_size, 1, 1])

        step_cond_m = self.axis2_conc([act_mlon, act_mlat, act_y, act_f, act_fadj])
        step_cond_y = self.axis2_conc([act_mlon, act_mlat, act_y, act_fadj])
        step_cond_f = act_f
        step_cond_fadj = act_fadj


        for step in tf.range(steps_n):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (gauss_param_mlon, tf.TensorShape([None,None,None])),
                            (gauss_param_mlat, tf.TensorShape([None,None,None])),
                            (gauss_param_y, tf.TensorShape([None,None,None])),
                            (gauss_param_f, tf.TensorShape([None,None,None])),
                            (gauss_param_fadj, tf.TensorShape([None,None,None])),
                            (step_cond_m, tf.TensorShape([None,None,5])),
                            (step_cond_y, tf.TensorShape([None,None,4])),
                            (step_cond_f, tf.TensorShape([None,None,1])),
                            (step_cond_fadj, tf.TensorShape([None,None,1]))])

            """Merger vehicle long
            """
            outputs, state_h_m, state_c_m = self.lstm_layer_m(self.axis2_conc([step_cond_m, enc_h]), \
                                    initial_state=[state_h_m, state_c_m])
            outputs = self.linear_layer_m(outputs)

            alphas = self.alphas_mlon(outputs)
            mus = self.mus_mlon(outputs)
            sigmas = self.sigmas_mlon(outputs)
            gauss_param_vec = self.pvector([alphas, mus, sigmas])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_mlon = self.sample_action(gmm, batch_size)
            gauss_param_mlon = self.concat_vecs(gauss_param_vec, gauss_param_mlon, step)
            """Merger vehicle lat
            """
            alphas = self.alphas_mlat(outputs)
            mus = self.mus_mlat(outputs)
            sigmas = self.sigmas_mlat(outputs)
            gauss_param_vec = self.pvector([alphas, mus, sigmas])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_mlat = self.sample_action(gmm, batch_size)
            gauss_param_mlat = self.concat_vecs(gauss_param_vec, gauss_param_mlat, step)
            """Yielder vehicle
            """
            outputs, state_h_y, state_c_y = self.lstm_layer_y(self.axis2_conc([step_cond_y, enc_h]), \
                                    initial_state=[state_h_y, state_c_y])
            outputs = self.linear_layer_y(outputs)

            alphas = self.alphas_y(outputs)
            mus = self.mus_y(outputs)
            sigmas = self.sigmas_y(outputs)
            gauss_param_vec = self.pvector([alphas, mus, sigmas])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_y = self.sample_action(gmm, batch_size)
            gauss_param_y = self.concat_vecs(gauss_param_vec, gauss_param_y, step)
            """F vehicle
            """
            outputs, state_h_f, state_c_f = self.lstm_layer_f(self.axis2_conc([step_cond_f, enc_h]), \
                                    initial_state=[state_h_f, state_c_f])
            outputs = self.linear_layer_f(outputs)

            alphas = self.alphas_f(outputs)
            mus = self.mus_f(outputs)
            sigmas = self.sigmas_f(outputs)
            gauss_param_vec = self.pvector([alphas, mus, sigmas])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_f = self.sample_action(gmm, batch_size)
            gauss_param_f = self.concat_vecs(gauss_param_vec, gauss_param_f, step)
            """Fadj vehicle
            """
            outputs, state_h_fadj, state_c_fadj = self.lstm_layer_fadj(self.axis2_conc([step_cond_fadj, enc_h]), \
                                    initial_state=[state_h_fadj, state_c_fadj])
            outputs = self.linear_layer_fadj(outputs)

            alphas = self.alphas_fadj(outputs)
            mus = self.mus_fadj(outputs)
            sigmas = self.sigmas_fadj(outputs)
            gauss_param_vec = self.pvector([alphas, mus, sigmas])
            gmm = get_pdf(gauss_param_vec, 'other_vehicle')
            sample_fadj = self.sample_action(gmm, batch_size)
            gauss_param_fadj = self.concat_vecs(gauss_param_vec, gauss_param_fadj, step)

            """Conditioning
            """
            if self.model_use == 'training' or self.model_use == 'validating':
                if step < steps_n-1:
                    act_mlon = tf.slice(conditions[0], [0, step+1, 0], [batch_size, 1, 1])
                    act_mlat = tf.slice(conditions[1], [0, step+1, 0], [batch_size, 1, 1])
                    act_y = tf.slice(conditions[2], [0, step+1, 0], [batch_size, 1, 1])
                    act_f = tf.slice(conditions[3], [0, step+1, 0], [batch_size, 1, 1])
                    act_fadj = tf.slice(conditions[4], [0, step+1, 0], [batch_size, 1, 1])

                    step_cond_f = sample_f
                    step_cond_fadj = sample_fadj

                    step_cond_m = self.axis2_conc([sample_mlon, sample_mlat,
                                                            act_y,
                                                            act_f,
                                                            act_fadj])

                    step_cond_y = self.axis2_conc([act_mlon, act_mlat,
                                                            sample_y,
                                                            act_fadj])

            elif self.model_use == 'inference':
                step_cond_f = sample_f
                step_cond_fadj = sample_fadj

                step_cond_m = self.axis2_conc([sample_mlon, sample_mlat,
                                                        sample_y,
                                                        sample_f,
                                                        sample_fadj])

                step_cond_y = self.axis2_conc([sample_mlon, sample_mlat,
                                                        sample_y,
                                                        sample_fadj])

                pred_act_mlon = self.concat_vecs(sample_mlon, pred_act_mlon, step)
                pred_act_mlat = self.concat_vecs(sample_mlat, pred_act_mlat, step)
                pred_act_y = self.concat_vecs(sample_y, pred_act_y, step)
                pred_act_f = self.concat_vecs(sample_f, pred_act_f, step)
                pred_act_fadj = self.concat_vecs(sample_fadj, pred_act_fadj, step)

        if self.model_use == 'training' or self.model_use == 'validating':
            gmm_mlon = get_pdf(gauss_param_mlon, 'other_vehicle')
            gmm_mlat = get_pdf(gauss_param_mlat, 'other_vehicle')
            gmm_y = get_pdf(gauss_param_y, 'other_vehicle')
            gmm_f = get_pdf(gauss_param_f, 'other_vehicle')
            gmm_fadj = get_pdf(gauss_param_fadj, 'other_vehicle')

            return gmm_mlon, gmm_mlat, gmm_y, gmm_f, gmm_fadj

        elif self.model_use == 'inference':
            sampled_actions = [pred_act_mlon, pred_act_mlat, pred_act_y, \
                                                pred_act_f, pred_act_fadj]
            gmm_mlon = get_pdf(gauss_param_mlon, 'other_vehicle')
            gmm_mlat = get_pdf(gauss_param_mlat, 'other_vehicle')

            return sampled_actions
            # return sampled_actions, gmm_mlon, gmm_mlat



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
