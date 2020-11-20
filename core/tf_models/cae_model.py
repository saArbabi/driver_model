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
        self.pred_h = config['data_config']['pred_h']
        self.steps_n = None # note self.steps_n =< self.pred_h
        self.model_use = model_use # can be training or inference
        self.architecture_def()

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

    def concat_param_vecs(self, step_gauss_param_vec, veh_gauss_param_vec, step):
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

    def tf_polyder(self, spline_coeffs, batch_size):
        spline_coeffs = tf.slice(spline_coeffs, [0, 0, 0], [batch_size, 1, 3])
        der1 = tf.multiply(spline_coeffs, [3, 2, 1])

        spline_coeffs = tf.slice(der1, [0, 0, 0], [batch_size, 1, 2])
        der2 = tf.multiply(spline_coeffs, [6, 2])

        der1 = [tf.slice(der1, [0, 0, n], [batch_size, 1, 1]) for n in range(3)]
        der2 = [tf.slice(der2, [0, 0, n], [batch_size, 1, 1]) for n in range(2)]

        return der1, der2

    def scale(self, param, motion_type):
        return tf.divide(param, self.coef_scaler[motion_type])

    def unScale(self, param, motion_type):
        return tf.multiply(param, self.coef_scaler[motion_type])

    def get_spline(self, prev_spline, pred_coeff, spline_param, batch_size, motion_type):
        """
        Given parameters of previous spline and predicted pred_coeff
        parameter (both scaled), it returns scaled/unscaled spline coeffs.
        (1) un-scale coeffs
        (2) get new spline
        (3) concat new spline param to existing ones
        (4) scale coeffs - to be used as next-step conditional
        """
        # (1)
        pred_coeff = tf.multiply(pred_coeff, self.coef_scaler[motion_type][0])
        prev_spline = self.unScale(prev_spline, motion_type)

        # (2)
        x_eval = 5.0
        der1, der2 = self.tf_polyder(prev_spline, batch_size)
        p_dydx = tf.math.polyval(der1, x_eval)
        p_d2ydx2 = tf.math.polyval(der2, x_eval)
        prev_spline = [tf.slice(prev_spline, [0, 0, n], [batch_size, 1, 1]) for n in range(4)]
        next_spline = tf.concat([pred_coeff, tf.divide(p_d2ydx2, 2), p_dydx, \
                            tf.math.polyval(prev_spline, x_eval)], axis=1)

        # (3)
        spline_param = tf.concat([next_spline, spline_param], axis=1)

        # (4)
        next_spline = self.scale(next_spline, motion_type)

        return next_spline, spline_param

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        conditions = inputs[0]

        if self.model_use == 'training':
            batch_size = tf.shape(conditions[0])[0] # dynamiclaly assigned
            steps_n = tf.shape(conditions[0])[1] # dynamiclaly assigned

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)

        # input[1] = encoder states
        state_h, state_c = inputs[1] # encoder cell state
        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state

        # first step conditional
        coeff_mx = tf.slice(conditions[0], [0, 0, 0], [batch_size, 1, 4]) #long
        coeff_my = tf.slice(conditions[1], [0, 0, 0], [batch_size, 1, 4]) #lat
        coeff_y = tf.slice(conditions[2], [0, 0, 0], [batch_size, 1, 4])
        coeff_f = tf.slice(conditions[3], [0, 0, 0], [batch_size, 1, 4])
        coeff_fadj = tf.slice(conditions[4], [0, 0, 0], [batch_size, 1, 4])

        step_cond_m = self.axis2_conc([coeff_mx, coeff_my, coeff_y, coeff_f, coeff_fadj])
        step_cond_y = step_cond_m
        step_cond_ffadj = self.axis2_conc([coeff_f, coeff_fadj])

        # first step's LSTM cell and hidden state
        state_h_m = state_h
        state_c_m = state_c

        state_h_y = state_h
        state_c_y = state_c

        state_h_ffadj = state_h
        state_c_ffadj = state_c

        if self.model_use == 'training':
            # Initialize param vector
            gauss_param_m = tf.zeros([batch_size,0,30], dtype=tf.float32)
            gauss_param_y = tf.zeros([batch_size,0,15], dtype=tf.float32)
            gauss_param_f = tf.zeros([batch_size,0,15], dtype=tf.float32)
            gauss_param_fadj = tf.zeros([batch_size,0,15], dtype=tf.float32)

            for step in tf.range(steps_n):
                # for step in tf.range(3):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (gauss_param_m, tf.TensorShape([None,None,None])),
                            (gauss_param_y, tf.TensorShape([None,None,None])),
                            (gauss_param_f, tf.TensorShape([None,None,None])),
                            (gauss_param_fadj, tf.TensorShape([None,None,None])),
                            (step_cond_m, tf.TensorShape([None,None,20])),
                            (step_cond_y, tf.TensorShape([None,None,20])),
                            (step_cond_ffadj, tf.TensorShape([None,None,8])),
                            ])

                """Merger vehicle
                """
                outputs, state_h_m, state_c_m = self.lstm_layer_m(self.axis2_conc([enc_h, step_cond_m]), \
                                                                initial_state=[state_h_m, state_c_m])

                alphas = self.alphas_m(outputs)
                mus_long = self.mus_long_m(outputs)
                sigmas_long = self.sigmas_long_m(outputs)
                mus_lat = self.mus_lat_m(outputs)
                sigmas_lat = self.sigmas_lat_m(outputs)
                rhos = self.rhos_m(outputs)
                gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
                gmm = get_pdf(gauss_param_vec, 'merge_vehicle')
                sample_m = tf.reshape(gmm.sample(1), [batch_size, 1, 2])
                gauss_param_m = self.concat_param_vecs(gauss_param_vec, gauss_param_m, step)
                """Yielder vehicle
                """
                outputs, state_h_y, state_c_y = self.lstm_layer_y(self.axis2_conc([enc_h, step_cond_y]), \
                                                                initial_state=[state_h_y, state_c_y])

                alphas = self.alphas_y(outputs)
                mus_long = self.mus_long_y(outputs)
                sigmas_long = self.sigmas_long_y(outputs)
                gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(gauss_param_vec, 'other_vehicle')
                sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                gauss_param_y = self.concat_param_vecs(gauss_param_vec, gauss_param_y, step)
                """F vehicle
                """
                outputs, state_h_ffadj, state_c_ffadj = self.lstm_layer_ffadj(self.axis2_conc([enc_h, step_cond_ffadj]), \
                                                                initial_state=[state_h_ffadj, state_c_ffadj])
                alphas = self.alphas_f(outputs)
                mus_long = self.mus_long_f(outputs)
                sigmas_long = self.sigmas_long_f(outputs)
                gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(gauss_param_vec, 'other_vehicle')
                sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                gauss_param_f = self.concat_param_vecs(gauss_param_vec, gauss_param_f, step)
                """Fadj vehicle
                """
                alphas = self.alphas_fadj(outputs)
                mus_long = self.mus_long_fadj(outputs)
                sigmas_long = self.sigmas_long_fadj(outputs)
                gauss_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(gauss_param_vec, 'other_vehicle')
                sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                gauss_param_fadj = self.concat_param_vecs(gauss_param_vec, gauss_param_fadj, step)

                """Conditioning
                """
                if step < steps_n-1:
                    ################################
                    coeff_mx = tf.slice(conditions[0], [0, step+1, 0], [batch_size, 1, 4])
                    coeff_my = tf.slice(conditions[1], [0, step+1, 0], [batch_size, 1, 4])
                    coeff_y = tf.slice(conditions[2], [0, step+1, 0], [batch_size, 1, 4])
                    coeff_f = tf.slice(conditions[3], [0, step+1, 0], [batch_size, 1, 4])
                    coeff_fadj = tf.slice(conditions[4], [0, step+1, 0], [batch_size, 1, 4])

                    step_cond_ffadj = self.axis2_conc([coeff_f, coeff_fadj])
                    step_cond_m = self.axis2_conc([coeff_mx, coeff_my, coeff_y, coeff_f, coeff_fadj])
                    step_cond_y = step_cond_m

            gmm_m = get_pdf(gauss_param_m, 'merge_vehicle')
            gmm_y = get_pdf(gauss_param_y, 'other_vehicle')
            gmm_f = get_pdf(gauss_param_f, 'other_vehicle')
            gmm_fadj = get_pdf(gauss_param_fadj, 'other_vehicle')

            return gmm_m, gmm_y, gmm_f, gmm_fadj


        elif self.model_use == 'inference':
            # Initialize spline param vector
            spline_param_mx = self.unScale(coeff_mx, 'long')
            spline_param_my = self.unScale(coeff_my, 'lat')
            spline_param_y = self.unScale(coeff_y, 'long')
            spline_param_f = self.unScale(coeff_f, 'long')
            spline_param_fadj = self.unScale(coeff_fadj, 'long')

            for step in tf.range(steps_n):
                # for step in tf.range(3):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (spline_param_mx, tf.TensorShape([None,None,None])),
                            (spline_param_my, tf.TensorShape([None,None,None])),
                            (spline_param_y, tf.TensorShape([None,None,None])),
                            (spline_param_f, tf.TensorShape([None,None,None])),
                            (spline_param_fadj, tf.TensorShape([None,None,None])),
                            (step_cond_m, tf.TensorShape([None,None,20])),
                            (step_cond_y, tf.TensorShape([None,None,20])),
                            (step_cond_ffadj, tf.TensorShape([None,None,8])),
                            ])

                """Merger vehicle
                """
                outputs, state_h_m, state_c_m = self.lstm_layer_m(self.axis2_conc([enc_h, step_cond_m]), \
                                                                initial_state=[state_h_m, state_c_m])

                alphas = self.alphas_m(outputs)
                mus_long = self.mus_long_m(outputs)
                sigmas_long = self.sigmas_long_m(outputs)
                mus_lat = self.mus_lat_m(outputs)
                sigmas_lat = self.sigmas_lat_m(outputs)
                rhos = self.rhos_m(outputs)
                spline_param_vec = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
                gmm = get_pdf(spline_param_vec, 'merge_vehicle')
                sample_m = tf.reshape(gmm.sample(1), [batch_size, 1, 2])
                """Yielder vehicle
                """
                outputs, state_h_y, state_c_y = self.lstm_layer_y(self.axis2_conc([enc_h, step_cond_y]), \
                                                                initial_state=[state_h_y, state_c_y])

                alphas = self.alphas_y(outputs)
                mus_long = self.mus_long_y(outputs)
                sigmas_long = self.sigmas_long_y(outputs)
                spline_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(spline_param_vec, 'other_vehicle')
                sample_y = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                """F vehicle
                """
                outputs, state_h_ffadj, state_c_ffadj = self.lstm_layer_ffadj(self.axis2_conc([enc_h, step_cond_ffadj]), \
                                                                initial_state=[state_h_ffadj, state_c_ffadj])
                alphas = self.alphas_f(outputs)
                mus_long = self.mus_long_f(outputs)
                sigmas_long = self.sigmas_long_f(outputs)
                spline_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(spline_param_vec, 'other_vehicle')
                sample_f = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                """Fadj vehicle
                """
                alphas = self.alphas_fadj(outputs)
                mus_long = self.mus_long_fadj(outputs)
                sigmas_long = self.sigmas_long_fadj(outputs)
                spline_param_vec = self.pvector([alphas, mus_long, sigmas_long])
                gmm = get_pdf(spline_param_vec, 'other_vehicle')
                sample_fadj = tf.reshape(gmm.sample(1), [batch_size, 1, 1])
                """Conditioning
                """
                coeff_mx, spline_param_mx = self.get_spline(coeff_mx,
                                tf.slice(sample_m, [0, 0, 0], [batch_size, 1, 1]),
                                                spline_param_mx, batch_size, 'long')

                coeff_my, spline_param_my = self.get_spline(coeff_my,
                                tf.slice(sample_m, [0, 0, 1], [batch_size, 1, 1]),
                                                spline_param_my, batch_size, 'lat')

                coeff_y, spline_param_y = self.get_spline(coeff_y, sample_y,
                                                spline_param_y, batch_size, 'long')

                coeff_f, spline_param_f = self.get_spline(coeff_f, sample_f,
                                                spline_param_f, batch_size, 'long')

                coeff_fadj, spline_param_fadj = self.get_spline(coeff_fadj, sample_fadj,
                                                spline_param_fadj, batch_size, 'long')

                step_cond_ffadj = self.axis2_conc([coeff_f, coeff_fadj])
                step_cond_m = self.axis2_conc([coeff_mx, coeff_my, coeff_y, coeff_f, coeff_fadj])
                step_cond_y = step_cond_m

            return spline_param_mx, spline_param_my, spline_param_y, spline_param_f, spline_param_fadj

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
