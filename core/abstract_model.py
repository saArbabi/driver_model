import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import os
from tensorflow.python.ops import math_ops
import random

seed_value = 2020
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)


config = {
 "model_config": {
    "learning_rate": 1e-2,
    "hi": 2,
    "1": 2,
    "n_gmm_components": 4
},
"data_config": {
    "step_size": 3,
    "sequence_length": 5,
    "features": ['vel', 'pc','gap_size', 'dx', 'act_long_p', 'act_lat_p','lc_type'],
    "history_drop": {"percentage":0, "vehicle":'mveh'},
    "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx',
                                'act_long_p', 'act_lat_p', 'act_long', 'act_lat']},
    "scaler_path": './driver_model/experiments/scaler001'
},
"experiment_path": './driver_model/experiments/exp001',
"experiment_type": {"vehicle_name":'mveh', "model":"controller"}
}


class AbstractModel():
    def __init__(self, config):
        self.config = config['model_config']

    def _model_def(self):
        self.init_op = tf.global_variables_initializer()
        pi, sigma_lat, sigma_long, mu_lat, mu_long ,rho = self.get_mixture_coef(self.output)
        self.lossfunc = self.gnll_loss(self.ego_action, pi, sigma_lat, sigma_long,
                                                            mu_lat, mu_long ,rho)
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.lossfunc)

        self.training_loss = tf.summary.scalar('training_loss', tf.squeeze(self.lossfunc))
        self.validation_loss = tf.summary.scalar('validation_loss', tf.squeeze(self.lossfunc))
        # self.merged = tf.summary.merge_all()
        # cov_mat = self.cov_mat(sigma_lat, sigma_long)

    def _architecture_def(self):
        raise NotImplementedError()

    def get_mixture_coef(self, output):
        pi, sigma_lat, sigma_long, mu_lat, mu_long, rho = tf.split(
                    output, num_or_size_splits= 6, axis=1)
        pi = tf.nn.softmax(tf.keras.activations.linear(pi))
        rho = tf.nn.tanh(rho)

        sigma_lat =  math_ops.exp(sigma_lat)
        sigma_long = math_ops.exp(sigma_long)
        return pi, sigma_lat, sigma_long, mu_lat, mu_long, rho

    def gnll_loss(self, y, pi, sigma_lat, sigma_long, mu_lat, mu_long, rho):
        """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        """
        with tf.name_scope("gnll_loss") as scope:
            mu = tf.stack([mu_lat, mu_long], axis=2, name='mu')

            with tf.name_scope("Cov_mat") as scope:

                sig_lat_squared = tf.math.square(sigma_lat)
                sig_long_squared = tf.math.square(sigma_long)
                cor0 = tf.math.multiply(sigma_lat,sigma_long)
                cor1 = tf.math.multiply(cor0,rho)

                mat1 = tf.stack([sig_lat_squared, cor1], axis=2)
                mat2 = tf.stack([cor1, sig_long_squared], axis=2)
                cov = tf.stack([mat1, mat2], axis=2, name='cov')

            with tf.name_scope("Mixture_density") as scope:
                mvn = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(
                        probs=pi),
                    components_distribution=tfd.MultivariateNormalFullCovariance(
                        loc=mu,
                        covariance_matrix=cov[0]))
                shape = tf.shape(y)
                # Evaluate log-probability of y
                log_likelihood = mvn.log_prob(tf.reshape(y, [1, shape[0], 2]))

        return -tf.reduce_mean(log_likelihood, axis=-1)
