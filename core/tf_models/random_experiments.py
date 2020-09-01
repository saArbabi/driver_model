

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

# load data
X_train, X_test, y_train, y_test = build_toy_dataset()
plt.scatter(X_test, y_test)
y_train[0]
plt.scatter(X_train, y_train)
plt.scatter(x, means)


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
