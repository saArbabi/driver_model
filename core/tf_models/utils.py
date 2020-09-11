import tensorflow as tf
from tensorflow_probability import distributions as tfd
import warnings
warnings.filterwarnings("always")

def get_CovMatrix(rhos, sigmas_long, sigmas_lat):
    covar = tf.math.multiply(tf.math.multiply(sigmas_lat,sigmas_long),rhos)
    diag_long = tf.math.square(sigmas_long)
    diag_lat = tf.math.square(sigmas_lat)

    col1 = tf.stack([diag_long, covar], axis=2)
    col2 = tf.stack([covar, diag_lat], axis=2)
    cov = tf.stack([col1, col2], axis=2, name='cov')
    return cov

def get_pdf(parameter_vector, config):

    if config['model_type'] == '///':
        alpha, mus, sigmas = slice_pvector(parameter_vector, config) # Unpack parameter vectors
        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(
                loc=mus,
                scale=sigmas))

    if config['model_type'] == 'merge_controller':
        alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos = slice_pvector(
                                                            parameter_vector, config)

        cov = get_CovMatrix(rhos, sigmas_long, sigmas_lat)

        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=alphas),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=tf.stack([mus_long, mus_lat], axis=2, name='mu'),
                scale_tril=tf.linalg.cholesky(cov)), name='MultivariateNormalTriL')
    return mvn

def slice_pvector(parameter_vector, config):
    """ Returns an unpacked list of paramter vectors.
    """
    if config['model_type'] == '///':
        n_params = 3 # number of parameters being learned
    if config['model_type'] == 'merge_controller':
        n_params = 6

    if tf.is_tensor(parameter_vector):
        return tf.split(parameter_vector, n_params, axis=1)
    else:
        return tf.split(parameter_vector, n_params, axis=0)


def nll_loss(config):
    def loss(y, parameter_vector):
        """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        """
        mvn = get_pdf(parameter_vector, config)
        log_likelihood = mvn.log_prob(y) # Evaluate log-probability of y

        return -tf.reduce_mean(log_likelihood)
    return loss

def get_predictionMean(parameter_vector, config):
    mvn = get_pdf(tf.convert_to_tensor(parameter_vector), config)
    return mvn.mean()

def get_pdf_samples(samples_n, parameter_vector, config):
    mvn = get_pdf(tf.convert_to_tensor(parameter_vector), config)
    return mvn.sample(samples_n)
