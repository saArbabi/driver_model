import tensorflow as tf
from tensorflow_probability import distributions as tfd

def get_pdf(parameter_vector):
    alpha, mu, sigma = slice_pvector(parameter_vector) # Unpack parameter vectors
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    return gm

def slice_pvector(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    no_parameters = 3
    components = 4
    axis = 1

    if tf.shape(parameter_vector)[0] == 1:
        axis = 0

    return tf.split(parameter_vector, 3, axis=axis)

def nll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    gm = get_pdf(parameter_vector)
    log_likelihood = gm.log_prob(tf.transpose(y)) # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)
