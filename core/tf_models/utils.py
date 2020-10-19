import tensorflow as tf
from tensorflow_probability import distributions as tfd
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# %%
def get_CovMatrix(rhos, sigmas_long, sigmas_lat):
    covar = tf.math.multiply(tf.math.multiply(sigmas_lat,sigmas_long),rhos)
    col1 = tf.stack([tf.math.square(sigmas_long), covar], axis=3, name='stack')
    col2 = tf.stack([covar, tf.math.square(sigmas_lat)], axis=3, name='stack')
    # sigmas_long**2 is covariance of sigmas_long with itself
    cov = tf.stack([col1, col2], axis=3, name='cov')
    return cov

def get_pdf(param_vec, vehicle_type):
    # see https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
    # for info on shapes
    if vehicle == 'other_vehicle':
        alpha, mus, sigmas = slice_pvector(param_vec, vehicle_type) # Unpack parameter vectors
        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(
                loc=mus,
                scale=sigmas))

    if vehicle == 'merge_vehicle':
        alphas, mus_long, sigmas_long, mus_lat, \
                            sigmas_lat, rhos = slice_pvector(param_vec, vehicle_type)


        cov = get_CovMatrix(rhos, sigmas_long, sigmas_lat)
        mus = tf.stack([mus_long, mus_lat], axis=3, name='mus')
        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=alphas),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=mus,
                scale_tril=tf.linalg.cholesky(cov), name='MultivariateNormalTriL'))
    # print('mus shape: ', mus.shape)
    return mvn

def slice_pvector(param_vec, vehicle_type):
    """ Returns an unpacked list of paramter vectors.
    """
    if vehicle == 'other_vehicle':
        n_params = 3 # number of parameters being learned per GMM compnent
    if vehicle == 'merge_vehicle':
        n_params = 6

    if tf.is_tensor(param_vec):
        return tf.split(param_vec, n_params, axis=2)
    else:
        return tf.split(param_vec, n_params, axis=1)

def covDet_min(mvn):
    """Use as a metric
    """
    return tf.math.reduce_min(tf.linalg.det(mvn.covariance()))

def loss_merge(y, mvn):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        Loss for the merge vehicle
    """
    y_shape = y.shape
    log_likelihood = mvn.log_prob(tf.reshape(y, [y_shape[0], y_shape[1], 2]))

    # shape: [sample_shape, batch_shape, event_shape]
    return -tf.reduce_mean(log_likelihood)

def loss_other(y, mvn):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        Loss for the yield vehicle
    """
    y_shape = y.shape
    log_likelihood = mvn.log_prob(tf.reshape(y, [y_shape[0], y_shape[1]]))
    # shape: [sample_shape, batch_shape]

    return -tf.reduce_mean(log_likelihood)
