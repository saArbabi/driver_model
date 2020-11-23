import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from tensorflow import keras
import random
import json
from keras.callbacks import History
from models.core.tf_models.utils import nll_loss
from models.core.train_eval.utils import loadConfigBase
from models.core.preprocessing.data_prep import DataObj

# %%
# load data
# plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
config_base = loadConfigBase('baseline_run.json')
my_data = DataObj(config_base)
x_train, y_train, x_val ,y_val = my_data.data_prep()
# def reshape([x_train, y_train, x_val ,y_val]):
#     for item in
# %%
from models.core.tf_models import abstract_model
from models.core.tf_models import utils


reload(abstract_model)
reload(utils)
config_base['model_type']
model = abstract_model.FFMDN(config_base)


model.compile(loss=utils.nll_loss(config_base), optimizer=model.optimizer)
history = model.fit(x=x_train, y=y_train, epochs=1, validation_data=(x_val, y_val),
                    verbose=2, batch_size=128, callbacks=model.callback)
# %%
rhos, alphas, mus_long, sigmas_long, mus_lat, sigmas_lat = slice_pvector(
                                                    param_vec, config)
cov = get_CovMatrix(rhos, sigmas_long, sigmas_lat)
covar = tf.math.multiply(tf.math.multiply(sigmas_lat,sigmas_long),rhos)
diag_long = tf.math.square(sigmas_long)
diag_lat = tf.math.square(sigmas_lat)

col1 = tf.stack([diag_long, covar], axis=2)
col2 = tf.stack([covar, diag_lat], axis=2)
cov = tf.stack([col1, col2], axis=2, name='cov')
mvn = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=alphas),
    components_distribution=tfd.MultivariateNormalTriL(
        loc=tf.stack([mus_long, mus_lat], axis=2, name='mu'),
        scale_tril=tf.linalg.cholesky(cov)), name='MultivariateNormalTriL')



# %%
np.array(y_val)
a = np.array([1,2,3,4,5,6,7])
np.transpose(a)
x_train[0]
len(x_val[0][0])
len(y_val[0])
print(history.history.keys())
print(history.history['loss'])

# model.save(model.exp_dir+'/trained_model')
len(predictions[0])
predictions = model.predict(x_val)
mean_values = get_predictionMean(predictions, config_base)
plt.scatter(X_test, y_test)
plt.scatter(X_test[0:500], mean_values[0:500])
# %%
exp_dir = './models/experiments/exp004'
config = utils.loadConfig('exp004')
model = keras.models.load_model(exp_dir+'/trained_model',
                                        custom_objects={'loss': nll_loss(config)})
predictions = model.predict(y_test)
mean_values = get_predictionMean(predictions, config)
sampled_values = get_predictionSamples(1, predictions, config)
len(sampled_values[0])
plt.scatter(X_test, y_test)
plt.scatter(X_test[0:500], mean_values[0:500])
plt.scatter(X_test[0:500], sampled_values[0][0:500])



######################
model.predict(y_test[0])
import tensorflow as tf

@tf.function
def tracemodel(x):
    """Trace model execution - use for writing model graph
    :param: A sample input
    """
    return model(x)
log='./models/experiments/log1/%s' # stamp
writer = tf.summary.create_file_writer(log)
tf.summary.trace_on(graph=True, profiler=True)
# Forward pass
z = tracemodel(y_test[0].reshape(-1,1))
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log)
writer.close()
