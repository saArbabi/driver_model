import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from tensorflow import keras
import random
import json
from keras.callbacks import History
from models.core.tf_models.utils import nll_loss
from models.core.preprocessing.data_prep import DataObj

reload(utils)

# %%

# %%
# load data
# plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
config_base = utils.loadConfigBase('baseline_run.json')
model = am.FFMDN(config_base)
my_data = DataObj(config_base)
x_train, y_train, x_val ,y_val = my_data.data_prep()

model.compile(loss=nll_loss(config_base), optimizer=model.optimizer)


history = model.fit(x=x_train, y=y_train,epochs=3, validation_data=(x_val, y_val),
                    verbose=2, batch_size=1280, callbacks=model.callback)

len(x_val[0][0])
len(y_val[0])
print(history.history.keys())
print(history.history['loss'])

model.save(model.exp_dir+'/trained_model')

predictions = model.predict(y_test)
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
