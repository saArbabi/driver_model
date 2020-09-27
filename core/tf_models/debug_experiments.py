import models.core.tf_models.abstract_model as am
from models.core.train_eval.utils import loadConfig
# from models.core.train_eval.model_evaluation import modelEvaluate
import tensorflow as tf
from models.core.tf_models import utils
from tensorflow_probability import distributions as tfd
from models.core.preprocessing.data_obj import DataObj

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from importlib import reload
import time

# %%
"""
Use this script for debugging the following:
- models.core.tf_models.utils
- models.core.tf_models.utils

Particularly ensure:
[] Distribution shapes are reasonable.
    See print('Distribution description: ',str(mvn))
    see print('covariance shape: ', cov.shape)
    see print('mu shape: ', mu.shape)
[] Shape of log_likelihood is reasonable
    See print('log_likelihood shape: ', log_likelihood.shape)

See:
https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes
"""
# %%
reload(am)
reload(utils)
nll_loss = utils.nll_loss
get_predictionMean = utils.get_predictionMean
get_pdf_samples = utils.get_pdf_samples

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

def modelTrain(config):
    model = am.FFMDN(config)
    optimizer = tf.optimizers.Adam(model.learning_rate)

    x_train, x_val, y_train, y_val = build_toy_dataset()
    train_ds = model.batch_data(x_train, y_train)
    test_ds = model.batch_data(x_val, y_val)

    write_graph = 'True'
    batch_i = 0
    t0 = time.time()
    for epoch in range(50):
        for xs, targets in train_ds:
            if write_graph == 'True':
                print(tf.shape(xs))
                graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
                tf.summary.trace_on(graph=True, profiler=False)
                model.train_step(xs, targets, optimizer)
                with graph_write.as_default():
                    tf.summary.trace_export(name='graph', step=0)
                write_graph = 'False'
            else:
                model.train_step(xs, targets, optimizer)
            model.save_batch_metrics(xs, targets, batch_i)
            batch_i += 1

        for xs, targets in test_ds:
            model.test_step(xs, targets)

        model.save_epoch_metrics(epoch)
    # modelEvaluate(model, validation_data, config)
    print('experiment duration ', time.time() - t0)
    return model

config = {
 "model_config": {
     "learning_rate": 1e-2,
     "neurons_n": 15,
     "layers_n": 3,
     "epochs_n": 30,
     "batch_n": 1024,
     "components_n": 20
},
"data_config": {},
"exp_id": "debug_experiment_1",
"model_type": "///",
"Note": ""
}
model = modelTrain(config)
X_train, X_test, y_train, y_test = build_toy_dataset()
predictions = model(X_test)
y_pred = get_pdf_samples(1, predictions, '///')

plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_pred)

# %%
reload(am)
reload(utils)
nll_loss = utils.nll_loss
get_predictionMean = utils.get_predictionMean
get_pdf_samples = utils.get_pdf_samples

config = loadConfig('series000exp001')
config['exp_id'] = 'debug_experiment_2'

def modelTrain(config):
    model = am.FFMDN(config)
    optimizer = tf.optimizers.Adam(model.learning_rate)
    x_train, y_train, x_val, y_val = DataObj(config).loadData()
    train_ds = model.batch_data(x_train, y_train)
    test_ds = model.batch_data(x_val, y_val)

    write_graph = 'True'
    batch_i = 0
    t0 = time.time()
    for epoch in range(1):
        for xs, targets in train_ds:
            if write_graph == 'True':
                print(tf.shape(xs))
                graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
                tf.summary.trace_on(graph=True, profiler=False)
                model.train_step(xs, targets, optimizer)
                with graph_write.as_default():
                    tf.summary.trace_export(name='graph', step=0)
                write_graph = 'False'
            else:
                model.train_step(xs, targets, optimizer)
            # model.save_batch_metrics(xs, targets, batch_i)
            batch_i += 1

        for xs, targets in test_ds:
            model.test_step(xs, targets)

        model.save_epoch_metrics(epoch)
    # modelEvaluate(model, validation_data, config)
    print('experiment duration ', time.time() - t0)
    return model


model = modelTrain(config)



# %%
y_pred = get_pdf_samples(1, predictions, '///')
# y_pred = get_predictionMean(predictions, '///')
plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_pred)


# %%
