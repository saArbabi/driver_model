import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
import tensorflow as tf
from models.core.tf_models.utils import nll_loss, get_predictionMean, get_pdf_samples
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from importlib import reload


# %%
reload(am)

def modelTrain(config):
    model = am.FFMDN(config)
    optimizer = tf.optimizers.Adam(model.learning_rate)

    x_train, x_val, y_train, y_val = build_toy_dataset()
    train_ds = model.batch_data(x_train, y_train)
    test_ds = model.batch_data(x_val, y_val)

    write_graph = 'False'
    batch_i = 0

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
            # model.save_batch_metrics(xs, targets, batch_i)
            batch_i += 1

        for xs, targets in test_ds:
            model.test_step(xs, targets)

        model.save_epoch_metrics(epoch)
        print(model.test_loss.result())
    # modelEvaluate(model, validation_data, config)
    # model.save(model.exp_dir+'/model_dir',save_format='tf')
    return model

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "neurons_n": 15,
     "layers_n": 3,
     "epochs_n": 30,
     "batch_n": 1024,
     "components_n": 20
},
"data_config": {"step_size": 1,
                "obsSequence_n": 1,
                "m_s":["vel", "pc"],
                "y_s":["vel", "dv", "dx", "da", "a_ratio"],
                "retain":["vel"],
                # "Note": "baseline - no time stamps"
                # "Note": "Here I am adding the time stamp"
                "Note": "Here I remove partially correlated vehicles completely"
},
"exp_id": "debug_experiment_1",
"model_type": "///",
"Note": ""
}
model = modelTrain(config)
X_train, X_test, y_train, y_test = build_toy_dataset()
predictions = model(X_test)
y_pred = get_predictionMean(predictions, '///')
plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_pred)
# %%
# load data
X_train, X_test, y_train, y_test = build_toy_dataset()
predictions = model(X_test)
# y_pred = get_pdf_samples(1, predictions, '///')
y_pred = get_predictionMean(predictions, '///')
# plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_pred)
a = x = tf.constant([[ 1,  2,  3],
                  [ 4,  5,  6]])

a
tf.transpose(y_test)
y_test
predictions[0]
alphas, mus_long, sigmas_long = tf.split(predictions[0], 3, axis=0)
alphas
np.sum(alphas)
max(sigmas_long)
max()
min(sigmas_long)

# %%
y_train[0]
plt.scatter(x, means)
X_test.shape
