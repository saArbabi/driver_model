import models.core.tf_models.abstract_model as am
# from models.core.train_eval.model_evaluation import modelEvaluate
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import time

# %%
"""
Generate data.
"""
x_len = 10
y_len = 10
time_axis = np.arange(0, 500, 0.1)
time_stamp = np.arange(0, y_len/10, 0.1)
time_stamp.shape = (y_len,1)
scale=0.1
sin = np.sin(time_axis)
# plt.plot(sin[0:100])
# plt.plot(sin[0:100]+np.random.normal(scale=scale,size=(100)))

# sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))
df = pd.DataFrame(dict(sine=sin), index=time_axis, columns=['sine'])

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
# Take all numerical cols and normalize data b/w 0 and 1

def create_dataset(data, x_len, y_len):
    Xs, ys_input, ys_target = [], [], []
    for i in range(len(data) - x_len - y_len):
        r_data = np.float32(np.random.normal(scale=scale, size=(y_len, 1))) # random noise
        # r_data = 0
        x_sequence = data.iloc[i:(i + x_len)].values
        y_input_sequence = data.iloc[(i + x_len - 1):(i + x_len + y_len - 1)].values
        y_target_sequence = data.iloc[(i + x_len):(i + x_len + y_len)].values
        Xs.append(x_sequence)
        ys_input.append(y_input_sequence+r_data)
        ys_target.append(y_target_sequence+r_data)

    return np.array(Xs), np.array(ys_input), np.array(ys_target)
# reshape to [samples, time_steps, n_features]
xs, ys_input, ys_target = create_dataset(train, x_len, y_len)
xs_val, ys_input_val, ys_target_val = create_dataset(test, x_len, y_len)
print(xs.shape, ys_input.shape, ys_target.shape)
print(xs_val.shape, ys_input_val.shape, ys_target_val.shape)
# %%
reload(am)

def modelTrain(config):
    model = am.FFMDN(config)
    optimizer = tf.optimizers.Adam(model.learning_rate)
    train_ds = model.batch_data(xs, ys_input, ys_target)
    test_ds = model.batch_data(xs_val, ys_input_val, ys_target_val)

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


model = modelTrain(config)



# %%
y_pred = get_pdf_samples(1, predictions, '///')
# y_pred = get_predictionMean(predictions, '///')
plt.scatter(X_test, y_test)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_pred)


# %%
