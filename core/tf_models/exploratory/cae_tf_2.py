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
states[0]
targets[0]
conditions[0]
start = 0

for i in range(50):
    end = start + x_len
    plt.plot(range(start+x_len, end+y_len), targets[i])
    plt.plot(range(start, end), states[i])
    start += 1
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
start = 0
for i in range(50):
    end = start + x_len
    plt.plot(range(start+x_len, end+y_len), ys_target_val[i])
    plt.plot(range(start, end), xs_val[i])
    start += 1
# %%
states.shape
targets.shape
conditions.shape
tf.shape(targets)
self.nll_loss = lambda y, p_y: -p_y.log_prob()
tf.reshape(targets, (tf.shape(targets)[0], 10)).shape
# %%
reload(am)
enc_model = am.Encoder(config)
dec_model = am.Decoder(config)
model = am.CAE(enc_model, dec_model, config)

optimizer = tf.optimizers.Adam(model.learning_rate)
train_ds = model.batch_data(xs, ys_target, ys_input)
test_ds = model.batch_data(xs_val, ys_target_val, ys_input_val)

write_graph = 'True'
t0 = time.time()
train_loss = []
valid_loss = []
for epoch in range(5):
    for states, targets, conditions in train_ds:
        if write_graph == 'True':
            print(tf.shape(states))
            graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
            tf.summary.trace_on(graph=True, profiler=False)
            model.train_step(states, targets, conditions, optimizer)
            with graph_write.as_default():
                tf.summary.trace_export(name='graph', step=0)
            write_graph = 'False'
        else:
            model.train_step(states, targets, conditions, optimizer)

    for states, targets, conditions in test_ds:
        model.test_step(states, targets, conditions)
    train_loss.append(round(model.train_loss.result().numpy().item(), 2))
    valid_loss.append(round(model.test_loss.result().numpy().item(), 2))
    # model.save_epoch_metrics(states, targets, conditions, epoch)
# modelEvaluate(model, validation_data, config)
print('experiment duration ', time.time() - t0)

config = {
 "model_config": {
     "learning_rate": 1e-3,
     "neurons_n": 15,
     "layers_n": 3,
     "epochs_n": 10,
     "batch_n": 32,
     "components_n": 20
},
"data_config": {},
"exp_id": "debug_experiment_3",
"model_type": "///",
"Note": ""
}

plt.plot(train_loss)
plt.plot(valid_loss)
# %%

# %%
"""
Model evaluation
"""
# Reverse-lookup token index to decode sequences back to
# something readable.
y_feature_n = 1
target_seq = np.zeros((1, 1, y_feature_n))
t0 = time.time()
def decode_sequence(input_seq, input_t0, step_n):
    # Encode the input as state vectors.
    encoder_states_value = enc_model.predict(input_seq)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    sequences = []
    for n in range(5):
        states_value = encoder_states_value
        decoded_seq = []
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, y_feature_n))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 0] = input_t0
        # output_ = dec_model([target_seq, states_value])
        # print(output_.stddev())

        for i in range(step_n):
            output_ = dec_model([target_seq, states_value])
            h, c = dec_model.state_h, dec_model.state_c
            # print(output_.stddev())

            # Sample a token
            output_ = output_.sample(1).numpy()
            decoded_seq.append(output_)
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, y_feature_n))
            target_seq[0, 0, 0] = output_

            # Update states
            states_value = [h, c]

        sequences.append(np.array(decoded_seq).flatten())

    return np.array(sequences)

input_seq = xs_val[0]
step_n = 50
input_seq.shape = (1, 10, 1)
sequences = decode_sequence(input_seq, input_seq[0][-1][0], step_n)
# plt.plot(input_seq.flatten())
# plt.plot(range(9, 19), decoded_seq.flatten())
compute_time = time.time() - t0
compute_time
# %%
# %%
for i in range(5):
    plt.plot(range(9, 9+step_n), sequences[i], color='grey')

start = 0
for i in range(50):
    end = start + x_len
    plt.plot(range(start, end), xs_val[i].flatten(), color='red')
    start += 1
