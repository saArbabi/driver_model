import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time

# %%
"""
Ensure you can deal with the following:
[] running code in parallel (e.g. copying a single h for different sampled actions)
[] Generating multiple predictions ^^ somehow.
[] end-to-end learning ... a sine curve with two heads. One flipped of the other
[] How we doing with computaiton time?
"""
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
        y_conditioninguence = data.iloc[(i + x_len):(i + x_len + y_len)].values
        Xs.append(x_sequence)
        ys_input.append(y_input_sequence+r_data)
        ys_target.append(y_conditioninguence+r_data)

    return np.array(Xs), np.array(ys_input), np.array(ys_target)
# reshape to [samples, time_steps, n_features]
xs, ys_input, ys_target = create_dataset(train, x_len, y_len)
xs_val, ys_input_val, ys_target_val = create_dataset(test, x_len, y_len)
# X_test, y_test = create_dataset(test, x_len, y_len)
print(xs.shape, ys_input.shape, ys_target.shape)
print(xs_val.shape, ys_input_val.shape, ys_target_val.shape)
# plt.plot(range(x_len),xs[0])
# plt.plot(range(x_len,x_len+y_len),ys_input[0])
# plt.plot(range(x_len + 1,x_len+y_len + 1),ys_target[0])

ys_i_low_train, ys_t_low_train = ys_input-0.5, ys_target-0.5
ys_i_low_val, ys_t_low_val = ys_input_val-0.5, ys_target_val-0.5

ys_i_high_train, ys_t_high_train = ys_input+0.5, ys_target+0.5
ys_i_high_val, ys_t_high_val = ys_input_val+0.5, ys_target_val+0.5

# %%

start = 0
for i in range(50):
    end = start + x_len
    plt.plot(range(start+x_len, end+y_len), ys_t_high_val[i])
    plt.plot(range(start+x_len, end+y_len), ys_t_low_val[i])
    plt.plot(range(start, end), xs_val[i])
    start += 1

# %%

"""model configuration
"""
latent_dim = 20  # Latent dimensionality of the encoding space.
"""build model
"""

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, 2))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
high_dense = keras.layers.Dense(2)
low_dense = keras.layers.Dense(2)

high_dense_outputs = high_dense(decoder_outputs)
low_dense_outputs = low_dense(decoder_outputs)

high_normal = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., 0],
                                                    scale=tf.keras.backend.exp(t[..., 1])), name='high')
low_normal = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., 0],
                                                    scale=tf.keras.backend.exp(t[..., 1])), name='low')

high_output_dis = high_normal(high_dense_outputs)
low_output_dis = low_normal(low_dense_outputs)

# Define the model that will turn
# `encoder_input_data` & `ys_input` into `ys_target`
model = keras.Model([encoder_inputs, decoder_inputs], [high_output_dis, low_output_dis])
# model.summary()
"""
Train model
"""
# ys_input_val[0]
# tf.reshape(ys_input_val[0:32], (32,10))

# tf.transpose(ys_input_val[0]).shape[1]
# negloglik = lambda y, p_y: -p_y.log_prob(y)
negloglik_high = lambda y, p_y: -p_y.log_prob(tf.reshape(y, (tf.shape(y)[0], 10)))
negloglik_low = lambda y, p_y: -p_y.log_prob(tf.reshape(y, (tf.shape(y)[0], 10)))

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={
            'high':negloglik_high,
            'low':negloglik_low,
    },
    loss_weights=[1, 1],
)
ys_i_high_train.shape

history = model.fit(
    [xs, np.concatenate([ys_i_high_train, ys_i_low_train], axis=2)],
    {'high': ys_t_high_train, 'low':ys_t_low_train},
    batch_size=32,
    epochs=10,
    shuffle=False,
    validation_data=([xs_val, np.concatenate([ys_i_high_val, ys_i_low_val], axis=2)], {'high': ys_t_high_val, 'low':ys_t_low_val}),
    verbose=0)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'train_loss'])
plt.grid()


# %%
"""
Train inference
"""
import copy
# Define sampling models
model = copy.copy(model)
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
enc_model = keras.Model(encoder_inputs, encoder_states) # initialize inference_encoder

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
                            decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
high_dense = model.layers[4]
low_dense = model.layers[5]
high_dense_outputs = high_dense(decoder_outputs)
low_dense_outputs = low_dense(decoder_outputs)
high_normal =  model.layers[6]
low_normal =  model.layers[7]
high_output_dis = high_normal(high_dense_outputs)
low_output_dis = low_normal(low_dense_outputs)

dec_model = keras.Model(
    [decoder_inputs, decoder_states_inputs], [high_output_dis, low_output_dis, decoder_states]
)
model.summary()

# Define the model that will turn
# `encoder_input_data` & `ys_input` into `ys_target`
# %%
"""
Model evaluation
"""
# Reverse-lookup token index to decode sequences back to
# something readable.
y_feature_n = 2
conditioning = np.zeros((1, 1, y_feature_n))
t0 = time.time()
def decode_sequence(input_seq, cond, step_n):
    # Encode the input as state vectors.
    encoder_states_value = enc_model.predict(input_seq)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    cond_shape = [1, 1, 2]
    conditioning  = tf.reshape(cond, cond_shape)

    sequences_high = []
    sequences_low = []
    for n in range(20):
        states_value = encoder_states_value
        decoded_seq_high = []
        decoded_seq_low = []
        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        for i in range(step_n):
            high_dis, low_dis, states_value = dec_model([conditioning] + states_value)
            # print(output_.stddev())
            # Sample a token
            high_dis_sample = high_dis.sample(1).numpy()
            low_dis_sample = low_dis.sample(1).numpy()
            low_dis_sample = tf.reshape(low_dis_sample, [1])
            high_dis_sample = tf.reshape(high_dis_sample, [1])
            conditioning  = tf.concat([high_dis_sample, low_dis_sample], 0)
            conditioning  = tf.reshape(conditioning, cond_shape)

            # print(high_dis_sample)

            decoded_seq_high.append(high_dis_sample)
            decoded_seq_low.append(low_dis_sample)
            # Update the target sequence (of length 1).

            # Update states

        sequences_high.append(np.array(decoded_seq_high).flatten())
        sequences_low.append(np.array(decoded_seq_low).flatten())

    return np.array(sequences_high), np.array(sequences_low)

input_seq = xs_val[0]
step_n = 50
input_seq.shape = (1, 10, 1)

conditioing = np.concatenate([ys_i_high_val, ys_i_low_val], axis=2)[0]
cond = conditioing[0]
sequences_high,  sequences_low = decode_sequence(input_seq, cond, step_n)
# plt.plot(input_seq.flatten())
# plt.plot(range(9, 19), decoded_seq.flatten())
compute_time = time.time() - t0
compute_time
# %%

# %%
for i in range(20):
    plt.plot(range(9, 9+step_n), sequences_high[i], color='grey')

for i in range(20):
    plt.plot(range(9, 9+step_n), sequences_low[i], color='green')

start = 0
for i in range(50):
    end = start + x_len
    plt.plot(range(start, end), xs_val[i].flatten(), color='red')
    start += 1
