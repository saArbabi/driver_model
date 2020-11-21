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
from scipy.interpolate import CubicSpline
import time
from collections import deque

# %%
"""
Generate data.
"""
time_axis = np.arange(0, 500, 0.1)
scale=0.1
sin = np.sin(time_axis)
df = pd.DataFrame(dict(sine=sin), index=time_axis, columns=['sine'])
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# %%
def cubic_spline(x, w):
    return w[0]*x**3 + w[1]*x**2 + w[2]*x + w[3]

def obsSequence(full_traj, obs_n):
    i_reset = 0
    i = 0
    traj_len = len(full_traj)
    snip_n = 10
    pred_h = 5 # number of snippets

    states = []
    targs = []
    conds = []

    prev_states = deque(maxlen=obs_n)
    while i < (len(full_traj)-pred_h):
        # 2 is minimum prediction horizon
        prev_states.append(full_traj[i])
        if len(prev_states) == obs_n:
            state_seq = np.array(prev_states)
            indx = np.arange(i, i+(pred_h+1)*snip_n, snip_n)
            indx = indx[indx<len(full_traj)]
            targ_indx = indx
            target_seq = full_traj[indx[1:]]
            cond_seq = full_traj[indx[:-1]]
            seq_len = len(target_seq)

            if seq_len == pred_h:
                states.append(state_seq)
                targs.append(target_seq)
                conds.append(cond_seq)

        i += 1

    return np.array(states), np.array(targs), np.array(conds)


# states_val, targs_val, conds_val = obsSequence(test.values[0:100], 10)
states_val, targs_val, conds_val = obsSequence(test.values, 10)
states_train, targs_train, conds_train = obsSequence(train.values, 10)
states_train.shape

# %%
latent_dim = 50
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = keras.Input(shape=(None, 1))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)



dense_layer1 = keras.layers.Dense(1)
dense_outputs = dense_layer1(decoder_outputs)
# model = keras.Model(encoder_inputs, dense_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='MeanSquaredError'
)
history = model.fit([states_train, conds_train],
    targs_train,
    batch_size=32,
    epochs=20,
    shuffle=False,
    validation_data=([states_val, conds_val],
    targs_val),
    verbose=1)

plt.plot(history.history['val_loss'][5:])
plt.plot(history.history['loss'][5:])
plt.legend(['val', 'train'])

# %%
# sample = 57
"""
Inference
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
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)

dec_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)
model.summary()
# %%
"""
Model evaluation
"""
sample = 73
state = states_val[sample]
state.shape = (1, 10, 1)

cond = conds_val[sample][0]
states_value = enc_model.predict(state)
dec_seq = []
seq_len = 40
for i in range(seq_len):
    output_, h, c = dec_model.predict([cond] + states_value)
    states_value = [h, c]
    # cond = get_spline(cond, output_[0][0])
    dec_seq.append(output_)

    output_.shape = (1,1,1)
    cond = output_
    # cond = get_spline(cond, output_/10000)


dec_seq = np.array(dec_seq)
dec_seq.shape = (40)
dec_seq

# %%
x = np.arange(0, 40, 1)
f = CubicSpline(x, dec_seq)
coefs = np.stack(f.c, axis=1)

x = np.arange(0, 1.1, 0.1)
start = 0
for c in coefs:
    plt.plot(x+start, np.poly1d(c)(x), color='grey')
    start += 1
plt.plot(dec_seq, linestyle='--')
plt.grid()
