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

# %%
"""
Generate data.
"""
x_len = 10
y_len = 10
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

def obsSequence(full_traj, x_len, y_len):
    traj_len = len(full_traj)
    snip_n = 10
    pred_horizon = 2 # number of snippets

    states = np.empty([traj_len, x_len, 1])
    targs = np.empty([traj_len, pred_horizon, 4])
    conds = np.empty([traj_len, pred_horizon, 4])
    coefs = np.empty([traj_len, 4])
    states[:] = np.nan
    targs[:] = np.nan
    coefs[:] = np.nan
    conds[:] = np.nan
    # create traj_snippets
    for i in range(snip_n):
        indx = [i]
        indx.extend(np.arange(i+snip_n-1, traj_len, snip_n))
        traj_snippets = full_traj[indx]
        f = CubicSpline(indx, traj_snippets)
        coefs[indx[:-1], :] = np.stack(f.c, axis=1)[:,:, 0] # number of splines = knots_n - 1

    for i in range(0, traj_len):
        x_sequence = full_traj[i:(i + x_len)]
        end_indx = i + x_len - 1
        states[end_indx, :, :] = x_sequence
        target_snippet_indxs = [end_indx+(snip_n-1)*n for n in range(pred_horizon)]
        cond_snippet_indxs = [(end_indx-(snip_n-1))+(snip_n-1)*n for n in range(pred_horizon)]

        if max(target_snippet_indxs) < traj_len:
            targs[end_indx, :, :] = coefs[target_snippet_indxs, :]
            conds[end_indx, :, :] = coefs[cond_snippet_indxs, :]
            # TODO make it varied sequence length
        else:
            break

    s_indx = np.argwhere(~np.isnan(states[:,:,0]).any(axis=1))
    t_indx = np.argwhere(~np.isnan(targs[:,:,0]).any(axis=1))
    c_indx = np.argwhere(~np.isnan(conds[:,:,0]).any(axis=1))
    indx = np.intersect1d(s_indx, t_indx, assume_unique=False)
    indx = np.intersect1d(indx, c_indx, assume_unique=False)
    return states[indx], targs[indx], conds[indx]


states_val, targs_val, conds_val = obsSequence(test.values, x_len, y_len)
states_train, targs_train, conds_train = obsSequence(train.values, x_len, y_len)
states_train.shape
targs_train.shape
conds_train.shape

# %%

plt.plot(range(9, 19), cubic_spline(x, cofs_val[0][0]))
plt.plot(cubic_spline(x, conds_val[0][0]))
# plt.plot(xs_val[0])
# %%

# %%
# %%
latent_dim = 20
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = keras.Input(shape=(None, 1))
decoder_lstm = keras.layers.LSTM(20, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)



dense_layer1 = keras.layers.Dense(1)
dense_outputs = dense_layer1(decoder_outputs)
# model = keras.Model(encoder_inputs, dense_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-2),
    loss='MeanSquaredError'
)

history = model.fit([states_train, conds_train[:, :, 0:1]*1000],
    targs_train[:, :, 0:1]*1000,
    batch_size=100,
    epochs=20,
    shuffle=False,
    validation_data=([states_val, conds_val[:, :, 0:1]*1000],
    targs_val[:, :, 0:1]*1000),
    verbose=1)

plt.plot(history.history['val_loss'][5:])
plt.plot(history.history['loss'][5:])
plt.legend(['val', 'train'])



# %%

model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs)
states_val, targs_val, conds_val = obsSequence(test.values, x_len, y_len)
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

cond = conds_val[sample][:, 0:1][0]*1000
cond.shape = (1,1,1)
# trial.shape = (1, 10)

states_value = enc_model.predict(state)
dec_seq = []
for i in range(2):
    output_, h, c = dec_model.predict([cond] + states_value)
    states_value = [h, c]
    cond = output_
    dec_seq.append(output_/1000)

dec_seq = np.array(dec_seq)
dec_seq.shape = (2, 1)
# %%
def get_spline(prev_coeffs, next_coeff):
    """Given parameters of previous spline and predicted next_coeff parameter, it returns
        full next_coeff params
    """
    x_eval = 9
    prev_snip = np.poly1d(prev_coeffs)
    p_dydx = np.poly1d(np.polyder(prev_snip, 1))(x_eval)
    p_d2ydx2 = np.poly1d(np.polyder(prev_snip, 2))(x_eval)

    coeffs = np.zeros(4)

    coeffs[0] = next_coeff
    coeffs[1] = p_d2ydx2/2
    coeffs[2] = p_dydx
    coeffs[3] = prev_snip(x_eval)

    return coeffs


targs_val[sample][0]
x= np.arange(0, 10, 1)
pointer = 9
weights2 = get_spline(weights1, np.array(dec_seq[1]))

plt.plot(np.poly1d(conds_val[sample][0])(x), color='grey', linestyle='--')
for step in range(2):
    if step == 0:
        weights = get_spline(conds_val[sample][0], np.array(dec_seq[0]))
    else:
        weights = get_spline(weights, np.array(dec_seq[step]))


    plt.plot(range(pointer, pointer+10), np.poly1d(targs_val[sample][step])(x), color='grey', linestyle='--')
    plt.plot(range(pointer, pointer+10), np.poly1d(weights)(x))
    pointer += 9
plt.grid()


# %%


    plt.plot(range(18, 28), np.poly1d(targs_val[sample][1])(x), color='grey', linestyle='--')

    plt.plot(np.poly1d(conds_val[sample][0]))(x)


plt.plot(cubic_spline(x, weights1))
plt.plot(range(9,19), cubic_spline(x, weights2))
plt.plot(cubic_spline(x, targs_val[sample][0]), color='red', linestyle='--')
plt.plot(range(9,19),cubic_spline(x, targs_val[sample][1]), color='grey', linestyle='--')

plt.legend(['pred', 'truth'])
plt.grid()
# %%

for sample in range (50,70):
    plt.figure()
    trial = xs_val[sample]
    trial.shape = (1, 10)

    # weights = [0.0033]
    weights = list(model(trial).numpy()[0]/1000)
    weights.extend(cofs_val[sample][1:].tolist())

    x= np.arange(0, 10, 1)
    plt.plot(cubic_spline(x, weights))
    plt.plot(cubic_spline(x, cofs_val[sample]))

    plt.legend(['pred', 'truth'])
    plt.grid()

# %%
