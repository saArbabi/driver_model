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
        target_snippet_indxs = [end_indx+snip_n*n for n in range(pred_horizon)]
        cond_snippet_indxs = [(end_indx-snip_n)+snip_n*n for n in range(pred_horizon)]

        if max(target_snippet_indxs) < traj_len:
            targs[end_indx, :, :] = coefs[target_snippet_indxs, :]
            conds[end_indx, :, :] = coefs[cond_snippet_indxs, :]
            # TODO make it varied sequence length
        else:
            break
            # target_snippet_indxs = [end_indx+i*n for n in range(pred_horizon)]

    s_indx = np.argwhere(~np.isnan(states[:,0,0]))
    t_indx = np.argwhere(~np.isnan(targs[:,0,0]))
    c_indx = np.argwhere(~np.isnan(conds[:,0,0]))
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

cofs.shape
xs_val.shape

 conds_val[:, :, 0:1].shape
conds_train[0:4]
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
    epochs=3,
    shuffle=False,
    validation_data=([states_val, conds_val[:, :, 0:1]*1000],
    targs_val[:, :, 0:1]*1000),
    verbose=1)

plt.plot(history.history['val_loss'][5:])
plt.plot(history.history['loss'][5:])
plt.legend(['val', 'train'])



# %%


# %%
# sample = 57
sample = 73
trial = xs_val[sample]
trial.shape = (1, 10)

# weights = [0.0033]
pred = list(model(trial).numpy()[0]/1000)
weights1 = [pred[0]]
weights2 = [pred[1]]
weights1.extend(cofs_val[sample][0][1:].tolist())
weights2.extend(cofs_val[sample][1][1:].tolist())

x= np.arange(0, 10, 1)
plt.plot(cubic_spline(x, weights1))
# plt.plot(range(9,19),cubic_spline(x, cofs_val[sample][1]))
# plt.plot(cubic_spline(x, cofs_val[sample][0]))
plt.plot(range(9,19), cubic_spline(x, weights2))

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
