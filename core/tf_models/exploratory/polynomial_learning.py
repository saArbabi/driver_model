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
    set = np.empty([traj_len, 1 + x_len + 4]) # array index + x_len + n spline coefs
    set[:] = np.nan
    set[:, 0] = range(0, traj_len)

    # xs
    for i in range(0, traj_len - x_len):
        x_sequence = full_traj[i:(i + x_len)]
        end_indx = i + x_len - 1
        set[end_indx, 1:-4] = x_sequence

    # coeffcient
    for i in range(y_len):
        indx = [i]
        indx.extend(np.arange(i+y_len-1, traj_len, y_len))
        traj_snippets = full_traj[indx]
        f = CubicSpline(indx, traj_snippets)
        set[indx[:-1], -4:] = np.stack(f.c, axis=1) # number of splines = knots_n - 1

    set = set[~np.isnan(set).any(axis=1)]
    # pred_step = 2
    features = []
    coefs = []
    for i in range(x_len-1, len(set)-y_len):
        features.append(set[i,1:-4])
        coefs.append(set[[i,i+y_len-1], -4:])

    features = np.reshape(np.array(features), [len(features), 10, 1])
    return features, np.array(coefs)

xs_val, cofs_val = obsSequence(np.ndarray.flatten(test.values), x_len, y_len)
xs, cofs = obsSequence(np.ndarray.flatten(train.values), x_len, y_len)

cofs.shape

# %%
sample = 57
plt.plot(range(9,19), cubic_spline(x, cofs_val[sample][1]))
plt.plot(cubic_spline(x, cofs_val[sample][0]))
# plt.plot(range(9,19), cubic_spline(x, cofs_val[67][0]))
# %%
indx = [0]
# indx =
# a =

cofs.shape
xs_val.shape



# %%
# %%
latent_dim = 20
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim)
encoder_outputs = encoder(encoder_inputs)

dense_layer1 = keras.layers.Dense(latent_dim)
dense_layer2 = keras.layers.Dense(2)
dense_outputs = dense_layer2(dense_layer1(encoder_outputs))
model = keras.Model(encoder_inputs, dense_outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-2),
    loss='MeanSquaredError'
)

history = model.fit(xs, cofs[:, :, 0]*1000,
    batch_size=100,
    epochs=30,
    shuffle=False,
    validation_data=(xs_val, cofs_val[:, :, 0]*1000),
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
