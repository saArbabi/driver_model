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
Generate data.
"""
x_len = 10
y_len = 10
time_axis = np.arange(0, 500, 0.1)
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
    Xs, ys_target = [], []
    for i in range(len(data) - x_len - y_len):
        x_sequence = data.iloc[i:(i + x_len)].values
        y_sequence = data.iloc[(i + x_len-1):(i + x_len + y_len-1)].values
        Xs.append(x_sequence)
        ys_target.append(y_sequence)

    return np.array(Xs), np.array(ys_target)
# reshape to [samples, time_steps, n_features]
xs, ys_target = create_dataset(train, x_len, y_len)
xs_val, ys_target_val = create_dataset(test, x_len, y_len)

xs.shape
ys_target.shape
xs[0]
ys_target[0]
# %%
def obsSequence(full_traj, x_len, y_len):
    traj_len = len(full_traj)
    set = np.empty([traj_len, 1 + x_len + 4]) # array index + x_len + n spline coefs
    set[:] = np.nan
    set[:, 0] = range(traj_len)
    # xs
    for i in range(traj_len - x_len):
        x_sequence = full_traj[i:(i + x_len)]
        end_indx = i + x_len - 1
        set[end_indx, 1:-4] = x_sequence

    # coeffcient
    for i in range(y_len):
        indx = np.arange(i, traj_len, y_len)
        traj_snippets = full_traj[indx]
        f = CubicSpline(indx, traj_snippets)
        set[indx[:-1], -4:] = np.stack(f.c, axis=1)

    # return set[:,1:-1], set[:, -1:]
    set = set[~np.isnan(set).any(axis=1)]
    coefs = set[:, -4:]
    features = np.reshape(set[:,1:-4], [len(set), 10, 1])
    return features, coefs

xs, cofs = obsSequence(np.ndarray.flatten(train.values), x_len, y_len)
xs_val, cofs_val = obsSequence(np.ndarray.flatten(test.values), x_len, y_len)


# set[-12]
cofs.shape
xs.shape

xs[0]
cofs[0]
# %%

sample = 57
plt.plot(xs_val[sample])
ins = np.arange(0,10,1)
def cubic_spline(x, w):
    return w[0]*x**3 + w[1]*x**2 + w[2]*x + w[3]
y = cubic_spline(ins, cofs_val[sample])
plt.plot(range(9,19), y)

# %%
cofs_val[:,0].min()
# %%
plt.plot(range(10), xs[0,:,:])
plt.plot(range(9,19), ys_target[0,:,:])


# %%
latent_dim = 20
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim)
encoder_outputs = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.

# Set up the decoder, using `encoder_states` as initial state.
param_n = 1
dense_layer1 = keras.layers.Dense(latent_dim)
dense_layer2 = keras.layers.Dense(param_n)
dense_outputs = dense_layer2(dense_layer1(encoder_outputs))
model = keras.Model(encoder_inputs, dense_outputs)
#
# def custom_loss(true, weights):
#     def poly_term(weight, power, batch_size):
#         x = tf.repeat([tf.linspace(0.0,4,5)], batch_size, axis=0)
#         return tf.math.multiply(weight, tf.math.pow(x, power))
#
#     def poly_fun(true, weights):
#
#         batch_size = tf.shape(weights)[0]
#         param_n = 1
#
#         # w_slice = tf.slice(weights, [0, 0], [batch_size, 1])
#         # print(weights.shape)
#         # print(weights[:, 0,0])
#         poly_terms = []
#
#         targ0 = tf.slice(true, [0, 0], [batch_size, 1])
#         targ1 = tf.slice(true, [0, 1], [batch_size, 1])
#         # targ2 = tf.slice(true, [0, 2], [batch_size, 1])
#
#         der1 = tf.math.subtract(targ1, targ0)
#         # der2 = tf.math.subtract(targ2, targ1)
#         # double_der = tf.math.divide(tf.math.subtract(der2, der1), 2)
#
#         poly_terms.append(poly_term(targ0, 0, batch_size))
#         poly_terms.append(poly_term(der1, 1, batch_size))
#         # poly_terms.append(poly_term(double_der, 2, batch_size))
#         poly_terms.append(poly_term(weights, 2, batch_size))
#
#         y = tf.math.add_n(poly_terms)
#
#         return y
#
#     pred = poly_fun(true, weights)
#     prior_mean = tf.square(tf.math.subtract(true, pred))
#
#     return tf.reduce_mean(prior_mean)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='MeanSquaredError'
)

history = model.fit(xs, cofs[:, 0]*1000,
    batch_size=100,
    epochs=40,
    shuffle=False,
    validation_data=(xs_val, cofs_val[:, 0]*1000),
    verbose=1)

plt.plot(history.history['val_loss'][5:])
plt.plot(history.history['loss'][5:])


# %%


# %%
# sample = 57
sample = 57
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
from scipy.interpolate import CubicSpline

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

f = CubicSpline(x, y)
x
y
a = CubicSpline([4, 5], [-0.93454613,-0.65364362])

plt.plot(a(np.arange(5,6,0.1)))
xs = np.arange(-0.5, 9.6, 0.1)
plt.plot(f(xs))
f.c
f.c
# %%

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
