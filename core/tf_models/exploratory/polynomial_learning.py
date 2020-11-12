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
y_len = 5
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

def custom_loss(true, weights):
    def poly_term(weight, power, batch_size):
        x = tf.repeat([tf.linspace(0.0,4,5)], batch_size, axis=0)
        return tf.math.multiply(weight, tf.math.pow(x, power))

    def poly_fun(true, weights):

        batch_size = tf.shape(weights)[0]
        param_n = 1

        w_slice = tf.slice(weights, [0, 0], [batch_size, 1])
        # print(weights.shape)
        # print(weights[:, 0,0])
        poly_terms = []

        targ0 = tf.slice(true, [0, 0], [batch_size, 1])
        targ1 = tf.slice(true, [0, 1], [batch_size, 1])
        targ2 = tf.slice(true, [0, 2], [batch_size, 1])

        der1 = tf.math.subtract(targ1, targ0)
        der2 = tf.math.subtract(targ2, targ1)
        double_der = tf.math.divide(tf.math.subtract(der2, der1), 2)

        poly_terms.append(poly_term(targ0, 0, batch_size))
        poly_terms.append(poly_term(der1, 1, batch_size))
        poly_terms.append(poly_term(double_der, 2, batch_size))
        poly_terms.append(poly_term(weights, 3, batch_size))

        y = tf.math.add_n(poly_terms)

        return y

    pred = poly_fun(true, weights)
    prior_mean = tf.square(tf.math.subtract(true, pred))

    return tf.reduce_mean(prior_mean)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=custom_loss
)

history = model.fit(xs, ys_target,
    batch_size=32,
    epochs=20,
    shuffle=False,
    validation_data=(xs_val, ys_target_val),
    verbose=1)

plt.plot(history.history['val_loss'][5:])
plt.plot(history.history['loss'][5:])


# %%
xs_val, ys_target_val = create_dataset(test, x_len, y_len)

trial.shape



# %%
sample =55
trial = xs_val[sample]
trial.shape = (1, 10)

weights = list(model(trial).numpy()[0])

targ0 = ys_target_val[sample][0][0]
targ1 = ys_target_val[sample][1][0]
targ2 = ys_target_val[sample][2][0]
der1 = targ1 - targ0
der2 = targ2 - targ1
double_der = (der2 - der1)/2

weights.append(double_der)
weights.append(der1)
weights.append(targ0)

def fun(weights):
    param_n = 3
    x = np.array(range(5))
    i = param_n
    poly_terms = []
    for w_n in weights:
        poly_terms.append(w_n*x**i)
        print(i)
        i -= 1

    # y = x**4 - 2*x**2
    return np.sum(poly_terms, axis=0)

# plt.plot(xs_val[sample])
plt.plot(fun(weights))
plt.plot(ys_target_val[sample])
plt.legend(['pred', 'truth'])

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
