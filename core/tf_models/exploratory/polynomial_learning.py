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
    traj_len = len(full_traj)
    snip_n = 10
    pred_h = 5 # number of snippets

    states = []
    targs = []
    conds = []
    coefs = np.empty([traj_len-snip_n, 4])
    coefs[:] = np.nan

    for i in range(snip_n):
        indx = []
        indx.extend(np.arange(i, traj_len, snip_n))
        traj_snippets = full_traj[indx]
        f = CubicSpline(indx, traj_snippets)
        coefs[indx[:-1], :] = np.stack(f.c, axis=2)[:,0,:] # number of splines = knots_n - 1

    coefs = coefs.tolist()
    for i in range(traj_len):
        end_indx = i + obs_n - 1
        targ_indx = [end_indx+(snip_n)*n for n in range(pred_h)]
        targ_indx = [num for num in targ_indx if num < len(coefs)]

        if len(targ_indx) == pred_h:
            cond_indx = [end_indx-snip_n]
            cond_indx.extend(targ_indx[:-1])
            targs.append([coefs[num] for num in targ_indx])
            conds.append([coefs[num] for num in cond_indx])
            states.append(full_traj[i:(i + obs_n), :].tolist())
        elif not targ_indx:
            break

    return np.array(states), np.array(targs), np.array(conds)


states_val, targs_val, conds_val = obsSequence(test.values, 10)
states_train, targs_train, conds_train = obsSequence(train.values, 10)

states_train.shape

# %%

# %%
states_train[0]
conds_train[0]
targs_train[0]

# %%

# plt.plot(xs_val[0])
# %%
x
pointer = 10
ts = targs_val
cs = conds_val
plt.plot(np.poly1d(cs[1][0])(x), color='grey')
for step in range(0, 5):
    weights = ts[1][step]

    if step%2 == 0:
        plt.plot(range(pointer, pointer+11), np.poly1d(weights)(x), color='red', linestyle='--')
    else:
        plt.plot(range(pointer, pointer+11), np.poly1d(weights)(x), color='blue', linestyle='--')

    pointer += 10
plt.grid()

# %%
# %%
latent_dim = 50
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, 1))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = keras.Input(shape=(None, 4))
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
history = model.fit([states_train, conds_train[:, :, :]*np.array([10000, 100, 10, 1])],
    targs_train[:, :, 0:1]*10000,
    batch_size=100,
    epochs=20,
    shuffle=False,
    validation_data=([states_val, conds_val[:, :, :]*np.array([10000, 100, 10, 1])],
    targs_val[:, :, 0:1]*10000),
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
def get_spline(prev_coeffs, next_coeff):
    """Given parameters of previous spline and predicted next_coeff parameter, it returns
        full next_coeff params
    """
    x_eval = 10
    prev_snip = np.poly1d(prev_coeffs)
    p_dydx = np.poly1d(np.polyder(prev_snip, 1))(x_eval)
    p_d2ydx2 = np.poly1d(np.polyder(prev_snip, 2))(x_eval)

    coeffs = np.zeros(4)

    coeffs[0] = next_coeff
    coeffs[1] = p_d2ydx2/2
    coeffs[2] = p_dydx
    coeffs[3] = prev_snip(x_eval)

    return coeffs
# %%
coefficients = [1.0, 2.5, -4.2]
x = 5.0
y = tf.math.polyval(coefficients, x)
y
# %%

sample = 73
state = states_val[sample]
state.shape = (1, 10, 1)

cond = conds_val[sample][:, :][0]
states_value = enc_model.predict(state)
dec_seq = []
seq_len = 20
for i in range(seq_len):
    cond.shape = (1,1,4)
    cond = cond*np.array([10000, 100, 10, 1])
    # cond *= np.array([[[1,0,0,0]]])
    output_, h, c = dec_model.predict([cond] + states_value)
    states_value = [h, c]
    cond.shape = (4)
    # cond = get_spline(cond, output_[0][0])
    output_.shape = 1
    cond = get_spline(cond*np.array([1/10000, 1/100, 1/10, 1]), output_/10000)
    # cond = get_spline(cond, output_/10000)
    dec_seq.append(cond.copy())


dec_seq = np.array(dec_seq)
dec_seq
# dec_seq.shape = 20
# plt.plot(dec_seq)
np.zeros([2,0,2])
# %%
plt.plot(range(20), dec_seq[:,0])
plt.scatter(range(20), dec_seq[:,0])
plt.grid()

# %%
x= np.arange(0, 11, 1)
pointer = 10
plt.plot(np.poly1d(conds_val[sample][0])(x), color='grey', linestyle='--')
for step in range(10):
    weights =  dec_seq[step]

    plt.plot(range(pointer, pointer+11), np.poly1d(weights)(x))
    pointer += 10
plt.grid()
