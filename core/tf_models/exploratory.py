import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential

# %%
"""
Ensure you can deal with the following:
[] Masking/padding (varying seq length)
[] Multi-head inputs
[] Embeddings in hidden states (with varying embedding dim)
[] Multi-layers/long sequences
[] running code in parallel (e.g. copying a single h for different sampled actions)
[] end to end setup, predicting two outputs?
[] problem definition accounts for flaws of the past (e.g. model cheating)
[] iterative prediction -
[] action conditioning, ensuring no cheating occurs
"""

# %%
"""Autoencoder - stacking two RNNs
    Note conditioning here is only on encoder states.
"""
X = list()
Y = list()
X = [x for x in range(5, 301, 5)]
Y = [y for y in range(20, 316, 5)]

X = np.array(X).reshape(20, 3, 1)
Y = np.array(Y).reshape(20, 3, 1)

model = Sequential()

# encoder layer
model.add(layers.LSTM(100, activation='relu', input_shape=(3, 1)))

# repeat vector
model.add(layers.RepeatVector(3))

# decoder layer
model.add(layers.LSTM(100, activation='relu', return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')

print(model.summary())
history = model.fit(X, Y, epochs=30, validation_split=0.2, verbose=0, batch_size=3)
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'train_loss'])
# %%
"""Autoencoder - Recursive prediction
"""
time = np.arange(0, 100, 0.1)
sin = np.sin(time)
# sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))
df = pd.DataFrame(dict(sine=sin), index=time, columns=['sine'])

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
def create_dataset(data, x_len, y_len):
    Xs, ys_input, ys_target = [], [], []
    for i in range(len(data) - x_len - y_len):
        conditioning_param = np.random.choice(range(-3,3)) # to test if I can condition the model
        x_sequence = data.iloc[i:(i + x_len)].values
        y_input_sequence = data.iloc[(i + x_len):(i + x_len + y_len)].values
        y_target_sequence = data.iloc[(i + x_len + 1):(i + x_len + y_len + 1)].values + conditioning_param

        y_input_sequence = np.insert(y_input_sequence, [0], conditioning_param, axis=1)
        Xs.append(x_sequence)
        ys_input.append(y_input_sequence)
        ys_target.append(y_target_sequence)

    return np.array(Xs), np.array(ys_input), np.array(ys_target)

x_len = 10
y_len = 10
# reshape to [samples, time_steps, n_features]
xs, ys_input, ys_target = create_dataset(train, x_len, y_len)
xs_val, ys_input_val, ys_target_val = create_dataset(test, x_len, y_len)
# X_test, y_test = create_dataset(test, x_len, y_len)
len(y_test)
print(xs.shape, ys_input.shape, ys_target.shape)
print(xs_val.shape, ys_input_val.shape, ys_target_val.shape)
# plt.plot(range(x_len),xs[0])
# plt.plot(range(x_len,x_len+y_len),ys_input[0])
# plt.plot(range(x_len + 1,x_len+y_len + 1),ys_target[0])
# %%


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
decoder_dense = keras.layers.Dense(1)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `ys_input` into `ys_target`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# %%
"""
Train model
"""
model.compile(
    optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.MeanSquaredError()
)
history = model.fit(
    [xs, ys_input],
    ys_target,
    batch_size=32,
    epochs=50,
    validation_data=([xs_val, ys_input_val], ys_target_val),
    shuffle=False,
    verbose=0)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'train_loss'])

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
encoder_model = keras.Model(encoder_inputs, encoder_states) # initialize inference_encoder

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

decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)
# %%
# Reverse-lookup token index to decode sequences back to
# something readable.
y_feature_n = 2
target_seq = np.zeros((1, 1, y_feature_n))
def decode_sequence(input_seq, input_t0, step_n):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, y_feature_n))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 1] = input_t0
    conditioning_param = 1
    target_seq[0, 0, 0] = conditioning_param

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_seq = []
    for i in range(step_n):
        output_, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        decoded_seq.append(output_)
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, y_feature_n))
        target_seq[0, 0, 1] = output_
        target_seq[0, 0, 0] = conditioning_param

        # Update states
        states_value = [h, c]
    return decoded_seq

input_seq = xs_val[0]
step_n = 10
input_seq.shape = (1, 10, 1)
decoded_seq = decode_sequence(input_seq, ys_input_val[0][0][1], step_n)
np.array(decoded_seq).flatten()
# %%

# encoder_model.predict(input_seq)
len(x_true)
# plt.plot(range(100),x_true)
# plt.plot(range(x_len,x_len+y_len),ys_input_val[0])
plt.plot(range(y_len),ys_input_val[0][:, 1])
plt.plot(range(step_n), np.array(decoded_seq).flatten())

plt.plot(range(y_len),ys_input_val[0][:, 1])
plt.plot(range(step_n), np.array(decoded_seq).flatten())

# %%
for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
# %%
# test_input = np.array([300, 305, 310])
test_input = np.array([200, 205, 210])
test_input = test_input.reshape((1, 3, 1))
test_output = model.predict(test_input, verbose=0)
print(test_output)

# %%
