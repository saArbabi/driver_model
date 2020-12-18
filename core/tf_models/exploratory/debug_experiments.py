from models.core.tf_models.cae_model import CAE
from models.core.train_eval.utils import loadConfig
# from models.core.train_eval.model_evaluation import modelEvaluate
import tensorflow as tf
from models.core.tf_models import utils
from tensorflow_probability import distributions as tfd
from models.core.preprocessing.data_obj import DataObj

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from importlib import reload
import time

exp_trains = {}
exp_vals = {}
durations = {}

# %%
def teacher_check(true, sample):
    allowed_error = 1
    error = tf.math.abs(tf.math.subtract(sample, true))
    less = tf.cast(tf.math.less(error, allowed_error), dtype='float')
    greater = tf.cast(tf.math.greater_equal(error, allowed_error), dtype='float')
    return  tf.math.add(tf.multiply(greater, true), tf.multiply(less, sample))

true = tf.constant([[5],[1.3]])
max = tf.constant([[4],[1.1]])
min = tf.constant([[2.9],[0.9]])
tf.clip_by_value(true, clip_value_min=min, clip_value_max=max)

# %%
sample = tf.constant([[2,0.4],[2.9,1]])
teacher_check(true, sample)
# %%
"""
Use this script for debugging the following:
- models.core.tf_models.utils

Particularly ensure:
[] Distribution shapes are reasonable.
    See range(3, config['data_config']['pred_horizon'])('Distribution description: ',str(mvn))
    see range(3, config['data_config']['pred_horizon'])('covariance shape: ', cov.shape)
    see range(3, config['data_config']['pred_horizon'])('mu shape: ', mu.shape)
[] Shape of log_likelihood is reasonable
    See range(3, config['data_config']['pred_horizon'])('log_likelihood shape: ', log_likelihood.shape)

See:
https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes
"""
# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "enc_units": 100,
     "dec_units": 100,
     "epochs_n": 50,
     "components_n": 10,
    "batch_size": 256,
    "teacher_percent": 0.7,
},
"data_config": {"obs_n": 20,
                "pred_step_n": 4,
                "step_size": 5,
                "Note": "lat/long motion not considered jointly"
                # "Note": "jerk as target"

},
"exp_id": "NA",
"Note": ""
}

from models.core.tf_models import utils
reload(utils)

from models.core.tf_models import cae_model
reload(cae_model)
from models.core.tf_models.cae_model import  Encoder, Decoder, CAE


# config = loadConfig('series000exp001')
config['exp_id'] = 'debug_experiment_2'

def train_debugger():
    model = CAE(config, model_use='training')
    data_objs = DataObj(config).loadData()

    t0 = time.time()
    for epoch in range(1):
        model.train_loop(data_objs[0:3])
        model.test_loop(data_objs[3:], epoch)
        print(epoch, 'epochs completed')

    print(time.time() - t0)


def train_exp(durations, exp_trains, exp_vals, config, exp_name):

    if exp_name in (exp_trains or exp_vals):
        raise  KeyError("Experiment already completed")

    train_loss = []
    valid_loss = []

    model = CAE(config, model_use='training')
    data_objs = DataObj(config).loadData()

    t0 = time.time()
    for epoch in range(5):
        t1 = time.time()
        model.dec_model.model_use = 'training'
        model.train_loop(data_objs[0:3])
        model.dec_model.model_use = 'validating'
        model.test_loop(data_objs[3:], epoch)
        train_loss.append(round(model.train_loss.result().numpy().item(), 2))
        valid_loss.append(round(model.test_loss.result().numpy().item(), 2))
        # modelEvaluate(model, validation_data, config)
        print(epoch, 'epochs completed')
        print('train_loss', train_loss[-1])
        print('valid_loss', valid_loss[-1])
        print(time.time() - t1)

    exp_trains[exp_name] = train_loss
    exp_vals[exp_name] = valid_loss
    durations[exp_name] = time.time() - t0


    return durations, exp_trains, exp_vals

# train_debugger()
durations, exp_trains, exp_vals = train_exp(durations, exp_trains,
                                        exp_vals, config, 'exp002')
# del exp_trains['exp003']
# del exp_vals['exp004']
# del exp_trains['exp004']


legend = [
            'truth',
            'all-sample',
            '0.7-tf-100c',
            '0.6-tf-100c',
        ]

# legend = [
#         'context[rnn]',
#         'context[rnn+linear]',
#         # 'multi-head 200unit - ts[both]',

#         ]

# %%
for item in exp_vals:
# for item in ['exp005', 'exp003']:
    plt.plot(exp_vals[item])
    # plt.plot(exp_trains[item], '--')

plt.grid()
plt.xticks(np.arange(10))

plt.legend(legend)
# %%
for item in exp_vals:
# for item in ['exp005', 'exp003']:
    # plt.plot(exp_vals[item])
    plt.plot(exp_trains[item], '--')

plt.grid()
plt.xticks(np.arange(10))

plt.legend(legend)
# %%
a = tf.constant([[1,2,3,4,5],[1,2,3,9,5]])
for i in tf.range(2):

print(tf.slice(a,[0,0],[2,1]))


print(tf.slice(a,[0,2],[2,2]))
tf.gather(a,[0,0])
# %%
for item in exp_trains:
    plt.plot(exp_trains[item])

plt.grid()
plt.legend(legend)

# %%
np.zeros([10,0,5])

# %%
conditions.shape
state_obs = tf.reshape(states[0], [1, 20, 10])
cond = tf.reshape(conditions[0], [1, 20, 3])
state_obs.shape
enc_state = enc_model(state_obs)
param_vec = dec_model([cond, enc_state])
utils.get_pdf_samples(samples_n=1, param_vec=param_vec, model_type='merge_policy')
targets[0]

# %%
"""
Recursive prediction
"""
y_feature_n = 3
t0 = time.time()
def decode_sequence(state_obs, condition, step_n):
    # Encode the input as state vectors.
    encoder_states_value = enc_model(state_obs)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    sequences = []
    for n in range(1):
        states_value = encoder_states_value
        decoded_seq = []
        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        cond_shape = [1, 1, 3]
        conditioning  = tf.reshape(condition[0, 0, :], cond_shape)
        for i in range(20):

            param_vec = dec_model([conditioning , states_value])
            # range(3, config['data_config']['pred_horizon'])(output_.stddev())
            output_ = utils.get_pdf_samples(samples_n=1, param_vec=param_vec, model_type='merge_policy')
            output_ = tf.reshape(output_, [2])
            decoded_seq.append(output_)
            # Update the target sequence (of length 1).

            if i != 19:
                cond_val  = condition[0, i+1, -1]
                conditioning  = tf.concat([output_, [cond_val]], 0)
                conditioning  = tf.reshape(conditioning, cond_shape)

            # Update states
            states_value = dec_model.state


        sequences.append(np.array(decoded_seq))
    sequences = np.array(sequences)

    return sequences

step_n = 20
sequences = decode_sequence(state_obs, cond, step_n)
# plt.plot(state_obs.flatten())
# plt.plot(range(9, 19), decoded_seq.flatten())
compute_time = time.time() - t0
compute_time
#
plt.plot(targets[0][:, 1], color='red')
for traj in sequences:
    plt.plot(traj[:, 1], color='grey')

# plt.plot(targets[0][:, 0], color='red')
# for traj in sequences:
#     plt.plot(traj[:, 0], color='grey')
# %%
