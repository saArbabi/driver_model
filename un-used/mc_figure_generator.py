
"""
Code runs monte-carlo simulations.
It has functions for the following processes:
-   Forward simulations
-   Action predictions
"""

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from importlib import reload
from collections import deque

from src import GRU_model as dm
from src import data_preprocessing as ldp
import os
from importlib import reload
import pickle

from tensorflow.python.ops import math_ops


datasets = {
        "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
        "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
        "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
        "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
        "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
        "i80_3": "trajdata_i80_trajectories-0515-0530.txt"}

col = ['a_long_c', 'a_long_p', 'bb_a', 'bb_id', 'bb_long', 'bb_v', 'bb_v_long',
       'ff_a', 'ff_bb_v', 'ff_id', 'ff_long', 'ff_v', 'ff_v_long', 'frm', 'id',
       'lc_bool', 'pc', 'scenario', 'v_lat_c', 'v_lat_p', 'v_long']

file_name = './' + 'data_files/' + '/val_set'
pickle_in = open(file_name,"rb")
val_set = pickle.load(pickle_in)
pickle_in.close()

file_name = './experiments/' + 'scalers_container' + '/sample_scaler'
pickle_in = open(file_name,"rb")
sample_scaler = pickle.load(pickle_in)
pickle_in.close()
val_set = val_set.loc[~((val_set['scenario'] == 'i80_2') & (val_set['id'] == 1869))] # this car has missing frms!




# %%
""" check the model's input """
reload(ldp)

exp_drop= 100
dropouts = (exp_drop)*[0] + (100-exp_drop)*[1]
pred_step = 1 # number of timesteps into the future to be predicted by model
hist_step = 1 # spacing between each frame
SEQ_LEN = 5 # number of past frames
hist = 5
val_prepped = ldp.data_prep(val_set, hist, SEQ_LEN)
x_val, y_val = ldp.data_split(val_prepped, dropouts)
sample_col = ['ff_v','bb_v','ff_a','bb_a'] + ['v_long','pc','ff_long','bb_long']
cols_all = sample_col + ['lc_bool']

len(x_val[5][0])
print(cols_all)
print(x_val[2001])

print(x_val[300])
print(y_val[5])

# %%

""" defs
"""
sample_col = ['ff_v','bb_v','ff_a','bb_a','v_long'] + ['pc','ff_long','bb_long']

def scale(data):
    data = data.reset_index(drop = True)
    ############################
    sample_std = sample_scaler.transform(data[sample_col].values)
    ############################
    scaled_data = pd.concat([pd.DataFrame(sample_std).round(3),
                                        data['lc_bool'].astype('int')], axis=1).values

    return scaled_data

def state_concatenation(new_state, prev_set, hist_step, SEQ_LEN):
    car_df = state_set_update(new_state, prev_set)
    prev_set = car_df.copy()

    frm_cs = car_df['frm'].iloc[-1]  #frm being predicted
    frm_pred = frm_cs + 1
    car_df = scale(car_df[sample_col + ['lc_bool']])
    sequential_data = []
    i = 0
    prev_states = deque(maxlen=SEQ_LEN)
    while i < len(car_df):
        row = car_df[i]
        prev_states.append(row)
        if len(prev_states) == SEQ_LEN:
            sequential_data.append([np.array(prev_states)])
        i += hist_step


    x = sequential_data[0][0]
    x_mod = x_modification(x)
    return np.array(x_mod).reshape(input_shape), int(frm_cs), int(frm_pred), prev_set

def initial_fvector(car_df, hist_step, SEQ_LEN):
    size = hist_step*(SEQ_LEN-1) + 1
    car_df = car_df.iloc[0:size]
    prev_set = car_df.copy()

    frm_cs = car_df['frm'].iloc[-1]  #frm being predicted
    frm_pred = frm_cs + 1
    car_df = scale(car_df[sample_col + ['lc_bool']])
    sequential_data = []
    i = 0
    prev_states = deque(maxlen=SEQ_LEN)

    while i < len(car_df):
        row = car_df[i]
        prev_states.append(row)
        if len(prev_states) == SEQ_LEN:
            sequential_data.append([np.array(prev_states)])
        i += hist_step

    x = sequential_data[0][0]
    x_mod = x_modification(x)

    return np.array(x_mod).reshape(input_shape), int(frm_cs), int(frm_pred), prev_set


def state_set_update(new_state, prev_set):
    return prev_set.drop(0).append(new_state, ignore_index = True)


def inv_trans_y(values):
    """returns the unscaled action values
    """
    a_long = []
    a_lat = []

    for item in values:
        a_long.append([item[0][0]])
        a_lat.append([item[0][1]])
    return a_long, a_lat
# a_longs, a_lats = action_predict(feat_0, 3)

def action_predict(x_feat, n):
    parameters = sess1.run(model.output, feed_dict={model.input_features: x_feat})
    action_samples = inv_trans_y(sess2.run(samples, feed_dict={param:parameters, n_samples:n}))

    return action_samples
#
# a_longs, a_lats = action_predict(feat_0, n)
# parameters = sess1.run(model.output, feed_dict={model.input_features: feat_0})
# action_samples = inv_trans_y(sess2.run(samples, feed_dict={param:parameters, n_samples:2}))
# action_val = sess2.run(samples, feed_dict={param:parameters, n_samples:3})

def traj_compute(trace, step):
    """ given ego states (a_long, v_long, v_lat), computes
    ego car's trajectory, assuming constant acceleration """
    traj = []
    x = 0
    y = 0
    for state in trace:
        traj.append([x,y])
        along = state[0]
        vlong = state[1]
        vlat = state[2]
        # x = round(x + step*vlong, 2)
        x = round(x + step*vlong + 0.5*along*step**2, 2)
        y = round(y + step*vlat,2)
    return np.array(traj)
#
# def x_modification(x):
#     x_mod = []
#     x = x.tolist()
#     step_i = 0
#     for time_step in x:
#         if step_i != 4: # if not current state
#             n_i = 0
#             state_concat = []
#             for n in time_step:
#                 if n_i == 4: #v_long
#                     state_concat.append(0)
#                 else:
#                     state_concat.append(n)
#                 n_i += 1
#             x_mod.append(state_concat)
#         else:
#             x_mod.append(time_step)
#
#         step_i += 1
#     return np.array(x_mod)
def x_modification(x):
    return x


def nsp (a_long_ii,v_lat_ii, statei, frm_ps_val_set, step=0.1):
    """ Next State Predictor
        takes in current step stateure and computes next step stateure.
        statei: stateure now
        stateii: stateure next
    """

    # stateii = statei.copy()
    stateii = statei.copy()

    stateii[['a_long_c','v_lat_c']] = a_long_ii, v_lat_ii

    v_long_i = statei['v_long'][0]
    v_long_ii = v_long_i + a_long_ii*step


    ff_long_ii = statei['ff_long'][0] + statei['ff_v'][0]*step
    bb_long_ii = statei['bb_long'][0] + statei['bb_v'][0]*step

    if ff_long_ii < 0:
        ff_long_ii = 0
    elif ff_long_ii > 70:
        ff_long_ii = 70

    if bb_long_ii < 0:
        bb_long_ii = 0
    elif bb_long_ii > 70:
        bb_long_ii = 70

    pc_ii = statei['pc'][0] + v_lat_ii*step
    ff_v_ii = frm_ps_val_set['ff_v_long'] - v_long_ii
    bb_v_ii = v_long_ii - frm_ps_val_set['bb_v_long']
    stateii['ff_v'] = ff_v_ii
    stateii['bb_v'] = bb_v_ii
    stateii['ff_bb_v'] = frm_ps_val_set['ff_bb_v']

    stateii['lc_bool'] = frm_ps_val_set['lc_bool']
    stateii['frm'] = frm_ps_val_set['frm']
    stateii[['ff_v_long','bb_v_long']] = frm_ps_val_set[['ff_v_long','bb_v_long']]
    stateii[['ff_a','bb_a']] = frm_ps_val_set[['ff_a','bb_a']]

    stateii[['a_long_p','v_lat_p']] = a_long_ii, v_lat_ii

    stateii['v_long'] = v_long_ii
    stateii['ff_long'] = ff_long_ii
    stateii['bb_long'] = bb_long_ii
    stateii['pc'] = pc_ii

    return stateii

# %%

# %%
"""
Setting up tf graph
"""
# experiment_names = [
# 'lag_seq_35',
# 'lag_seq_45',
# 'lag_seq_53',
# 'lag_seq_54'
# ]
#
#
# epochs_instances = [
# '49_-64400',
# '49_-59850',
# '49_-66650',
# '49_-61000'
# ]
#

# ]
experiment_names = [
'lag_seq_55_prob30_t0drop'
]


epochs_instances = [
'49_-55350'
]
state_col = ['a_long_c','v_long','v_lat_c']
n_exps = len(experiment_names)
for exp_n in range(n_exps):

    epochs_instance = epochs_instances[exp_n]
    experiment_name = experiment_names[exp_n]

    os.mkdir('./experiments/' + experiment_name + '/plt_data')

    scalers_container = 'scalers_container'
    log = './experiments/' + experiment_name + '/LOGS/' + epochs_instance
    # SEQ_LEN =  int(experiment_name[-1]) # number of past frames
    # hist_step = int(experiment_name[-2])
    SEQ_LEN =  5
    hist_step = 5
    input_shape = (1,SEQ_LEN,9)

    """
    Setting up tf graph
    """
    input_size = (None, input_shape[1], input_shape[2])

    initializers = tf.contrib.layers.xavier_initializer()
    activations = tf.nn.relu
    tfd = tfp.distributions
    tf.reset_default_graph()
    model = dm.Model(input_size, 2, initializers, activations, 4, num_layers=2,hidden_size=40)
    saver = tf.train.Saver()
    graph1 = tf.get_default_graph()
    graph2 = tf.Graph()
    sess1 = tf.Session(graph=graph1)
    sess2 = tf.Session(graph=graph2)
    saver.restore(sess1, log)

    with graph2.as_default():
        param = tf.placeholder(tf.float64, shape= None)
        n_samples = tf.placeholder(tf.int32, shape= None)
        pi, sigma_lat, sigma_long, mu_lat, mu_long,rho  = tf.split(
                param, num_or_size_splits= 6, axis=1)
        pi = tf.nn.softmax(tf.keras.activations.linear(pi))
        rho = tf.nn.tanh(rho)

        # pi = tf.nn.softmax(tf.nn.relu(pi)
        sigma_lat =  math_ops.exp(sigma_lat)
        sigma_long = math_ops.exp(sigma_long)
        mu = tf.stack([mu_lat, mu_long], axis=2, name='mu')


        sig_lat_squared = tf.math.square(sigma_lat)
        sig_long_squared = tf.math.square(sigma_long)
        cor0 = tf.math.multiply(sigma_lat,sigma_long)
        cor1 = tf.math.multiply(cor0,rho)

        mat1 = tf.stack([sig_lat_squared, cor1], axis=2)
        mat2 = tf.stack([cor1, sig_long_squared], axis=2)
        cov = tf.stack([mat1, mat2], axis=2, name='cov')

        mvn = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=pi),
        components_distribution=tfd.MultivariateNormalFullCovariance(
            loc=mu,
            covariance_matrix=cov[0]))

        samples = mvn.sample(n_samples)
        print('graph setup successfully')

    #
    # #############################################################################
    # #############################################################################
    # #############################################################################
    # """
    # trajectory traces for a single car
    # """
    # scenario = 'i101_2'
    # ego_id = 2128
    # df_id =  val_set.loc[(val_set['scenario'] == scenario) &
    #                     (val_set['id'] == ego_id)].reset_index(drop = True)
    #
    # """ qualitative results for the paper
    # trajectory traces for a single car
    # """
    # state_col = ['a_long_c','v_long','v_lat_c']
    # step=0.1
    # n = 20 # number of sampled actions
    # horizon = 70 # 0.1 seconds
    # # for scenario in val_setsets:
    # feat_0, frm_cs, frm_ps, prev_set_0 = initial_fvector(df_id, hist_step, SEQ_LEN)
    # state_cs_0 = prev_set_0.loc[prev_set_0['frm'] == frm_cs][col].reset_index(drop = True)
    # true_f_val_set0 = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)
    #
    # a_longs, a_lats = action_predict(feat_0, n)
    # state_traces = []
    #
    # for a_count in range(n):
    #
    #     state_trace = []
    #     state_cs_1 = nsp(a_longs[a_count][0], a_lats[a_count][0], state_cs_0, true_f_val_set0, step=0.1)
    #     feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs_1, prev_set_0, hist_step, SEQ_LEN)
    #
    #     state_trace.append(state_cs_0[state_col].round(3).values.tolist()[0])
    #     state_trace.append(state_cs_1[state_col].round(3).values.tolist()[0])
    #     state_cs = state_cs_1.copy()
    #     for i in range(horizon):
    #         true_f_val_set = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)
    #         a_long, a_lat = action_predict(feat, 1)
    #         state_cs = nsp(a_long[0][0], a_lat[0][0], state_cs, true_f_val_set, step=0.1)
    #         feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs, prev_set, hist_step, SEQ_LEN)
    #         state_trace.append(state_cs[state_col].round(3).values.tolist()[0])
    #
    #     state_traces.append(state_trace)
    #
    #
    # txt_name = './experiments/' + experiment_name + '/plt_data/state_traces.txt'
    # with open(txt_name, 'w') as f:
    #     for item in state_traces:
    #         f.write("%s\n" % item)
    #
    # print(experiment_name + ' completed and saved')


    ###################################

    # check_predict = pd.DataFrame()
    step=0.1
    m = 50 # number of cars
    n = 20 # number of sampled actions
    count = 0
    car_traces_pred = []
    car_traces_true = []
    horizon = 70
    for scenario in datasets:
        if count > m:
            break
        df_scene =  val_set.loc[(val_set['scenario'] == scenario)]
        ids = df_scene['id'].unique()
        for id in ids:
            if count > m:
                break

            frm_start = df_scene.loc[(df_scene['id'] == id) & (df_scene['lc_bool'] == 1)]['frm'].iloc[0] - 20
            df_id = df_scene.loc[(df_scene['id'] == id) & (df_scene['frm'] >= frm_start)].reset_index(drop = True)
            if len (df_id) > 93:

                feat_0, frm_cs, frm_ps, prev_set_0 = initial_fvector(df_id, hist_step, SEQ_LEN)
                state_cs_0 = prev_set_0.loc[prev_set_0['frm'] == frm_cs][col].reset_index(drop = True)
                true_f_val_set0 = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)


                a_longs, a_lats = action_predict(feat_0, n)
                car_traces_true.append(df_id.loc[(df_id['frm'] >= frm_cs)][state_col].round(3).values.tolist())
                state_traces = []


                for a_count in range(n):
                    state_trace = []
                    state_cs_1 = nsp(a_longs[a_count][0], a_lats[a_count][0], state_cs_0, true_f_val_set0, step=0.1)
                    feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs_1, prev_set_0, hist_step, SEQ_LEN)

                    state_trace.append(state_cs_0[state_col].round(3).values.tolist()[0])
                    state_trace.append(state_cs_1[state_col].round(3).values.tolist()[0])
                    state_cs = state_cs_1.copy()
                    for i in range(horizon):
                        true_f_val_set = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)

                        if true_f_val_set.empty:
                            state_traces.append(state_trace)
                            break

                        a_long, a_lat = action_predict(feat, 1)
                        state_cs = nsp(a_long[0][0], a_lat[0][0], state_cs, true_f_val_set, step=0.1)
                        feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs, prev_set, hist_step, SEQ_LEN)
                        state_trace.append(state_cs[state_col].round(3).values.tolist()[0])

                    state_traces.append(state_trace)
                car_traces_pred.append(state_traces)
                count += 1
                print(count)



    """rwse for x
    """
    error_traces = []

    for car in range(count):
        traj_true =  traj_compute(car_traces_true[car][0:horizon], step=0.1)
        pos_true = traj_true[:,0]
        state_traces = car_traces_pred[car]

        for num in range(n):
            traj_pred = traj_compute(state_traces[num], step=0.1)[:,0]
            error_trace = []
            for step in range(len(traj_pred)):
                if step < len(pos_true):
                    error_trace.append((traj_pred[step] - pos_true[step])**2)
            error_traces.append(error_trace)

    rwse = []
    for step in range(horizon):
        step_error_all = []
        for trace in error_traces:
            if step < len(trace):
                step_error_all.append(trace[step])
        rwse.append(np.sqrt(np.mean(step_error_all)))

    t = []
    for ti in range(horizon):
        t.append(0+ti/10)

    #
    x_err_df = pd.DataFrame(rwse, columns=None)
    txt_name = './experiments/' + experiment_name + '/plt_data/x_err_df.txt'
    x_err_df.to_csv(txt_name, header=None, index=None, sep=' ', mode='a')


    """rwse for y
    """
    error_traces = []

    for car in range(count):
        traj_true =  traj_compute(car_traces_true[car][0:horizon], step=0.1)
        pos_true = traj_true[:,1]
        state_traces = car_traces_pred[car]

        for num in range(n):
            traj_pred = traj_compute(state_traces[num], step=0.1)[:,1]
            error_trace = []
            for step in range(len(traj_pred)):
                if step < len(pos_true):
                    error_trace.append((traj_pred[step] - pos_true[step])**2)
            error_traces.append(error_trace)


    # for tt in error_traces:
    #     plt.plot(tt)

    rwse = []

    for step in range(horizon):
        step_error_all = []
        for trace in error_traces:
            if step < len(trace):
                step_error_all.append(trace[step])
        rwse.append(np.sqrt(np.mean(step_error_all)))

    t = []
    for ti in range(horizon):
        t.append(0+ti/10)

    y_err_df = pd.DataFrame(rwse, columns=None)
    txt_name = './experiments/' + experiment_name + '/plt_data/y_err_df.txt'
    y_err_df.to_csv(txt_name, header=None, index=None, sep=' ', mode='a')


    """rwse for v
    """
    def v_compute(trace):
        pos = []
        for state in trace:
            pos.append(state[1])
        return pos

    error_traces = []
    for car in range(count):
        traj_true =  v_compute(car_traces_true[car])
        state_traces = car_traces_pred[car]

        for num in range(n):
            traj_pred = v_compute(state_traces[num])
            error_trace = []
            for step in range(len(traj_pred)):
                if step < len(pos_true):
                    error_trace.append((traj_pred[step] - traj_true[step])**2)
            error_traces.append(error_trace)

    #
    # for tt in error_traces:
    #     plt.plot(tt)
    rwse = []

    for step in range(horizon):
        step_error_all = []
        for trace in error_traces:
            if step < len(trace):
                step_error_all.append(trace[step])
        rwse.append(np.sqrt(np.mean(step_error_all)))

    t = []
    for ti in range(horizon):
        t.append(0+ti/10)


    v_err_df = pd.DataFrame(rwse, columns=None)
    txt_name = './experiments/' + experiment_name + '/plt_data/v_err_df.txt'
    v_err_df.to_csv(txt_name, header=None, index=None, sep=' ', mode='a')
