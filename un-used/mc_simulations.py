
"""
Code runs monte-carlo simulations.
It has functions for the following processes:
-   Forward simulations
-   Action predictions
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sys
import os
from collections import deque
from collections import OrderedDict
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from src import GRU_model as dm
from src import data_preprocessing as ldp
from importlib import reload
import pickle
import random
from tensorflow.python.ops import math_ops


col = ['a_long_c', 'a_long_p', 'bb_a', 'bb_id', 'bb_long', 'bb_v', 'bb_v_long',
       'ff_a', 'ff_bb_v', 'ff_id', 'ff_long', 'ff_v', 'ff_v_long', 'frm', 'id',
       'lc_bool', 'pc', 'scenario', 'v_lat_c', 'v_lat_p', 'v_long']

file_name = './' + 'data_files/200404archive' + '/val_set'
pickle_in = open(file_name,"rb")
val_set = pickle.load(pickle_in)
pickle_in.close()




# reload(dm)
# %%

# %%

""" defs
"""
# sample_col= ['ff_v','bb_v','ff_a','bb_a','v_long'] + ['pc','ff_long','bb_long']
# sample_col = ['ff_v_long','bb_v_long','v_long'] + ['pc','ff_long','bb_long']
# sample_col = ['ff_v','bb_v','ff_bb_v'] + ['pc','ff_long','bb_long']
# sample_col = ['ff_v','bb_v','ff_bb_v','v_long'] + ['pc','ff_long','bb_long']
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
    elif ff_long_ii > 50:
        ff_long_ii = 50

    if bb_long_ii < 0:
        bb_long_ii = 0
    elif bb_long_ii > 50:
        bb_long_ii = 50

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


"""plotting a single traj against x20 traces
"""
scenario = 'i101_2'
ego_id = 2128
df_id =  val_set.loc[(val_set['scenario'] == scenario) &
                    (val_set['id'] == ego_id)].reset_index(drop = True)

plt.plot(df_id['pc'])
plt.grid()

# %%

"""
Setting up tf graph
"""
input_size = (None, input_shape[1], input_shape[2])

initializers = tf.contrib.layers.xavier_initializer()
activations = tf.nn.relu
tfd = tfp.distributions
tf.reset_default_graph()
model = dm.Model(input_size, 2, initializers, activations, 4, num_layers=2,hidden_size=hidden_size)
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

# %%
reload(dm)

experiment_name = 'lag_seq_55_prob30_t0drop_withuninoise'

# experiment_name = 'exp_hist_3'
# epochs_instance = '_epo_24_.ckpt-4775'
# epochs_instance = '_epo_49_.ckpt-9550'

# epochs_instance = '_epo_29_.ckpt-4080'
epochs_instance = '49_-55350'

scalers_container = 'scalers_container'
log = './experiments/' + experiment_name + '/LOGS/' + epochs_instance
SEQ_LEN = 5 # number of past frames
hist_step = 5
hidden_size = 40
input_shape = (1,SEQ_LEN,9)
file_name = './experiments/' + scalers_container + '/sample_scaler'
pickle_in = open(file_name,"rb")
sample_scaler = pickle.load(pickle_in)
pickle_in.close()

# %%

"""
single df trajectory
"""
step=0.1
n = 1 # number of sampled actions
horizon = 70 # 0.1 seconds
# for scenario in val_setsets:
# df_id.isnull().sum(axis = 0)
feat_0, frm_cs, frm_ps, prev_set_0 = initial_fvector(df_id, hist_step, SEQ_LEN)
state_cs_0 = prev_set_0.loc[prev_set_0['frm'] == frm_cs][col].reset_index(drop = True)
true_f_val_set0 = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)
a_long, a_lat = action_predict(feat_0, n)
# for i in range(5):
state_df = pd.DataFrame()
state_cs_1 = nsp(a_long[0][0], a_lat[0][0], state_cs_0, true_f_val_set0, step=0.1)
feat_1, frm_cs, frm_ps, prev_set_1 = state_concatenation(state_cs_1, prev_set_0, hist_step, SEQ_LEN)

state_df = state_df.append(state_cs_0, ignore_index=True)
state_df = state_df.append(state_cs_1, ignore_index=True)
state_cs = state_cs_1.copy()
prev_set = prev_set_1.copy()
feat = feat_1.copy()

for i in range(horizon):
    true_f_val_set = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)
    a_long, a_lat = action_predict(feat, 1)
    state_cs = nsp(a_long[0][0], a_lat[0][0], state_cs, true_f_val_set, step=0.1)
    feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs, prev_set, hist_step, SEQ_LEN)
    state_df = state_df.append(state_cs, ignore_index=True)

#     plt.plot(state_df['v_long'])
# plt.plot(ego_df['v_long'].values)

    ################################################################

ego_df = df_id.loc[(df_id['frm'] >= state_df.iloc[0]['frm']) & (df_id['frm'] <= state_df.iloc[-1]['frm'])]
len(ego_df)
len(state_df)

test_col = [ 'pc', 'lc_bool', 'v_long', 'a_long_c', 'v_lat_c',
       'ff_v', 'bb_v','ff_bb_v', 'ff_long','bb_long']
for item in test_col:
    plt.figure()
    plt.plot(ego_df[item].values)
    plt.plot(state_df[item].values)
    plt.legend(['true','prediction'])
    plt.title(item)
    # plt.savefig(item, bbox_inches='tight', dpi = 200)
# plt.plot(df_id['lc_bool'].iloc[0:20])
#
# plt.plot(df_id['pc'].iloc[0:20])
# plt.plot(df_id['v_lat_c'].iloc[0:100])

# state_df.to_csv(r'state_df.txt', header=None, index=None, sep=' ', mode='a')
# ego_df.to_csv(r'ego_df.txt', header=None, index=None, sep=' ', mode='a')
# %%
plt.plot(prev_set_0['a_long_c'])
state_concat = []
n = 5
state_concat.append(n + round(np.random.normal(0,0.05,1)[0],3))
state_concat
# %%
""" qualitative results for the paper
trajectory traces for a single car
"""
state_col = ['a_long_c','v_long','v_lat_c']
step=0.1
n = 20 # number of sampled actions
horizon = 70 # 0.1 seconds
# for scenario in val_setsets:
feat_0, frm_cs, frm_ps, prev_set_0 = initial_fvector(df_id, hist_step, SEQ_LEN)
state_cs_0 = prev_set_0.loc[prev_set_0['frm'] == frm_cs][col].reset_index(drop = True)
true_f_val_set0 = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)

a_longs, a_lats = action_predict(feat_0, n)
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
        a_long, a_lat = action_predict(feat, 1)
        state_cs = nsp(a_long[0][0], a_lat[0][0], state_cs, true_f_val_set, step=0.1)
        feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs, prev_set, hist_step, SEQ_LEN)
        state_trace.append(state_cs[state_col].round(3).values.tolist()[0])

    state_traces.append(state_trace)
################################################################

# %%
img_path = './experiments/' + experiment_name + 'ego_df_speed_profile.txt'

ego_df.to_csv(img_path, header=None, index=None, sep=' ', mode='a')
pd.DataFrame(state_traces)
with open('state_traces.txt', 'w') as f:
    for item in state_traces:
        f.write("%s\n" % item)
# %%
""" plot traj distributions """
x = []
y = []
len(state_traces)
for trace in state_traces: # time stamp:
    traj_pred = traj_compute(trace, step=0.1)
    for state in traj_pred:
        x.append(state[0]/10)
        y.append(state[1])


x_other =  np.linspace(-2, 5, 3)
y_other =  np.linspace(-2, 1, 3)

for x_i in x_other:
    x.append(x_i)
for y_i in y_other:
    y.append(y_i)

# %%
ego_df = df_id.loc[(df_id['frm'] >= state_df.iloc[0]['frm'] - 20) & (df_id['frm'] <= state_df.iloc[-1]['frm'])]
traj_true = traj_compute(ego_df[state_col].values, step=0.1)

x_true = []
y_true = []

for state in traj_true:
    x_true.append(state[0]/10)
    y_true.append(state[1])

x_origin = x_true[20]
y_origin = y_true[20]
x_true = np.array(x_true) - x_origin
y_true = np.array(y_true) - y_origin

# fig =  plt.figure(figsize=(20,7))
# ax = fig.add_subplot()
# heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)

# %%

# %%

def plot_heatmap(x, y, smoothing):
    fig =  plt.figure(figsize=(20,7))
    ax = fig.add_subplot()
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=800)
    s = smoothing
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges.min(), xedges.max(), yedges.min(), yedges.max()]
    # extent = [xedges.min(), xedges.max(), yedges.min(), yedges.max()]

    img, extent = heatmap.T, extent
    ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
    return fig, ax
fig, ax = plot_heatmap(x, y, 30)
ax.set_xticklabels([])
ax.set_yticklabels([])


# %%


# ego_df = df_id.loc[(df_id['frm'] >= 7962) & (df_id['frm'] <= 8048)]
# ego_df = df_id.loc[(df_id['frm'] >= 7965) & (df_id['frm'] <= 8062)]

ax.plot(x_true[20:], y_true[20:], lw=5.5, color='black')
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
ax.scatter(x_true[0:20:5], y_true[0:20:5],s=80,color='orange')
# ax.legend(['True traj','Past observations'])
fig.savefig("traj_distribution.png", bbox_inches='tight', dpi = 200)

fig
# x_true[10:]
# %%
x_true[0:15]
x_true[0:15:2]
traj_pred[0]
# %%
# ego_df = df_id.loc[(df_id['frm'] >= 7982) & (df_id['frm'] <= 8063)]
# ego_df = df_id.loc[(df_id['frm'] >= 7986) & (df_id['frm'] <= 8067)]
# ego_df = df_id.loc[(df_id['frm'] >= 7994) & (df_id['frm'] <= 8065)]

traj_true = traj_compute(ego_df[state_col].values, step=0.1)
fig, ax = plt.subplots()
ax.plot(traj_true[:,0],traj_true[:,1], color='red', linewidth=3)
for trace in state_traces: # time stamp:
    traj_pred = traj_compute(trace, step=0.1)

    ax.plot(traj_pred[:,0],traj_pred[:,1], color='grey')
    plt.xlabel('x ($m$)')
    plt.ylabel('y ($m$)')
img_path = './experiments/' + experiment_name+ '/traj_traces'
plt.savefig(img_path, bbox_inches='tight', dpi = 200)


# %%
fig, ax = plt.subplots()
set = ego_df.sort_values(by=['frm'])['v_long']
t =  set.index/10
ax.plot(t, set, color='red', linewidth=3)
for trace in state_traces: # time stamp:
    feat_plot = []
    for state in trace:
        feat_plot.append(state[1])
    ax.plot(t[0:len(trace)], feat_plot, color='grey')
    plt.xlabel('Horizon (s)')
    plt.ylabel('V_long ($m/s$)')
img_path = './experiments/' + experiment_name + '/v_long_traces'
plt.savefig(img_path, bbox_inches='tight', dpi = 200)

# %%
fig, ax = plt.subplots()
set = ego_df.sort_values(by=['frm'])['v_lat_p']
t =  set.index/10
ax.plot(t, set, color='red', linewidth=3)
for trace in state_traces: # time stamp:
    feat_plot = []
    for state in trace:
        feat_plot.append(state[2])
    ax.plot(t[0:len(trace)], feat_plot, color='grey')
    plt.xlabel('Horizon (s)')
    plt.ylabel('V_lat ($m/s$)')
img_path = './experiments/' + experiment_name + '/v_lat_traces'
plt.savefig(img_path, bbox_inches='tight', dpi = 200)
# %%
""" ######### PAPER PLOT ######### """
""" 4s traj predictions """

def state_trace_pred (n, car_val_set, frm_start, frm_end):
    df_id = car_val_set.loc[(car_val_set['frm'] > frm_start) & (car_val_set['frm'] < frm_end)].reset_index(drop = True)
    state0 = df_id[col].iloc[0]
    true_f_val_set1 = df_id[col].iloc[1]
    horizon = int(frm_end - frm_start)-1
    action_set = action_predict(state0, scaler_std, scaler_dist,target_scaler, n)
    state_traces = []

    for a in action_set:
        state_trace = []
        feat_i = state0.copy()
        feat_ii = nsp(a[0], feat_i, true_f_val_set1, step=0.1)

        state1 = feat_ii[state_col].round(3).tolist()
        state_trace.append(state0[state_col].round(3).tolist())
        state_trace.append(state1)
        for i in range(2,horizon):
            true_f_val_set = df_id[col].iloc[i]
            feat_i = feat_ii
            a = action_predict(feat_i, scaler_std, scaler_dist,target_scaler, 1)
            feat_ii = nsp(a[0][0], feat_i, true_f_val_set, step=0.1)


            staten = feat_ii[state_col].round(3).tolist()
            state_trace.append(staten)

        state_traces.append(state_trace)

    return state_traces

def traj_compute(trace, x, y, step):
    """ given ego states (a_long, v_long, v_lat), computes
        ego car's trajectory, assuming constant acceleration """
    traj = []
    for state in trace:
        traj.append([x,y])
        along = state[0]
        vlong = state[1]
        vlat = state[2]
        x = round(x + step*vlong, 2)
        # x = round(x + step*vlong + 0.5*along*step**2, 2)

        y = round(y + step*vlat,2)
    return np.array(traj)

def get_frm (car_val_set, slot_size):
    slot = 0
    n = len(car_val_set)
    frm_pairs = []
    while slot < n-slot_size:
        pairs = []
        frm0 = car_val_set.iloc[slot]['frm']
        frm1 = car_val_set.iloc[slot+slot_size]['frm']
        pairs.append(frm0)
        pairs.append(frm1)

        frm_pairs.append(pairs)
        slot += slot_size
    return frm_pairs

# %%
frm_pairs = get_frm (df_id, 20)

state_traces_slots = []
for frm_pair in frm_pairs:
    state_traces = state_trace_pred (5, df_id, frm_pair[0], frm_pair[1])
    state_traces_slots.append(state_traces)




frm_pairs[0]

# %%

traj_true  = traj_compute(df_id[state_col].values, 0, 0, step=0.1)
fig, ax = plt.subplots()
x_true = traj_true[:,0]
y_true = traj_true[:,1]
ax.plot(x_true,y_true, color='red', linewidth=3)

x = 0
y = 0

slot_size = 20
slot = 0


for state_traces in state_traces_slots:
    x = x_true[slot]
    y = y_true[slot]

    for trace in state_traces: # time stamp:
        traj_pred = traj_compute(trace, x, y, step=0.1)
        ax.plot(traj_pred[:,0],traj_pred[:,1], color='grey')

    slot += slot_size

plt.xlabel('x ($m$)')
plt.ylabel('y ($m$)')

# %%

fig, ax = plt.subplots()
set = df_id.sort_values(by=['frm'])['pc']
t =  set.index/10
ax.plot(t, set, color='red', linewidth=3)
for trace in state_traces: # time stamp:
    feat_plot = []
    for state in trace:
        feat_plot.append(state[-1])
    ax.plot(t[0:len(trace)], feat_plot, color='grey')
    plt.xlabel('Horizon (s)')
    plt.ylabel('pc ($m$)')


# %%
fig, ax = plt.subplots()
set = df_id.sort_values(by=['frm'])['v_long']
t =  set.index/10
ax.plot(t, set, color='red', linewidth=3)
for trace in state_traces: # time stamp:
    feat_plot = []
    for state in trace:
        feat_plot.append(state[1])
    ax.plot(t[0:len(trace)], feat_plot, color='grey')
    plt.xlabel('Horizon (s)')
    plt.ylabel('V_long ($m/s$)')

# %%
trace[0]
fig, ax = plt.subplots()
set = df_id.sort_values(by=['frm'])['v_lat']
t =  set.index/10
ax.plot(t, set, color='red', linewidth=3)
for trace in state_traces: # time stamp:
    feat_plot = []
    for state in trace:
        feat_plot.append(state[2])
    ax.plot(t[0:len(trace)], feat_plot, color='grey')
    plt.xlabel('Horizon (s)')
    plt.ylabel('V_lat ($m/s$)')

# %%
for_plot = []
for step in range(horizon):
    step_val = []
    for trace in state_traces: # time stamp
        step_val.append(trace[step][1])

    mean = np.array(step_val).mean()
    for_plot.append(mean)
len(state_traces)
plt.plot(for_plot)
plt.xlabel('Horizon (s)')
plt.ylabel('V_avg ($m/s$)')



# %%
"""rwse for x - for one car
"""
traj_true =  traj_compute(ego_df[state_col].values, step=0.1)
pos_true = traj_true[:,0]

error_traces = []
for num in range(n):
    traj_pred = traj_compute(state_traces[num], step=0.1)[:,0]
    error_trace = []
    for step in range(len(traj_pred)):
        error_trace.append((traj_pred[step] - pos_true[step])**2)
    error_traces.append(error_trace)

rwse = []
for step in range(horizon):
    step_error_all = []
    for trace in error_traces:
        step_error_all.append(trace[step])
    rwse.append(np.sqrt(np.mean(step_error_all)))

t = []
for ti in range(horizon):
    t.append(0+ti/10)
plt.plot(t,rwse)
plt.ylabel('x_err (m)')
plt.xlabel('Horizon (s)')

# %%
"""rwse for y - for one car
"""
traj_true =  traj_compute(ego_df[state_col].values, step=0.1)
y_true = traj_true[:,1]

error_traces = []
for num in range(n):
    traj_pred = traj_compute(state_traces[num], step=0.1)[:,1]
    error_trace = []
    for step in range(len(traj_pred)):
        error_trace.append((traj_pred[step] - y_true[step])**2)
    error_traces.append(error_trace)

rwse = []
for step in range(horizon):
    step_error_all = []
    for trace in error_traces:
        step_error_all.append(trace[step])
    rwse.append(np.sqrt(np.mean(step_error_all)))

t = []
for ti in range(horizon):
    t.append(0+ti/10)
plt.plot(t,rwse)
plt.ylabel('y_err (m)')
plt.xlabel('Horizon (s)')
################################################
################################################
################################################
################################################

# %%

minmax_col = ['v_long','pc','ff_long','bb_long']
state_col = ['a_long_c','v_long','v_lat_c']
# check_predict = pd.DataFrame()
step=0.1
m = 50 # number of cars
n = 20 # number of sampled actions
count = 0
car_traces_pred = []
car_traces_true = []
horizon = 80
for scenario in datasets:
    if count > m:
        break
    df_scene =  val_set.loc[(val_set['scenario'] == scenario)]
    ids = df_scene['id'].unique()
    for id in ids:
        if count > m:
            break

        df_id = df_scene.loc[(df_scene['id'] == id)].reset_index(drop = True)


        feat_0, frm_cs, frm_ps, prev_set_0 = initial_fvector(df_id, hist_step, SEQ_LEN)
        state_cs_0 = prev_set_0.loc[prev_set_0['frm'] == frm_cs][col].reset_index(drop = True)
        true_f_val_set0 = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)


        action_set = action_predict(feat_0, n)
        car_traces_true.append(df_id.loc[(df_id['frm'] >= frm_cs)][state_col].round(3).values.tolist())
        state_traces = []

        for a in action_set:
            state_trace = []
            state_cs_1 = nsp(a[0], state_cs_0, true_f_val_set0, step=0.1)
            feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs_1, prev_set_0, hist_step, SEQ_LEN)

            state_trace.append(state_cs_0[state_col].round(3).values.tolist()[0])
            state_trace.append(state_cs_1[state_col].round(3).values.tolist()[0])
            state_cs = state_cs_1.copy()
            for i in range(horizon):
                true_f_val_set = df_id.loc[df_id['frm'] == frm_ps][col].reset_index(drop = True)

                if true_f_val_set.empty:
                    state_traces.append(state_trace)
                    break

                a = action_predict(feat, 1)
                state_cs = nsp(a[0][0], state_cs, true_f_val_set, step=0.1)
                feat, frm_cs, frm_ps, prev_set = state_concatenation(state_cs, prev_set, hist_step, SEQ_LEN)
                state_trace.append(state_cs[state_col].round(3).values.tolist()[0])

            state_traces.append(state_trace)
        car_traces_pred.append(state_traces)
        count += 1
        print(count)
# %%

# %%
# os.chdir('../')
os.getcwd()
os.chdir('./' + experiment_name)


# %%

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
plt.plot(t,rwse)
plt.ylabel('x_err (m)')
plt.xlabel('Horizon (s)')
#
x_err_df = pd.DataFrame(rwse, columns=None)
x_err_df.to_csv(r'x_err_df.txt', header=None, index=None, sep=' ', mode='a')
# %%
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
plt.plot(t,rwse)
plt.ylabel('y_err (m)')
plt.xlabel('Horizon (s)')
y_err_df = pd.DataFrame(rwse, columns=None)
y_err_df.to_csv(r'y_err_df.txt', header=None, index=None, sep=' ', mode='a')
# %%
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
plt.plot(t,rwse)
plt.ylabel('v_err (m)')
plt.xlabel('Horizon (s)')
#`
v_err_df = pd.DataFrame(rwse, columns=None)
v_err_df.to_csv(r'v_err_df.txt', header=None, index=None, sep=' ', mode='a')

# %%
error_traces = []
pos_traces_pred = []
pos_traces_true = []

for car in range(count):
    traj_true = car_traces_true[car]
    pos_true = []
    for step in traj_true:
        pos_true.append(step[1])
    pos_traces_true.append(pos_true)

    state_traces = car_traces_pred[car]
    car_vtraces_pred = []
    for num in range(n):
        traj_pred = state_traces[num]
        pos_pred = []
        for step in traj_pred:
            pos_pred.append(step[1])
        car_vtraces_pred.append(pos_pred)
    pos_traces_pred.append(car_vtraces_pred)

len(car_traces_pred)
# %%
error_traces = []
for car in range(count):
    pos_trace_true = pos_traces_true[car]
    for pos_trace_pred in pos_traces_pred:
        error_trace = []
        for step in range(len(pos_trace_pred)):
            error_trace.append((pos_trace_pred[step] - pos_trace_true[step])**2)
        error_traces.append(error_trace)




rwse = []
for step in range(horizon):
    step_error_all = []
    for trace in error_traces:
        step_error_all.append(trace[step])
    rwse.append(np.sqrt(np.mean(step_error_all)))

t = []
for ti in range(horizon):
    t.append(0+ti/10)
plt.plot(t,rwse)
plt.ylabel('v_err (m/s)')
plt.xlabel('Horizon (s)')
