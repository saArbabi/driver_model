"""
- Outputs training set and validation set for a particular experiment prior to
start of training.
"""
from collections import deque
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import time
# %%
def read_episode_df():
    global m_df0, y_df0

    mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
           'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']

    yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

    m_df0 = pd.read_csv('./datasets/m_df0.txt', delimiter=' ',
                            header=None, names=mveh_col)
    y_df0 = pd.read_csv('./datasets/y_df0.txt', delimiter=' ',
                            header=None, names=yveh_col)

def read_episode_ids():
    global episode_ids

    episode_ids = {}
    for name in ['training_episodes', 'validation_episodes']:
        file_name = './datasets/'+name+'.txt'
        with open(file_name, "r") as file:
            my_list = [int(item) for item in file.read().split()]
        episode_ids[name] = my_list

read_episode_df()
read_episode_ids()

# %%
def get_timeStamps(size):
    ts = np.zeros([size, 1])
    t = 0.1
    for i in range(1, size):
        ts[i] = t
        t += 0.1
    return ts

ts = get_timeStamps(5)
a = np.ones([5, 1])
np.reshape(ts, [1, 5])
ts
ts.shape = (5, 1)

ts
np.array(ts, shape=[1, 5])
a
ts


np.linspace(0, 50, endpoint=False, retstep=0.1)
fix = np.linspace(0, 100, 100, endpoint=False)
var = np.linspace(0, 100, 100, endpoint=False)
var.shape = (-1, 1)
var
np.round(, decimals=1)
ts = range()
40000*30
0.04*30
# %%
episodes = []
data_size = 0
for step in range(len(fix)):
    epis_i = var[step:]
    epis_size = len(epis_i)
    ts = get_timeStamps(epis_size)
    epis_i = np.insert(epis_i, 1, fix[step], axis=1)

    episodes.append(np.concatenate([epis_i, ts], axis=1))
    data_size += len(epis_i)


data_size/len(fix)

# %%
class DataObj():
    random.seed(2020)

    def __init__(self, config):
        self.config = config['data_config']
        self.model_type = config['model_type']
        self.sequence_n = self.config["sequence_n"]
        self.step_size = self.config["step_size"]

        self.mveh_s = self.config["mveh_s"]
        self.yveh_s = self.config["yveh_s"]

        self.Xs = []
        self.Ys = []
        self.setState_indx()
        self.setScalers() # will set the scaler attributes


    def sequence(self, x_arr, y_arr):
        x_seq = [] # sequenced x array
        y_seq = []

        if self.sequence_n != 1:
            i_reset = 0
            i = 0
            for chunks in range(self.step_size):
                prev_states = deque(maxlen=self.sequence_n)
                while i < len(x_arr):
                    prev_states.append(x_arr[i])
                    if len(prev_states) == self.sequence_n:
                        x_seq.append(np.array(prev_states))
                        y_seq.append(y_arr[i])
                    i += self.step_size
                i_reset += 1
                i = i_reset
        else:
            for i in range(len(x_arr)):
                x_seq.append(x_arr[i])
                y_seq.append(y_arr[i])

        return x_seq, y_seq

    def shuffArr(self, arr):
        random.Random(2020).shuffle(arr)
        return np.array(arr)


    def mask_history(self, x_arr):
        pass
        # dropout_percentage = self.config['mask_history']['percentage']
        # if  dropout_percentage != 0:
        #     target_name = self.config['mask_history']['vehicle']
        #     if target_name == 'mveh':
        #         index = mveh.sample(int(len(mveh)*dropout_percentage)).index
        #         mveh.loc[:, index, mveh.columns != 'lc_type']=0


    def get_episode_df(self, episode_id):
        m_df = m_df0[m_df0['episode_id'] == episode_id]
        y_df = y_df0[y_df0['episode_id'] == episode_id]

        return m_df, y_df

    def applystateScaler(self, _arr):
        _arr = np.delete(_arr, self.retain_pointer, axis=1)
        _arr[:, self.bool_pointer:] = self.state_scaler.transform(_arr[:, self.bool_pointer:])

        return _arr

    def applytargetScaler(self, _arr):
        _arr = self.target_scaler.transform(_arr)
        return _arr

    def applyInvScaler(self, action_arr):
        """
        Note: only applies to target (i.e. action) values
        """
        return self.target_scaler.inverse_transform(action_arr)

    def setState_indx(self):
        i = 0
        self.retain_indx = {}
        self.bool_indx = {}
        self.scalar_indx = {}


        self.retain_indx['vel_mveh'] = i
        i += 1
        self.retain_indx['vel_yveh'] = i
        i += 1

        for state_key in self.mveh_s['boolean']:
            self.bool_indx[state_key+'_mveh'] = i
            i += 1

        for state_key in self.yveh_s['boolean']:
            self.bool_indx[state_key+'_yveh'] = i
            i += 1

        for state_key in self.mveh_s['scalar']:
            self.scalar_indx[state_key+'_mveh'] = i
            i += 1

        for state_key in self.yveh_s['scalar']:
            self.scalar_indx[state_key+'_yveh'] = i
            i += 1

        # these are used by the scaler
        self.bool_pointer = list(self.bool_indx.values())[-1]
        self.retain_pointer = list(self.retain_indx.values())

    def get_stateTarget_arr(self, m_df, y_df):
        """Note: Not all states are used by model for prediction. Some are needed
            for state propagation.
        """
        self.addFeatures(m_df, y_df)

        if self.model_type == 'merge_policy':
            target_df = m_df[['act_long','act_lat']]
        state_df = pd.DataFrame()

        if self.config['retain']:
            state_df = pd.concat([state_df, m_df[self.config['retain']]], axis=1)
            state_df = pd.concat([state_df, y_df[self.config['retain']]], axis=1)

        if self.mveh_s['boolean']:
            state_df = pd.concat([state_df, m_df[self.mveh_s['boolean']]], axis=1)

        if self.yveh_s['boolean']:
            state_df = pd.concat([state_df, y_df[self.yveh_s['boolean']]], axis=1)

        if self.mveh_s['scalar']:
            state_df = pd.concat([state_df, m_df[self.mveh_s['scalar']]], axis=1)

        if self.yveh_s['scalar']:
            state_df = pd.concat([state_df, y_df[self.yveh_s['scalar']]], axis=1)

        return state_df.values, target_df.values

    def setScalers(self):
        state_arr, target_arr = self.get_stateTarget_arr(m_df0, y_df0)
        state_arr = np.delete(state_arr, self.retain_pointer, axis=1)
        self.state_scaler = StandardScaler().fit(state_arr[:, self.bool_pointer:])
        self.target_scaler = StandardScaler().fit(target_arr)

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        m_df, y_df = self.get_episode_df(episode_id)
        x_arr, y_arr = self.get_stateTarget_arr(m_df, y_df)
        x_arr = self.applystateScaler(x_arr)
        y_arr = self.applytargetScaler(y_arr)
        x_arr, y_arr = self.sequence(x_arr, y_arr)

        # self.mask_history(x_df)
        for i in range(len(x_arr)):
            self.Xs.append(x_arr[i])
            self.Ys.append(y_arr[i])

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")

        for episode_id in episode_ids[episode_type]:
            self.episode_prep(episode_id)

        return self.shuffArr(self.Xs), self.shuffArr(self.Ys)


Data = DataObj(config)
# x_train, y_train = Data.data_prep('training_episodes')
x_val, y_val = Data.data_prep('validation_episodes')
# m_df, y_df = Data.get_episode_df(811)
# x_arr, y_arr = Data.get_stateTarget_arr(m_df, y_df)

# %%
config = {
 "model_config": {
     "learning_rate": 1e-2,
     "neurons_n": 50,
     "layers_n": 2,
     "epochs_n": 5,
     "batch_n": 128,
     "components_n": 5
},
"data_config": {"step_size": 3,
                "sequence_n": 1,
                "mveh_s":{"scalar":["vel", "pc", "gap_size", "dx", "act_long_p"],
                            "boolean":["lc_type"]},
                "yveh_s":{"scalar":["vel"],
                            "boolean":[]},
                "retain":["vel"]
},
"exp_id": "NA",
"model_type": "merge_policy",
"Note": "NA"
}

# %%
x_train[1]

x_train[0]
y_train[0]
import matplotlib.pyplot as plt
len(training_episodes)
len(x_train[0])
np.array(x_val)
len()
# %%
def vis_beforeAfterScale(x, features):
    i = 0
    # x = Data.state_scaler.inverse_transform(x[:,1:])
    for feature in features:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax1.hist(m_df0[feature], bins=125)
        # ax2.hist(x[:,i], bins=125)
        ax2.hist(x[:,i], bins=125)

        i += 1
        ax1.set_title(feature + ' before')
        ax2.set_title(feature + ' after')

vis_beforeAfterScale(x_val, Data.veh_states['mveh'])
