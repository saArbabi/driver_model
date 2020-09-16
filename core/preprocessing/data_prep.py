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
import matplotlib.pyplot as plt

# %%
def read_episode_df():
    global m_df0, y_df0

    m_col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type', 'act_long_p',
                                                'act_lat_p', 'act_long', 'act_lat']

    y_col = ['episode_id', 'frm','vel', 'dv', 'dx', 'da', 'a_ratio', 'act_long_p', 'act_long']

    m_df0 = pd.read_csv('./datasets/m_df0.txt', delimiter=' ',
                            header=None, names=m_col)
    y_df0 = pd.read_csv('./datasets/y_df0.txt', delimiter=' ',
                            header=None, names=y_col)

def read_episode_ids():
    global episode_ids

    episode_ids = {}
    for name in ['training_episodes', 'validation_episodes']:
        file_name = './datasets/'+name+'.txt'
        with open(file_name, "r") as file:
            my_list = [int(item) for item in file.read().split()]
        episode_ids[name] = my_list

def read_fixed_stateArr():
    global fixed_state_arr
    fixed_state_arr = pd.read_csv('./datasets/fixed_df0.txt', delimiter=' ',
                                                            header=None).values
    # First two columns are lc_type and episode_id
    fixed_state_arr[:,2:] = StandardScaler().fit(fixed_state_arr[:,2:]).transform(fixed_state_arr[:,2:])


read_episode_df()
read_episode_ids()
read_fixed_stateArr()


# %%
class DataObj():
    random.seed(2020)

    def __init__(self, config):
        self.config = config['data_config']
        self.model_type = config['model_type']
        self.obsSequence_n = self.config["obsSequence_n"]
        self.step_size = self.config["step_size"]

        self.m_s = self.config["m_s"]
        self.y_s = self.config["y_s"]

        self.Xs = []
        self.Ys = []
        self.setState_indx()
        self.setScalers() # will set the scaler attributes


    def obsSequence(self, v_x_arr):
        x_seq = [] # obsSequenced x array
        if self.obsSequence_n != 1:
            i_reset = 0
            i = 0
            for chunks in range(self.step_size):
                prev_states = deque(maxlen=self.obsSequence_n)
                while i < len(v_x_arr):
                    prev_states.append(v_x_arr[i])
                    if len(prev_states) == self.obsSequence_n:
                        x_seq.append(np.array(prev_states))
                    i += self.step_size
                i_reset += 1
                i = i_reset
        else:
            for i in range(len(v_x_arr)):
                x_seq.append(v_x_arr[i])

        return x_seq

    def shuffArr(self, arr):
        random.Random(2020).shuffle(arr)
        return np.array(arr)

    def mask_history(self, v_x_arr):
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
        return self.state_scaler.transform(_arr)

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

        for state_key in self.m_s:
            self.scalar_indx[state_key+'_mveh'] = i
            i += 1

        for state_key in self.y_s:
            self.scalar_indx[state_key+'_yveh'] = i
            i += 1

        # these are used by the scaler
        self.retain_pointer = list(self.retain_indx.values())

    def get_stateTarget_arr(self, m_df, y_df):
        """Note: Not all states are used by model for prediction. Some are needed
            for state propagation.
        """
        if self.model_type == 'merge_policy':
            target_df = m_df[['act_long','act_lat']]
        state_df = pd.DataFrame()

        if self.config['retain']:
            state_df = pd.concat([state_df, m_df[self.config['retain']]], axis=1)
            state_df = pd.concat([state_df, y_df[self.config['retain']]], axis=1)

        if self.m_s:
            state_df = pd.concat([state_df, m_df[self.m_s]], axis=1)

        if self.y_s:
            state_df = pd.concat([state_df, y_df[self.y_s]], axis=1)

        return state_df.values, target_df.values

    def setScalers(self):
        state_arr, target_arr = self.get_stateTarget_arr(m_df0, y_df0)
        state_arr = np.delete(state_arr, self.retain_pointer, axis=1)
        self.state_scaler = StandardScaler().fit(state_arr)
        self.target_scaler = StandardScaler().fit(target_arr)

    def get_fixedSate(self, episode_id):
        state_arr = fixed_state_arr[fixed_state_arr[:,1]==episode_id]
        return np.delete(state_arr, 1, axis=1)

    def get_timeStamps(self, size):
        ts = np.zeros([size, 1])
        t = 0.1
        for i in range(1, size):
            ts[i] = t
            t += 0.1
        return ts

    def get_vfArrs(self, v_x_arr, v_y_arr, f_x_arr):
        """
        Note: Output will be orders of magnitude larger in size.
        :Return: state arrays with time stamps and fixed features included
        """
        episode_len = len(v_x_arr)
        mini_episodes_x = []
        mini_episodes_y = []
        for step in range(episode_len):
            epis_i = v_x_arr[step:]
            target_i = v_y_arr[step:]

            episode_i_len = len(epis_i)
            ts = self.get_timeStamps(episode_i_len)
            epis_i = np.insert(epis_i, [0], f_x_arr[step], axis=1)
            mini_episodes_x.append(np.concatenate([ts, epis_i], axis=1))
            mini_episodes_y.append(target_i)

        return mini_episodes_x, mini_episodes_y

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        m_df, y_df = self.get_episode_df(episode_id)
        v_x_arr, v_y_arr = self.get_stateTarget_arr(m_df, y_df)
        v_x_arr = self.applystateScaler(v_x_arr)
        v_y_arr = self.applytargetScaler(v_y_arr)

        f_x_arr = self.get_fixedSate(episode_id)

        vf_x_arr, vf_y_arr = self.get_vfArrs(v_x_arr, v_y_arr, f_x_arr)
        # v_x_arr = self.obsSequence(v_x_arr)

        # self.mask_history(x_df)
        for i in range(len(vf_x_arr)):
            self.Xs.extend(vf_x_arr[i])
            self.Ys.extend(vf_y_arr[i])

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
# v_x_arr, v_y_arr = Data.get_stateTarget_arr(m_df, y_df)
# %%
m_df, y_df = Data.get_episode_df(811)
v_x_arr, v_y_arr = Data.get_stateTarget_arr(m_df, y_df)
v_x_arr = Data.applystateScaler(v_x_arr)
v_y_arr = Data.applytargetScaler(v_y_arr)

f_x_arr = Data.get_fixedSate(811)

# v_x_arr = Data.obsSequence(v_x_arr)
vf_x_arr, vf_y_arr = Data.get_vfArrs(v_x_arr, v_y_arr, f_x_arr)

vf_x_arr[0][1]
x_val[0]

vf_x_arr[0][1]
vf_x_arr[1][1]
len(vf_x_arr[10][0])
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
                "obsSequence_n": 1,
                "m_s":["vel", "pc", "act_long_p"],
                "y_s":["vel", "dv", "dx", "da", "a_ratio"],
                "retain":["vel"],
},
"exp_id": "NA",
"model_type": "merge_policy",
"Note": "NA"
}

# %%
def vis_dataDistribution(x):
    for i in range(len(x[0])):
        fig = plt.figure()
        plt.hist(x[:,i], bins=125)

vis_dataDistribution(x_val)
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

vis_beforeAfterScale(x_val, Data._states['mveh'])
