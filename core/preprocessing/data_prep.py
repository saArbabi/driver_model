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
    global mveh_df0, yveh_df0

    mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
           'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']

    yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

    mveh_df0 = pd.read_csv('./datasets/mveh_df0.txt', delimiter=' ',
                            header=None, names=mveh_col)
    yveh_df0 = pd.read_csv('./datasets/yveh_df0.txt', delimiter=' ',
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
class DataObj():
    random.seed(2020)

    def __init__(self, config):
        self.config = config['data_config']
        self.model_type = config['model_type']
        self.sequence_n = self.config["sequence_n"]
        self.step_size = self.config["step_size"]
        self.veh_states = self.config["veh_states"]
        self.states = []
        self.targets = []
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
        mveh_df = mveh_df0[mveh_df0['episode_id'] == episode_id]
        yveh_df = yveh_df0[yveh_df0['episode_id'] == episode_id]

        return mveh_df, yveh_df

    def applyScaler(self, x_arr, y_arr):
        x_arr[:,1:] = self.state_scaler.transform(np.delete(x_arr,
                                    self.mvehindx['lc_type'], axis=1))
        y_arr = self.target_scaler.transform(y_arr)

        return x_arr, y_arr

    def applyInvScaler(self, action_arr):
        """
        Note: only applies to target (i.e. action) values
        """
        return self.target_scaler.inverse_transform(action_arr)

    def setState_indx(self):
        veh_states = self.config['veh_states']
        i = 0
        self.mvehindx = {}
        self.yvehindx = {}

        for state_key in veh_states['mveh']:
            self.mvehindx[state_key] = i
            i += 1

        for state_key in veh_states['yveh']:
            self.yvehindx[state_key] = i
            i += 1

    def get_stateTarget_arr(self, mveh_df, yveh_df):

        if self.model_type == 'merge_controller':
            target_df = mveh_df[['act_long','act_lat']]

        mveh_df = mveh_df[self.veh_states['mveh']]
        yveh_df = yveh_df[self.veh_states['yveh']]
        state_df = pd.concat([mveh_df, yveh_df], axis=1)

        return state_df.values, target_df.values

    def setScalers(self):
        state_arr, target_arr = self.get_stateTarget_arr(mveh_df0, yveh_df0)

        self.state_scaler = StandardScaler().fit(np.delete(state_arr,
                                                self.mvehindx['lc_type'], axis=1))
        self.target_scaler = StandardScaler().fit(target_arr)


    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        mveh_df, yveh_df = self.get_episode_df(episode_id)
        x_arr, y_arr = self.get_stateTarget_arr(mveh_df, yveh_df)
        x_arr, y_arr = self.applyScaler(x_arr, y_arr)
        x_arr, y_arr = self.sequence(x_arr, y_arr)

        # self.mask_history(x_df)
        for i in range(len(x_arr)):
            self.states.append(x_arr[i])
            self.targets.append(y_arr[i])


    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")

        for episode_id in episode_ids[episode_type]:
            self.episode_prep(episode_id)

        return self.shuffArr(self.states), self.shuffArr(self.targets)
#
# Data = DataObj(config)
# x_train, y_train = Data.data_prep('training_episodes')
# x_val, y_val = Data.data_prep('validation_episodes')
#
# # %%
#
#
# config = {
#  "model_config": {
#      "learning_rate": 1e-2,
#      "neurons_n": 50,
#      "layers_n": 2,
#      "epochs_n": 5,
#      "batch_n": 128,
#      "components_n": 5
# },
# "data_config": {    "step_size": 3,
#     "sequence_n": 1,
#     "veh_states":{"mveh":["lc_type", "vel", "pc", "gap_size", "dx",'act_long_p'],
#                     "yveh":["vel"]}
# },
# "exp_id": "NA",
# "model_type": "merge_controller",
# "Note": "NA"
# }
# x_train[1]
#
# x_train[0]
# y_train[0]
# import matplotlib.pyplot as plt
# len(training_episodes)
# len(x_train[0])
# np.array(x_val)
# len()
# # %%
# def vis_beforeAfterScale(x, features):
#     i = 0
#     # x = Data.state_scaler.inverse_transform(x[:,1:])
#     for feature in features:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1,2,1)
#         ax2 = fig.add_subplot(1,2,2)
#
#         ax1.hist(mveh_df0[feature], bins=125)
#         # ax2.hist(x[:,i], bins=125)
#         ax2.hist(x[:,i], bins=125)
#
#         i += 1
#         ax1.set_title(feature + ' before')
#         ax2.set_title(feature + ' after')
#
# vis_beforeAfterScale(x_val, Data.veh_states['mveh'])
