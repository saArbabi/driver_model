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
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self._scale_dfs() # will scale the dataframes

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

    def mask_history(self, x_arr):
        pass
        # dropout_percentage = self.config['mask_history']['percentage']
        # if  dropout_percentage != 0:
        #     target_name = self.config['mask_history']['vehicle']
        #     if target_name == 'mveh':
        #         index = mveh.sample(int(len(mveh)*dropout_percentage)).index
        #         mveh.loc[:, index, mveh.columns != 'lc_type']=0


    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        x_arr, y_arr = self.get_xy_arr(episode_id)
        # self.mask_history(x_df)
        return self.sequence(x_arr, y_arr)

    def store_data(self, x_seq, y_seq, setName):
        self.shuffArr(x_seq)
        self.shuffArr(y_seq)

        if setName == 'train':
            self.x_train.append(x_seq)
            self.y_train.append(y_seq)
        else:
            self.x_val.append(x_seq)
            self.y_val.append(y_seq)


    def get_relevant_dfCols(self):
        """
        Gets the relevant df columns for this particular experiment.
        """
        mveh_keep = ['episode_id']+self.veh_states['mveh']
        yveh_keep = self.veh_states['yveh']
        mveh_df = mveh_df0[mveh_keep].copy()
        yveh_df = pd.DataFrame(yveh_df0[yveh_keep].values) # to avoid overlapping names
        state_df = pd.concat([mveh_df, yveh_df], axis=1)

        if self.model_type == 'merge_controller':
            target_df = mveh_df0[['episode_id','act_long','act_lat']].copy()

        return state_df, target_df

    def _scale_dfs(self):
        state_df, target_df = self.get_relevant_dfCols()
        state_col = state_df.columns[~state_df.columns.isin(['episode_id','lc_type'])]
        target_col = target_df.columns[~target_df.columns.isin(['episode_id'])]

        state_arr = state_df.loc[:, state_col].values
        target_arr = target_df.loc[:, target_col].values

        scaler = StandardScaler()
        self.state_scaler = scaler.fit(state_arr)
        scaler = StandardScaler()
        self.target_scaler = scaler.fit(target_arr)

        state_df.loc[:, state_col] = self.state_scaler.transform(state_arr)
        target_df.loc[:, target_col] = self.target_scaler.transform(target_arr)

        self.state_df = state_df # these are scaled
        self.target_df = target_df

    def flattenDataList(self):
        self.x_train = [row for item in self.x_train for row in item]
        self.y_train = [row for item in self.y_train for row in item]
        self.x_val = [row for item in self.x_val for row in item]
        self.y_val = [row for item in self.y_val for row in item]
        return np.array(self.x_train), np.array(self.y_train), np.array(self.x_val), np.array(self.y_val)
        # return self.x_train, self.y_train, self.x_val, self.y_val

    def get_xy_arr(self, episode_id):
        state_df_id = self.state_df[self.state_df['episode_id'] == episode_id]
        target_df_id = self.target_df[self.target_df['episode_id'] == episode_id]
        del state_df_id['episode_id']
        del target_df_id['episode_id']

        return state_df_id.values, target_df_id.values

    def data_prep(self):
        for episode_id in episode_ids['training_episodes']:
            x_seq, y_seq  = self.episode_prep(episode_id)
            self.store_data(x_seq, y_seq, 'train')

        for episode_id in episode_ids['validation_episodes']:
            x_seq, y_seq = self.episode_prep(episode_id)
            self.store_data(x_seq, y_seq, 'val')

        return self.flattenDataList()

# Data = DataObj(config)
# x_train, y_train, x_val ,y_val = Data.data_prep()
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
#
# x_train[0]
# y_train[0]
# import matplotlib.pyplot as plt
# len(training_episodes)
# len(x_train[0])
# np.array(x_val)
# # %%
# def vis_beforeAfterScale(x, features):
#     i = 0
#     x = Data.state_scaler.inverse_transform(x[:,1:])
#     features.remove('lc_type')
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
