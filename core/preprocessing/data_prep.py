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

    def sequence(self, x_arr, y_arr):
        sequential_data = []
        if self.sequence_n != 1:
            i_reset = 0
            i = 0
            for chunks in range(self.step_size):
                prev_states = deque(maxlen=self.sequence_n)
                while i < len(x_arr):
                    prev_states.append(x_arr[i])
                    if len(prev_states) == self.sequence_n:
                        sequential_data.append([np.array(prev_states), y_arr[i]])
                    i += self.step_size
                i_reset += 1
                i = i_reset
        else:
            for i in range(len(x_arr)):
                sequential_data.append([np.array(x_arr[i]), y_arr[i]])
        return sequential_data

    def mask_history(self, x_arr):
        pass
        # dropout_percentage = self.config['mask_history']['percentage']
        # if  dropout_percentage != 0:
        #     target_name = self.config['mask_history']['vehicle']
        #     if target_name == 'mveh':
        #         index = mveh.sample(int(len(mveh)*dropout_percentage)).index
        #         mveh.loc[:, index, mveh.columns != 'lc_type']=0

    def get_xy_arr(self, episode_id, mveh_df, yveh_df):
        """
        :Return: model input and target arrays
        """
        mveh = mveh_df[mveh_df['episode_id'] == episode_id]
        yveh = yveh_df[yveh_df['episode_id'] == episode_id]

        if 'mveh' in self.action_space.keys():
            y_arr = mveh.loc[:, self.action_space['mveh']].values

        x_arr = pd.concat([mveh[self.veh_states['mveh']],yveh[self.veh_states['yveh']]], axis=1).values
        return x_arr, y_arr

    def prep_episode(self, episode_id, mveh_df, yveh_df):
        x_arr, y_arr = self.get_xy_arr(episode_id, mveh_df, yveh_df)
        # self.mask_history(x_df)
        sequenced_arr = self.sequence(x_arr, y_arr)

        return sequenced_arr

    def store_data(self, sequenced_arr, setName):
        random.shuffle(sequenced_arr)

        if setName == 'train':
            _x, _y = self.x_train, self.y_train
        else:
            _x, _y = self.x_val, self.y_val

        for x, y in sequenced_arr:
            _x.append(x)
            _y.append(y)

    def get_relevant_cols(self):
        """
        Gets the relevant df columns for this particular experiment.
        """
        mveh_keep = ['episode_id']+self.veh_states['mveh']
        yveh_keep = ['episode_id']+self.veh_states['yveh']

        if self.model_type == 'merge_controller':
            mveh_keep += ['act_long','act_lat']
            self.action_space = {'mveh':['act_long','act_lat']}

        return mveh_keep, yveh_keep

    def scale_dfs(self, mveh_df, yveh_df):
        mveh_keep, yveh_keep = self.get_relevant_cols()
        mveh_df.loc[:, :] = mveh_df[mveh_keep]
        yveh_df.loc[:, :] = yveh_df[yveh_keep]
        mveh_keep = [item for item in mveh_keep if item not in ['episode_id','lc_type']]
        yveh_keep.remove('episode_id')

        mveh_arr = mveh_df[mveh_keep].values
        yveh_arr = yveh_df[yveh_keep].values

        scaler = StandardScaler()
        mveh_fit = scaler.fit(mveh_arr)
        scaler = StandardScaler()
        yveh_fit = scaler.fit(yveh_arr)

        mveh_df.loc[:, mveh_keep] = mveh_fit.transform(mveh_arr)
        yveh_df.loc[:, yveh_keep] = yveh_fit.transform(yveh_arr)

        return mveh_df, yveh_df

    def data_prep(self):
        # for episode_id in training_episodes:
        mveh_df,  yveh_df = self.scale_dfs(mveh_df0.copy(),  yveh_df0.copy())

        for episode_id in episode_ids['training_episodes']:
            sequenced_arr = self.prep_episode(episode_id, mveh_df,  yveh_df)
            self.store_data(sequenced_arr, 'train')

        for episode_id in episode_ids['validation_episodes']:
            sequenced_arr = self.prep_episode(episode_id, mveh_df,  yveh_df)
            self.store_data(sequenced_arr, 'val')

        return np.array(self.x_train), np.array(self.y_train), np.array(self.x_val), np.array(self.y_val)


# %%
# Data = DataObj(config)
# x_train, y_train, x_val ,y_val = Data.data_prep()

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
#     "veh_states":{"mveh":["lc_type", "vel", "pc", "gap_size", "dx"],
#                     "yveh":["vel"]}
# },
# "exp_id": "NA",
# "model_type": "merge_controller",
# "Note": "NA"
# }
#
# x_train
# y_train
# import matplotlib.pyplot as plt
# len(training_episodes)
# len(x_train)
# # %%
# def vis_beforeAfterScale(x, features):
#     i = 0
#     for feature in features:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1,2,1)
#         ax2 = fig.add_subplot(1,2,2)
#
#         ax1.hist(mveh_df0[feature], bins=125)
#         ax2.hist(x[:,i], bins=125)
#         i += 1
#         ax1.set_title(feature + ' before')
#         ax2.set_title(feature + ' after')
#
# vis_beforeAfterScale(np.array(x_val), Data.veh_states['mveh'])
