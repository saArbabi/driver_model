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
def read_list(name):
    file_name = './datasets/'+name+'.txt'
    file = open(file_name, "r")
    my_list = [int(item) for item in file.read().split()]
    file.close()
    return my_list

training_episodes = read_list('training_episodes')
validation_episodes = read_list('validation_episodes')

mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
       'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']


yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

mveh_df0 = pd.read_csv('./datasets/mveh_df0.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df0 = pd.read_csv('./datasets/yveh_df0.txt', delimiter=' ',
                        header=None, names=yveh_col)
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

"data_config": {
    "step_size": 3,
    "sequence_n": 1,
    "veh_states":{"mveh":["lc_type", "vel", "pc","gap_size", "dx", "act_long_p", "act_lat_p"],
                    "yveh":["vel", "act_long_p"]} ,
},
"exp_id": "NA",
"model_type": "merge_controller",
"Note": "NA"
}



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
        self.scalers = {}

    def getScalers(self):
        if not self.scalers:
            target_values = ['act_long', 'act_lat']
            scale_col = set(self.veh_states['mveh'] + self.veh_states['yveh'])
            scale_col.remove('lc_type')
            for veh_state in scale_col:
                scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
                if veh_state in self.veh_states['mveh']:
                    veh_state_array = mveh_df0[veh_state].values
                else:
                    veh_state_array = yveh_df0[veh_state].values

                scaler_fit = scaler.fit(veh_state_array.reshape(-1, 1))
                self.scalers[veh_state] = scaler_fit

            for veh_state in target_values:
                veh_state_array = mveh_df0[veh_state].values
                scaler_fit = scaler.fit(veh_state_array.reshape(-1, 1))
                self.scalers[veh_state] = scaler_fit

    def drop_redundants(self, mveh, yveh):
        drop_col = ['id', 'episode_id', 'name', 'frm', 'scenario']
        if self.model_type == 'merge_controller':
            self.action_size = 2
            mveh.drop(drop_col, inplace=True, axis=1)
            yveh.drop(drop_col+['act_long','lc_type'], inplace=True, axis=1)

    def stateScaler(self, mveh, yveh):
        mveh_col = mveh.columns
        yveh_col = yveh.columns

        for veh_state in self.scalers.keys():
            if veh_state in mveh_col:
                mveh[veh_state] = self.scalers[veh_state].transform(mveh[veh_state].values.reshape(-1,1))
            if veh_state in yveh_col:
                yveh[veh_state] = self.scalers[veh_state].transform(yveh[veh_state].values.reshape(-1,1))

    def sequence(self, episode_arr):
        sequential_data = []

        if self.sequence_n != 0:
            i_reset = 0
            i = 0
            for chunks in range(self.step_size):
                prev_states = deque(maxlen=self.sequence_n)
                while i < len(episode_arr):
                    row = episode_arr[i]
                    prev_states.append(row[:-self.action_size])
                    if len(prev_states) == self.sequence_n:
                        sequential_data.append([np.array(prev_states), row[-self.action_size:]])
                    i += self.step_size
                i_reset += 1
                i = i_reset
        else:
            for i in range( len(episode_arr)):
                row = episode_arr[i]
                sequential_data.append([np.array(row[:-self.action_size]), row[-self.action_size:]])
        return sequential_data

    def history_drop(self, mveh, yveh):
        pass
        # dropout_percentage = self.config['history_drop']['percentage']
        # if  dropout_percentage != 0:
        #     target_name = self.config['history_drop']['vehicle']
        #     if target_name == 'mveh':
        #         index = mveh.sample(int(len(mveh)*dropout_percentage)).index
        #         mveh.loc[index, mveh.columns != 'lc_type']=0

    def prep_episode(self, episode_id):
        mveh = mveh_df0[mveh_df0['episode_id'] == episode_id].copy()
        yveh = yveh_df0[yveh_df0['episode_id'] == episode_id].copy()
        self.drop_redundants(mveh, yveh)
        self.getScalers()
        self.stateScaler(mveh, yveh)
        self.history_drop(mveh, yveh)
        episode_arr = np.concatenate([mveh.values, yveh.values], axis=1)
        sequenced_arr = self.sequence(episode_arr)

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

    def data_prep(self):
        # for episode_id in training_episodes:
        for episode_id in training_episodes:
            sequenced_arr = self.prep_episode(episode_id)
            self.store_data(sequenced_arr, 'train')

        for episode_id in validation_episodes:
            sequenced_arr = self.prep_episode(episode_id)
            self.store_data(sequenced_arr, 'val')

        return self.x_train, self.y_train, self.x_val ,self.y_val

# %%
# Data = DataObj(conf)
# x_train, y_train, x_val ,y_val = Data.data_prep()
# len(Data.x_train[0][0])
# len(Data.x_train)
# x_train[0]
# x_train[1]
# import matplotlib.pyplot as plt
# x_val0 = np.array([x[0] for x in x_val])
# len(training_episodes)
# len(x_train[0][0])
# def vis_beforeAfterScale(x_val):
#     i = 0
#     for feature in Data.veh_states['mveh']:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1,2,1)
#         ax2 = fig.add_subplot(1,2,2)
#
#         ax1.hist(mveh_df0[feature].iloc[0:], bins=125)
#         ax2.hist(x_val[:,i], bins=125)
#         i += 1
#         ax1.set_title(feature + ' before')
#         ax2.set_title(feature + ' after')
#         plt.title(feature)
# # %%
#
# vis_beforeAfterScale(x_val0)
