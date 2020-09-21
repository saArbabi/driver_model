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
class DataPrep():
    random.seed(2020)

    def __init__(self, config, dirName):
        self.config = config['data_config']
        self.model_type = config['model_type']
        self.obsSequence_n = self.config["obsSequence_n"]
        self.step_size = self.config["step_size"]

        self.m_s = self.config["m_s"]
        self.y_s = self.config["y_s"]

        self.setState_indx()
        self.setScalers() # will set the scaler attributes
        self.dirName = dirName
        os.mkdir(dirName)

    def obsSequence(self, x_arr, y_arr):
        x_seq = [] # sequenced x array
        y_seq = []

        if self.obsSequence_n != 1:
            i_reset = 0
            i = 0
            for chunks in range(self.step_size):
                prev_states = deque(maxlen=self.obsSequence_n)
                while i < len(x_arr):
                    prev_states.append(x_arr[i])
                    if len(prev_states) == self.obsSequence_n:
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

    def mask_history(self, v_x_arr):
        pass
        # if self.config['seq_mask_prob'] != 0:
        #     if random.random() > probability:
        #         return n
        #     else:
        #         return round(np.random.uniform(low=-1, high=1),2)
        #     dropout_percentage = self.config['mask_history']['percentage']
        #     if  dropout_percentage != 0:
        #         target_name = self.config['mask_history']['vehicle']
        #         if target_name == 'mveh':
        #             index = mveh.sample(int(len(mveh)*dropout_percentage)).index
        #             mveh.loc[:, index, mveh.columns != 'lc_type']=0
        #
        #         self.scalar_indx[state_key+'_mveh'] = i
        # else:
        #     return v_x_arr

    def get_episode_df(self, mveh_df, yveh_df, episode_id):
        mveh_df = mveh_df[mveh_df['episode_id'] == episode_id]
        yveh_df = yveh_df[yveh_df['episode_id'] == episode_id]
        return mveh_df, yveh_df

    def applystateScaler(self, _arr):
        _arr[:, self.bool_pointer[-1]+1:] = self.state_scaler.transform(_arr[:, self.bool_pointer[-1]+1:])
        return np.delete(_arr, self.retain_pointer, axis=1)

    def applytargetScaler(self, _arr):
        return self.target_scaler.transform(_arr)

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
        self.bool_indx['lc_type'] = i
        i += 1

        for state_key in self.m_s:
            self.scalar_indx[state_key+'_mveh'] = i
            i += 1

        for state_key in self.y_s:
            self.scalar_indx[state_key+'_yveh'] = i
            i += 1

        # these are used by the scaler
        self.retain_pointer = list(self.retain_indx.values())
        self.bool_pointer = list(self.bool_indx.values())

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

        state_df = pd.concat([state_df, m_df['lc_type']], axis=1)
        state_df = pd.concat([state_df, m_df[self.m_s]], axis=1)
        state_df = pd.concat([state_df, y_df[self.y_s]], axis=1)

        return state_df.values, target_df.values

    def setScalers(self):
        state_arr, target_arr = self.get_stateTarget_arr(m_df0, y_df0)
        self.state_scaler = StandardScaler().fit(state_arr[:, self.bool_pointer[-1]+1:])
        self.target_scaler = StandardScaler().fit(target_arr)

    def get_fixedSate(self, episode_id):
        state_arr = fixed_state_arr[fixed_state_arr[:,1]==episode_id]
        return np.delete(state_arr, 1, axis=1)

    def get_timeStamps(self, size):
        ts = np.zeros([size, 1])
        t = 0.01
        for i in range(1, size):
            ts[i] = t
            t += 0.01
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
        m_df, y_df = self.get_episode_df(m_df0, y_df0, episode_id)
        v_x_arr, v_y_arr = self.get_stateTarget_arr(m_df, y_df)

        v_x_arr = self.applystateScaler(v_x_arr)
        v_y_arr = self.applytargetScaler(v_y_arr)

        # f_x_arr = self.get_fixedSate(episode_id)
        # vf_x_arr, vf_y_arr = self.get_vfArrs(v_x_arr, v_y_arr, f_x_arr)
        # v_x_arr, v_y_arr = self.obsSequence(v_x_arr, v_y_arr)
        # self.mask_history(x_df)

        # for i in range(len(vf_x_arr)):
        #     # use when generating ddata with time stamps
        #     self.Xs.extend(vf_x_arr[i])
        #     self.Ys.extend(vf_y_arr[i])

        self.Xs.extend(v_x_arr)
        self.Ys.extend(v_y_arr)

    def shuffArr(self, arr):
        random.Random(2020).shuffle(arr)
        return np.array(arr)

    def pickler(self, episode_type):
        if episode_type == 'validation_episodes':
            # also you want to save validation df for later use
            with open(self.dirName+'/x_val', "wb") as f:
                pickle.dump(self.shuffArr(self.Xs), f)

            with open(self.dirName+'/y_val', "wb") as f:
                pickle.dump(self.shuffArr(self.Ys), f)

            delattr(self, 'Xs')
            delattr(self, 'Ys')



        elif episode_type == 'training_episodes':
            with open(self.dirName+'/x_train', "wb") as f:
                pickle.dump(self.shuffArr(self.Xs), f)

            with open(self.dirName+'/y_train', "wb") as f:
                pickle.dump(self.shuffArr(self.Ys), f)
            delattr(self, 'Xs')
            delattr(self, 'Ys')

            with open(self.dirName+'/data_obj', "wb") as f:
                # Validation set is saved as part of data_obj for later use
                self.val_m_df = m_df0[m_df0['episode_id'].isin(episode_ids['validation_episodes'])]
                self.val_y_df = y_df0[y_df0['episode_id'].isin(episode_ids['validation_episodes'])]
                self.validation_episodes = episode_ids['validation_episodes']
                pickle.dump(self, f)

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")

        self.Xs = []
        self.Ys = []
        for episode_id in episode_ids[episode_type][0:3]:
            self.episode_prep(episode_id)

        self.pickler(episode_type)




# %%
# Data = DataObj(config)
# x_train, y_train = Data.data_prep('training_episodes')
# x_val, y_val = Data.data_prep('validation_episodes')
# # m_df, y_df = Data.get_episode_df(811)
# # v_x_arr, v_y_arr = Data.get_stateTarget_arr(m_df, y_df)
# # %%
# m_df, y_df = Data.get_episode_df(811)
# v_x_arr, v_y_arr = Data.get_stateTarget_arr(m_df, y_df)
# v_x_arr = Data.applystateScaler(v_x_arr)
# v_y_arr = Data.applytargetScaler(v_y_arr)
#
# f_x_arr = Data.get_fixedSate(811)
#
# # v_x_arr = Data.obsSequence(v_x_arr)
# vf_x_arr, vf_y_arr = Data.get_vfArrs(v_x_arr, v_y_arr, f_x_arr)
#
# %%


# %%
# config1 = {
#  "model_config": {
#      "learning_rate": 1e-2,
#      "neurons_n": 50,
#      "layers_n": 2,
#      "epochs_n": 5,
#      "batch_n": 128,
#      "components_n": 4
# },
# "data_config": {"step_size": 3,
#                 "obsSequence_n": 2,
#                 "m_s":["vel", "pc", "act_long_p"],
#                 "y_s":["vel", "dv", "dx", "da", "a_ratio"],
#                 "retain":["vel"],
# },
# "exp_id": "NA",
# "model_type": "merge_policy",
# "Note": "NA"
# }
#
#
#
# config2 = {
#  "model_config": {
#      "learning_rate": 1e-2,
#      "neurons_n": 50,
#      "layers_n": 2,
#      "epochs_n": 5,
#      "batch_n": 128,
#      "components_n": 5
# },
# "data_config": {"step_size": 3,
#                 "obsSequence_n": 1,
#                 "m_s":["vel", "pc", "act_long_p"],
#                 "y_s":["vel", "dv", "dx", "da", "a_ratio"],
#                 "retain":["vel"],
# },
# "exp_id": "NA",
# "model_type": "merge_policy",
# "Note": "NA"
# }
# config1==config2
# %%
# def vis_dataDistribution(x):
#     for i in range(len(x[0])):
#         fig = plt.figure()
#         plt.hist(x[:,i], bins=125)
#
# vis_dataDistribution(x_val)
# # %%
# def vis_beforeAfterScale(x, features):
#     i = 0
#     # x = Data.state_scaler.inverse_transform(x[:,1:])
#     for feature in features:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1,2,1)
#         ax2 = fig.add_subplot(1,2,2)
#
#         ax1.hist(m_df0[feature], bins=125)
#         # ax2.hist(x[:,i], bins=125)
#         ax2.hist(x[:,i], bins=125)
#
#         i += 1
#         ax1.set_title(feature + ' before')
#         ax2.set_title(feature + ' after')
#
# vis_beforeAfterScale(x_val, Data._states['mveh'])
