"""
- Outputs training set and validation set for a particular experiment prior to
start of training.
"""
from collections import deque
import numpy as np
from sklearn.utils import shuffle
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import dill
import tensorflow as tf

# %%
def read_episode_ids():
    global episode_ids

    episode_ids = {}
    for name in ['training_episodes', 'validation_episodes', 'test_episodes']:
        file_name = './datasets/'+name+'.txt'
        with open(file_name, "r") as file:
            my_list = [int(item) for item in file.read().split()]
        episode_ids[name] = my_list

def read_ffadj_stateArr():
    global all_state_arr, all_target_arr, all_condition_arr
    ffadj_arr0 = pd.read_csv('./datasets/ffadj_df0.txt', delimiter=' ',
                                                            header=None).values
    # First column is episode_id


read_episode_df()
read_episode_ids()
read_ffadj_stateArr()
# %%
class DataPrep():
    random.seed(2020)

    def __init__(self, config, dirName):
        self.config = config['data_config']
        self.obsSequence_n = self.config["obsSequence_n"]
        self.step_size = self.config["step_size"]
        self.pred_horizon = self.config["pred_horizon"]
        self.setScalers() # will set the scaler attributes
        self.dirName = dirName
        os.mkdir(dirName)

    def obsSequence(self, state_arr, target_arr, condition_arr):
        state_seq = []
        target_seq = []
        condition_seq = []

        step_size = 1
        i_reset = 0
        i = 0
        for chunks in range(step_size):
            prev_states = deque(maxlen=self.obsSequence_n)
            while i < (len(state_arr)-2):
                # 2 is minimum prediction horizon
                prev_states.append(state_arr[i])
                if len(prev_states) == self.obsSequence_n:
                    state_seq.append(np.array(prev_states))
                    target_seq.append(target_arr[i:i+self.pred_horizon].tolist())

                i += step_size
            i_reset += 1
            i = i_reset

        return state_seq, target_seq, condition_seq

    def mask_history(self, v_x_arr):
        pass
        # if self.config['seqask_prob'] != 0:
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
        #         self.scalar_indx[state_key+'veh'] = i
        # else:
        #     return v_x_arr

    def get_episode_arr(self, episode_id):
        state_arr = all_state_arr[all_state_arr[0] == episode_id]
        target_arr = all_target_arr[all_target_arr[0] == episode_id]
        condition_arr = all_condition_arr[all_condition_arr[0] == episode_id]
        return state_arr[:, 1], target_arr[:, 1], condition_arr[:, 1]

    def applystateScaler(self, _arr):
        _arr[:, :-4] = self.state_scaler.transform(_arr[:, :-4])
        return _arr

    def applytargetScaler(self, _arr):
        return self.target_scaler.transform(_arr)

    def apply_InvScaler(self, action_arr):
        """
        Note: only applies to target (i.e. action) values
        """
        return self.target_scaler.inverse_transform(action_arr)

    def setScalers(self):
        self.state_scaler = StandardScaler().fit(all_state_arr[1:, -4:])
        self.target_scaler = StandardScaler().fit(all_target_arr[1:])

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        state_arr, target_arr, condition_arr = self.get_episode_arr(episode_id)

        state_arr = self.applystateScaler(state_arr)
        target_arr = self.applytargetScaler(target_arr)
        condition_arr = self.applytargetScaler(condition_arr)
        state_seq, target_seq, condition_seq = self.obsSequence(state_arr, target_arr, condition_arr)
        # self.mask_history(x_df)
        self.states.extend(state_seq)
        self.targets.extend(target_seq)
        self.conditions.extend(condition_seq)

    def padArr(self, arr):
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                    arr, padding="post", maxlen=self.pred_horizon, dtype='float')
        return padded_inputs

    def pickler(self, episode_type):
        self.targets = self.padArr(self.targets)
        self.conditions = self.padArr(self.conditions)

        self.states = np.array(self.states)
        self.states, self.targets, self.conditions = shuffle(self.states,
                                                    self.targets, self.conditions)

        if episode_type == 'validation_episodes':
            # also you want to save validation df for later use
            with open(self.dirName+'/states_val', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_val', "wb") as f:
                pickle.dump(self.targets, f)

            with open(self.dirName+'/conditions_val', "wb") as f:
                pickle.dump(self.conditions, f)

            delattr(self, 'states')
            delattr(self, 'targets')
            delattr(self, 'conditions')

        elif episode_type == 'training_episodes':
            with open(self.dirName+'/states_train', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_train', "wb") as f:
                pickle.dump(self.targets, f)

            with open(self.dirName+'/conditions_train', "wb") as f:
                pickle.dump(self.conditions, f)

            delattr(self, 'states')
            delattr(self, 'targets')
            delattr(self, 'conditions')

            with open(self.dirName+'/data_obj', "wb") as f:
                dill.dump(self, f)

            with open(self.dirName+'/test_df', "wb") as f:
                pickle.dump(m_df0[m_df0['episode_id'].isin(episode_ids['test_episodes'])], f)

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")
        self.states = []
        self.targets = []
        self.targets_y = []
        self.conditions = []
        for episode_id in episode_ids[episode_type]:
            self.episode_prep(episode_id)
        self.pickler(episode_type)
