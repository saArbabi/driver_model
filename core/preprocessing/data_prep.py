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
import tensorflow as tf

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
    for name in ['training_episodes', 'validation_episodes', 'test_episodes']:
        file_name = './datasets/'+name+'.txt'
        with open(file_name, "r") as file:
            my_list = [int(item) for item in file.read().split()]
        episode_ids[name] = my_list

def read_fixed_stateArr():
    global fixed_arr0
    fixed_arr0 = pd.read_csv('./datasets/fixed_df0.txt', delimiter=' ',
                                                            header=None).values
    # First column is episode_id
    fixed_arr0[:,1:] = StandardScaler().fit(fixed_arr0[:,1:]).transform(fixed_arr0[:,1:])

read_episode_df()
read_episode_ids()
read_fixed_stateArr()
# %%
class DataPrep():
    random.seed(2020)

    def __init__(self, config, dirName):
        self.config = config['data_config']
        self.obsSequence_n = self.config["obsSequence_n"]
        self.step_size = self.config["step_size"]

        self.m_s = self.config["m_s"]
        self.y_s = self.config["y_s"]
        self.pred_horizon = self.config["pred_horizon"]
        self.setState_indx()
        self.setScalers() # will set the scaler attributes
        self.dirName = dirName
        os.mkdir(dirName)

    def obsSequence(self, state_arr, target_m_arr, target_y_arr,  condition_arr):
        state_seq = []
        target_m_seq = []
        target_y_seq = []
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
                    target_m_seq.append(target_m_arr[i:i+self.pred_horizon].tolist())
                    target_y_seq.append(target_y_arr[i:i+self.pred_horizon].tolist())
                    condition_seq.append(condition_arr[i:i+self.pred_horizon].tolist())

                i += step_size
            i_reset += 1
            i = i_reset

        return state_seq, target_m_seq, target_y_seq, condition_seq

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
        mveh_df = mveh_df[mveh_df['episode_id'] == episode_id].reset_index(drop=True)
        yveh_df = yveh_df[yveh_df['episode_id'] == episode_id].reset_index(drop=True)
        return mveh_df, yveh_df

    def applystateScaler(self, _arr):
        _arr[:, self.bool_pointer[-1]+1:] = self.state_scaler.transform(_arr[:, self.bool_pointer[-1]+1:])
        return _arr

    def applytarget_mScaler(self, _arr):
        return self.target_m_scaler.transform(_arr)

    def applytarget_yScaler(self, _arr):
        return self.target_y_scaler.transform(_arr.reshape(-1, 1))

    def applyconditionScaler(self, _arr):
        return self.condition_scaler.transform(_arr)

    def applyInvScaler(self, action_arr):
        """
        Note: only applies to target (i.e. action) values
        """
        return self.target_scaler.inverse_transform(action_arr)

    def setState_indx(self):
        i = 0
        self.bool_indx = {}
        self.scalar_indx = {}

        self.bool_indx['lc_type'] = i
        i += 1

        for state_key in self.m_s:
            self.scalar_indx[state_key+'_mveh'] = i
            i += 1

        for state_key in self.y_s:
            self.scalar_indx[state_key+'_yveh'] = i
            i += 1

        # these are used by the scaler
        self.bool_pointer = list(self.bool_indx.values())

    def get_stateTarget_arr(self, m_df, y_df):
        """Note: Not all states are used by model for prediction. Some are needed
            for state propagation.
        """
        target_m_df = m_df[['act_long','act_lat']]
        target_y_df = y_df['act_long']
        condition_df = y_df[['da', 'a_ratio']]

        state_df = pd.DataFrame()
        state_df = pd.concat([state_df, m_df['lc_type']], axis=1)
        state_df = pd.concat([state_df, m_df[self.m_s]], axis=1)
        state_df = pd.concat([state_df, y_df[self.y_s]], axis=1)

        return state_df.values, target_m_df.values, target_y_df.values, condition_df.values

    def setScalers(self):
        state_arr, target_m_arr, target_y_arr,  condition_arr = self.get_stateTarget_arr(m_df0, y_df0)
        self.state_scaler = StandardScaler().fit(state_arr[:, self.bool_pointer[-1]+1:])
        self.target_m_scaler = StandardScaler().fit(target_m_arr)
        self.target_y_scaler = StandardScaler().fit(target_y_arr.reshape(-1, 1))
        self.condition_scaler = StandardScaler().fit(condition_arr)

    def get_fixedSate(self, fixed_arr, episode_id):
        fixed_state_arr = fixed_arr[fixed_arr[:,0]==episode_id]
        return np.delete(fixed_state_arr, 0, axis=1)

    def get_timeStamps(self, size):
        ts = np.zeros([size, 1])
        t = 0.1
        for i in range(1, size):
            ts[i] = t
            t += 0.1
        return ts

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        m_df, y_df = self.get_episode_df(m_df0, y_df0, episode_id)
        state_arr, target_m_arr, target_y_arr,  condition_arr = self.get_stateTarget_arr(m_df, y_df)

        state_arr = self.applystateScaler(state_arr)
        target_m_arr = self.applytarget_mScaler(target_m_arr)
        target_y_arr = self.applytarget_yScaler(target_y_arr)
        condition_arr = self.applyconditionScaler(condition_arr)

        f_x_arr = self.get_fixedSate(fixed_arr0, episode_id)
        state_arr = np.concatenate([state_arr, f_x_arr], axis=1)

        state_arr, target_m_arr, target_y_arr,  condition_arr = self.obsSequence(
                                    state_arr, target_m_arr, target_y_arr,  condition_arr)
        # self.mask_history(x_df)

        # for i in range(len(vf_x_arr)):
        #     # use when generating ddata with time stamps
        #     self.states.extend(vf_x_arr[i])
        #     self.targets.extend(vf_y_arr[i])

        self.states.extend(state_arr)
        self.targets_m.extend(target_m_arr)
        self.targets_y.extend(target_y_arr)
        self.conditions.extend(condition_arr)

    def padArr(self, arr):
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                    arr, padding="post", maxlen=self.pred_horizon, dtype='float')
        return padded_inputs

    def pickler(self, episode_type):
        self.targets_m = self.padArr(self.targets_m)
        self.targets_y = self.padArr(self.targets_y)
        self.conditions = self.padArr(self.conditions)

        self.states = np.array(self.states)
        self.states, self.targets_m, self.targets_y, self.conditions = shuffle(self.states,
                                                    self.targets_m, self.targets_y, self.conditions)

        if episode_type == 'validation_episodes':
            # also you want to save validation df for later use
            with open(self.dirName+'/states_val', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_m_val', "wb") as f:
                pickle.dump(self.targets_m, f)

            with open(self.dirName+'/targets_y_val', "wb") as f:
                pickle.dump(self.targets_y, f)

            with open(self.dirName+'/conditions_val', "wb") as f:
                pickle.dump(self.conditions, f)

            delattr(self, 'states')
            delattr(self, 'targets_m')
            delattr(self, 'targets_y')
            delattr(self, 'conditions')

        elif episode_type == 'training_episodes':
            with open(self.dirName+'/states_train', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_m_train', "wb") as f:
                pickle.dump(self.targets_m, f)

            with open(self.dirName+'/targets_y_train', "wb") as f:
                pickle.dump(self.targets_y, f)

            with open(self.dirName+'/conditions_train', "wb") as f:
                pickle.dump(self.conditions, f)

            delattr(self, 'states')
            delattr(self, 'targets_m')
            delattr(self, 'targets_y')
            delattr(self, 'conditions')

            with open(self.dirName+'/data_obj', "wb") as f:
                pickle.dump(self, f)

            with open(self.dirName+'/test_m_df', "wb") as f:
                pickle.dump(m_df0[m_df0['episode_id'].isin(episode_ids['test_episodes'])], f)

            with open(self.dirName+'/test_y_df', "wb") as f:
                pickle.dump(y_df0[m_df0['episode_id'].isin(episode_ids['test_episodes'])], f)

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")

        self.states = []
        self.targets_m = []
        self.targets_y = []
        self.conditions = []
        for episode_id in episode_ids[episode_type]:
            self.episode_prep(episode_id)
        self.pickler(episode_type)
