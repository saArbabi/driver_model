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
random.seed(2020)
# a = [1,2,3,4,5,6,7]
# random.shuffle(a)

# car_dict = []{'scenario':'101', 'id': 23, 'lc_initiation_frm': 22, 'features': [[1,2,3],[1,2,3]]}
# car_dict
# car_dict['features']
#

config = {
 "model_config": {
    "1": 2,
    "hi": 2,
    "1": 2,
    "n_gmm_components": 4
},
"data_config": {
    "step_size": 3,
    "sequence_length": 5,
    "features": ['vel', 'pc','gap_size', 'dx', 'act_long_p', 'act_lat_p','lc_type'],
    "history_drop": {"percentage":0, "vehicle":['mveh', 'yveh']},
    "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx',
                                'act_long_p', 'act_lat_p', 'act_long', 'act_lat']},
    "scaler_path": './driver_model/experiments/scaler001'
},
"experiment_path": './driver_model/experiments/exp001',
"experiment_type": {"vehicle_name":'mveh', "model":"controller"}
}

config['data_config']['scaler_path']

# %%
def read_list(name):
    file_name = './driver_model/datasets/'+name+'.txt'
    file = open(file_name, "r")
    my_list = [int(item) for item in file.read().split()]
    file.close()
    return my_list

training_episodes = read_list('training_episodes')
validation_episodes = read_list('validation_episodes')

mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
       'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']


yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

mveh_df0 = pd.read_csv('./driver_model/datasets/mveh_df0.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df0 = pd.read_csv('./driver_model/datasets/yveh_df0.txt', delimiter=' ',
                        header=None, names=yveh_col)
# %%

class PrepData():

    def __init__(self, config):
        self.config = config['data_config']
        self.exp_path = config['experiment_path']
        self.exp_type = config['experiment_type']
        self.sequence_length = self.config["sequence_length"]
        self.step_size = self.config["step_size"]
        self.data_path = './driver_model/dataset'

    def get_scalers(self):
        dirName = self.config['scaler_path']+'/'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        if len(os.listdir(dirName)) == 0:
            print("Directory " , dirName ,  " Created ")
            for feature in self.config['scaler']['StandardScaler']:
                scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
                feature_array = mveh_df0[feature].values
                fit = scaler.fit(feature_array.reshape(-1, 1))
                file_name = dirName + feature
                pickle_out = open(file_name, "wb")
                pickle.dump(fit, pickle_out)
                pickle_out.close()
        else:
            print("Directory " , dirName ,  " already exists")

    def load_scaler(self, feature_name):
        dirName = self.config['scaler_path'] +'/'+ feature_name

        pickle_in = open(dirName,"rb")
        scaler = pickle.load(pickle_in)
        pickle_in.close()
        return scaler

    def drop_redundants(self, mveh, yveh):
        drop_col = ['id', 'episode_id', 'name', 'frm', 'scenario']
        if self.exp_type['vehicle_name'] == 'mveh' and self.exp_type['model'] == 'controller':
            self.action_size = 2
            mveh.drop(drop_col, inplace=True, axis=1)
            yveh.drop(drop_col+['act_long','lc_type'], inplace=True, axis=1)

    def scaler_transform(self, vehicle_df):
        vehicle_col = vehicle_df.columns
        for feature in self.config['scaler']['StandardScaler']:

            if feature in vehicle_col:
                scaler = self.load_scaler(feature)
                vehicle_df[feature] = scaler.transform(vehicle_df[feature].values.reshape(-1,1))
     
    def sequence(self, episode_arr):
        sequential_data = []
        i_reset = 0
        i = 0
        for chunks in range(self.step_size):
            prev_states = deque(maxlen=self.sequence_length)
            while i < len(episode_arr):
                row = episode_arr[i]
                prev_states.append([n for n in row[:-self.action_size]])
                if len(prev_states) == self.sequence_length:
                    sequential_data.append([np.array(prev_states), row[-self.action_size:]])
                i += self.step_size
            i_reset += 1
            i = i_reset

        return sequential_data

    def prep_episode(self, episode_id, setName):
        mveh = mveh_df0[mveh_df0['episode_id'] == episode_id].copy()
        yveh = yveh_df0[yveh_df0['episode_id'] == episode_id].copy()
        self.drop_redundants(mveh, yveh)
        self.scaler_transform(mveh)
        self.scaler_transform(yveh)
        episode_arr = np.concatenate([yveh.values,mveh.values], axis=1)
        sequenced_arr = self.sequence(episode_arr)

        self.test = episode_arr
        # return episode_arr
        return sequenced_arr

    def data_prep(self):
        self.get_scalers()
        # for episode_id in training_episodes:
        for episode_id in [811]:

            return self.prep_episode(episode_id, 'train')



prep = PrepData(config)

seq = prep.data_prep()
seq[0]

# %%
seq[1]
prep.test[0]
len(seq)
len(test_car)
# %%

test_car = mveh_df0.loc[mveh_df0['episode_id'] == 811].copy()
test_car.drop(['frm', 'vel'], inplace=True, axis=1)
test_car
