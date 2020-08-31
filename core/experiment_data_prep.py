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
    "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx', 'act_long', 'act_lat']}
},
"experiment_path": './driver_model/experiments/exp001'
}

config['data_config']
type(mveh_df0['lc_type']) .type
mveh_df0['lc_type'].dtype == 'int64'
# %%
file_name = './driver_model/datasets/'+'training_episodes'+'.txt'
file = open(file_name, "r")
[int(item) for item in file.read().split
()]
file.read()
int(file.readline().split()[0])
.split("\n")
file.close()
data

my_list = []
# %%
def read_list(name):
    file_name = './driver_model/datasets/'+name'+'.txt'
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
        self.sequence_length = self.config["sequence_length"]
        self.step_size = self.config["step_size"]
        self.data_path = './driver_model/dataset'

    def get_scaler(self):
        dirName = self.exp_path + '/scalers/'

        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")

        for feature in self.config['scaler']['StandardScaler']:
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            feature_array = mveh_df0[feature].values
            fit = scaler.fit(feature_array.reshape(-1, 1))
            file_name = dirName + feature
            pickle_out = open(file_name, "wb")
            pickle.dump(fit, pickle_out)
            pickle_out.close()

prep = PrepData(config)
prep.get_scaler()
