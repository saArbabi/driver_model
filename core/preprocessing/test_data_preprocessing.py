from models.core.preprocessing import data_prep
from models.core.preprocessing import data_obj
import numpy as np
import pickle
import matplotlib.pyplot as plt

from importlib import reload
reload(data_prep)
reload(data_obj)
DataPrep = data_prep.DataPrep
DataObj = data_obj.DataObj

config = {
 "model_config": {
     "learning_rate": 1e-3,
     "neurons_n": 50,
     "layers_n": 2,
     "epochs_n": 50,
     "batch_n": 1124,
     "components_n": 5
},
"data_config": {"step_size": 1,
                "obsSequence_n": 1,
                "m_s":["vel", "pc"],
                "y_s":["vel", "dv", "dx", "da", "a_ratio"],
                "retain":["vel"],
                "Note": "baseline - no time stamps"
                # "Note": "Here I am adding the time stamp"
},
"exp_id": "NA",
"model_type": "merge_policy",
"Note": "NA"
}

x_train, y_train, x_val, y_val = DataObj(config).loadData()
x_train[-1][0]
x_train[-1][1]

len(x_train)
len(x_val[0])
x_train[1]
y_train[1]
# %%
with open('./datasets/preprocessed/'+'20200921-123920'+'/'+'data_obj', 'rb') as f:
    data_obj = pickle.load(f)
data_obj.validation_episodes[3]

m_df, y_df = data_obj.get_episode_df(data_obj.val_m_df, data_obj.val_y_df, 1635)
v_x_arr, v_y_arr = data_obj.get_stateTarget_arr(m_df, y_df)
v_x_arr = data_obj.applystateScaler(v_x_arr)
v_y_arr = data_obj.applytargetScaler(v_y_arr)

v_x_arr, v_y_arr = data_obj.obsSequence(v_x_arr, v_y_arr)
v_x_arr[0]
10e-4
# %%

def vis_dataDistribution(x):
    for i in range(len(x[0])):
        fig = plt.figure()
        plt.hist(x[:1000000,i], bins=125)

vis_dataDistribution(x_train)
# %%
from collections import deque

v_x_arr = range(size0)
v_y_arr = range(size0)
x_seq = [] # obsSequenced x array
y_seq = [] # obsSequenced x array
obsSequence_n = 5
step_size = 0
if obsSequence_n != 1:
    i_reset = 0
    i = 0
    for chunks in range(step_size):
        prev_states = deque(maxlen=obsSequence_n)
        while i < len(v_x_arr):
            prev_states.append(v_x_arr[i])
            if len(prev_states) == obsSequence_n:
                x_seq.append(np.array(prev_states))
                y_seq.append(v_y_arr[i])
            i += step_size
        i_reset += 1
        i = i_reset
else:
    for i in range(len(v_x_arr)):
        x_seq.append(v_x_arr[i])




# %%
def mask_history(self, v_x_arr):
    if self.config['seq_mask_prob'] != 0:
        if random.random() > probability:
            return n
        else:
            return round(np.random.uniform(low=-1, high=1),2)
        dropout_percentage = self.config['mask_history']['percentage']
        if  dropout_percentage != 0:
            target_name = self.config['mask_history']['vehicle']
            if target_name == 'mveh':
                index = mveh.sample(int(len(mveh)*dropout_percentage)).index
                mveh.loc[:, index, mveh.columns != 'lc_type']=0

            self.scalar_indx[state_key+'_mveh'] = i
    else:
        return v_x_arr

# %%
def get_timeStamps(size):
    ts = np.zeros([size, 1])
    t = 0.1
    for i in range(1, size):
        ts[i] = t
        t += 0.1
    return ts
size = 50
v_x_arr = np.array(range(size))
v_y_arr = np.ones(size)*21
f_x_arr = np.array(range(size))
v_x_arr.shape = (size,1)
v_y_arr.shape = (size,1)
f_x_arr.shape = (size,1)


episode_len = len(v_x_arr)
mini_episodes_x = []
mini_episodes_y = []
for step in range(episode_len):
    epis_i = v_x_arr[step:step+10]
    target_i = v_y_arr[step:step+10]

    episode_i_len = len(epis_i)
    ts = get_timeStamps(episode_i_len)
    epis_i = np.insert(epis_i, [0], f_x_arr[step], axis=1)
    mini_episodes_x.append(np.concatenate([ts, epis_i], axis=1))
    mini_episodes_y.append(target_i)
mini_episodes_x
# return mini_episodes_x,
mini_episodes_y
