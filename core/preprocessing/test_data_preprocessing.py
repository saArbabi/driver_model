from models.core.preprocessing import data_prep
from models.core.preprocessing import data_obj
import numpy as np

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
},
"exp_id": "NA",
"model_type": "merge_policy",
"Note": "NA"
}

x_train, y_train, x_val, y_val = DataObj(config).loadData()
x_train[-1][0]
x_train[-1][1]

len(x_val[0])
y_train[0]
y_train[1]
# %%

def vis_dataDistribution(x):
    for i in range(len(x[0])):
        fig = plt.figure()
        plt.hist(x[:,i], bins=125)

vis_dataDistribution(x_train)
# %%
from collections import deque

v_x_arr = range(100)
v_y_arr = range(100)
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

def ch(x):
    pass

a = 2
a = ch(a)
a
