from models.core.preprocessing import data_prep
from models.core.preprocessing import data_obj
import numpy as np
import numpy as np
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
     "components_n": 5
},
"data_config": {"obs_n": 20,
                "pred_h": 4,
                "Note": "Splines////////////////////////"
},
"exp_id": "NA",
"Note": "NA"
}
data_objs =  DataObj(config).loadData()
states_train, targets_train, conditions_train, \
                            states_val, targets_val, conditions_val = data_objs

size = 0
for i in states_train.keys():
    size += states_train[i].shape[0]
size
# %%

conditions_train[3][0][0]

# %%
for i in range(0, 4):
    plt.figure()
    plt.hist(conditions_train[4][0][:,1,i], bins=125)


# %%
"""Distribution vis with sequence data
"""
def vis_dataDistribution(x):
    for i in range(len(x[0][0])):
        fig = plt.figure()
        plt.hist(x[:100000,:100000,i], bins=125)

vis_dataDistribution(targets_train)

# %%
"""Distribution vis for sinlge step data
"""
def vis_dataDistribution(x):
    for i in range(len(x[0])):
        fig = plt.figure()
        plt.hist(x[:1000000,i], bins=125)

vis_dataDistribution(states_train)


# %%
from collections import deque


obs = range(0, 100)
mveh_action = range(800, 900)
yveh_action = range(500, 600)

target_seq = []
input_seq = []
obs_seq = []
prediction_step_n = 20
obsSequence_n = 10

step_size = 3
i_reset = 0
i = 0
for chunks in range(step_size):
    prev_states = deque(maxlen=obsSequence_n)
    while i < (len(obs)):
        prev_states.append(obs[i])
        if len(prev_states) == obsSequence_n:
            obs_seq.append(np.array(prev_states))
            decoder_conditioning_seq = []
            for n in range(prediction_step_n-20):
                decoder_conditioning_seq.append([yveh_action[i+n], mveh_action[i+n]])
            input_seq.append(decoder_conditioning_seq)
            target_seq.append(list(mveh_action)[i+1:i + prediction_step_n+1])

        i += step_size
    i_reset += 1
    i = i_reset


#
# print(len(target_seq[-1]))
# input_seq[-1]

# %%
print(obs_seq[-1])
print(len(input_seq[-1]))
input_seq[-1]
input_seq[1]

# %%
def mask_history(self, obs):
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
        return obs

# %%
def get_timeStamps(size):
    ts = np.zeros([size, 1])
    t = 0.1
    for i in range(1, size):
        ts[i] = t
        t += 0.1
    return ts
size = 50
obs = np.array(range(size))
v_y_arr = np.ones(size)*21
f_states_arr = np.array(range(size))
obs.shape = (size,1)
v_y_arr.shape = (size,1)
f_states_arr.shape = (size,1)


episode_len = len(obs)
mini_episodes_x = []
mini_episodes_y = []
for step in range(episode_len):
    epis_i = obs[step:step+10]
    target_i = v_y_arr[step:step+10]

    episode_i_len = len(epis_i)
    ts = get_timeStamps(episode_i_len)
    epis_i = np.insert(epis_i, [0], f_states_arr[step], axis=1)
    mini_episodes_x.append(np.concatenate([ts, epis_i], axis=1))
    mini_episodes_y.append(target_i)
mini_episodes_x
# return mini_episodes_x,
mini_episodes_y
