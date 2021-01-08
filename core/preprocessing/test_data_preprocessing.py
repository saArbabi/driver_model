from models.core.preprocessing import data_prep
from models.core.preprocessing import data_obj
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
"data_config": {"obs_n": 10,
                "pred_step_n": 7,
                "step_size": 3,
                "Note": "lat/long motion not considered jointly"
                # "Note": "jerk as target"
},
"exp_id": "NA",
"Note": "NA"
}
data_objs =  DataObj(config).loadData()
states_train, targets_train, conditions_train, \
                            states_val, targets_val, conditions_val = data_objs


# %%
states_train[4][10][-1]
states_train[4][10].shape
conditions_train[4][2][10]
.shape
[1][10]
targets_train[4][1][10]

size = 0
for i in targets_train.keys():
    size += states_train[i].shape[0]
size

# %%
states_train[4].shape
# %%
plt.plot(states_train[4][108, :, 3])
plt.plot(states_train[4][108, :, 3]+np.random.normal(0, 0.01, 20))
# %%
for i in range(0, 5):
    plt.figure()
    plt.hist(conditions_val[4][i][:,0,:], bins=125)
# %%
for i in range(0, 5):
    plt.figure()
    plt.hist(targets_train[20][i][:,0,:], bins=125)

# %%
for i in range(0, 17):
    plt.figure()
    plt.hist(states_train[4][0:10000,1,i], bins=125)
# %%
