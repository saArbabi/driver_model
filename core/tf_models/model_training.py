
import models.core.tf_models.abstract_model as am
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from models.core.tf_models import utils
from importlib import reload
from tensorflow import keras
import random


seed_value = 2020
np.random.seed(seed_value)

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.set_random_seed(seed_value)



# %%
reload(am)
reload(utils)
config = {
 "model_config": {
    "learning_rate": 1e-2,
    "hidden_size": 2,
    "components_n": 4
},
"data_config": {
    "step_size": 3,
    "sequence_length": 5,
    "features": ['vel', 'pc','gap_size', 'dx', 'act_long_p', 'act_lat_p','lc_type'],
    "history_drop": {"percentage":0, "vehicle":'mveh'},
    "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx',
                                'act_long_p', 'act_lat_p', 'act_long', 'act_lat']},
    "scaler_path": './driver_model/experiments/scaler001'
},
"exp_name": 'exp001',
"exp_type": {"target_name":'yveh', "model":"controller"}
}
model = am.FFMDN(config)

model.compile(loss=utils.nll_loss(config), optimizer=model.optimizer)

model.fit(x=X_train, y=y_train,epochs=3, validation_data=(X_test, y_test),
                    verbose=1, batch_size=128, callbacks=model.callback)

model.save(model.exp_dir+'/trained_model')
model = keras.models.load_model(model.exp_dir+'/trained_model',
                                    custom_objects={'loss': utils.nll_loss(config)})


# %%


# %%
 
