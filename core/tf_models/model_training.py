
import models.core.tf_models.abstract_model as am
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from models.core.tf_models import utils
from importlib import reload
from tensorflow import keras
import random



random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.set_random_seed(seed_value)


seed_value = 2020
np.random.seed(seed_value)

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

# load data
X_train, X_test, y_train, y_test = build_toy_dataset()
plt.scatter(X_test, y_test)
y_train[0]
plt.scatter(X_train, y_train)
plt.scatter(x, means)

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
"experiment_name": 'exp001',
"experiment_type": {"vehicle_name":'mveh', "model":"controller"}
}
model = am.FFMDN(config)
# %%

model.compile(loss=utils.nll_loss, optimizer=model.optimizer)

model.fit(x=X_train, y=y_train,epochs=3, validation_data=(X_test, y_test),
                    verbose=1, batch_size=128, callbacks=model.callback)

model.save(model.exp_dir+'/trained_model')
model = keras.models.load_model(model.exp_dir+'/trained_model',
                                    custom_objects={'nll_loss': utils.nll_loss})

y_pred = model.predict(y_test)
# %%
# %%
y_pred = model.predict(y_test)

alpha_pred, mu_pred, sigma_pred = slice_pvector(y_pred)

alpha_pred, mu_pred, sigma_pred = slice_pvector(y_pred)

y_pred.ndim
parameter_vector = y_pred[0]
tf.split(parameter_vector, 3)
len(y_pred[0])
slice_pvector(y_pred[0]) # Unpack parameter vectors

slice_pvector(y_pred)[0] # Unpack parameter vectors
get_pdf(y_pred[0]).max()

means = []
x = []
for i in range(1000):
    gm = get_pdf(y_pred[0])
    means.append(float(gm.mean()))
    x.append(X_test[i])

plt.scatter(x, means)


len(y_pred)
gm = get_pdf(y_pred[0])
gm1 = get_pdf(y_pred[1])
float(gm.mean())
