import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
from models.core.train_eval import config_generator

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from importlib import reload
from tensorflow import keras
import random
import json
module = models.core.tf_models.utils
from keras.callbacks import History
import warnings
warnings.filterwarnings("always")
history = History()
from models.core.tf_models.utils import nll_loss, get_predictionMean
from  models.core.tf_models import  utils
reload(utils)

def build_toy_dataset(nsample=10000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.8)

# load data
X_train, X_test, y_train, y_test = build_toy_dataset()
plt.scatter(X_test, y_test)
plt.scatter(X_train, y_train)


config_base = utils.loadConfigBase('baseline_test.json')
config_generator.genExpSeires(config_base, test_variables=None)

X_train, X_test, y_train, y_test = build_toy_dataset()

model = am.FFMDN(config_base)

model.compile(loss=nll_loss(config_base), optimizer=model.optimizer)

history = model.fit(x=X_train, y=y_train,epochs=3, validation_data=(X_test, y_test),
                    verbose=2, batch_size=1280, callbacks=model.callback)
print(history.history.keys())
print(history.history['loss'])

model.save(model.exp_dir+'/trained_model')

predictions = model.predict(y_test)
mean_values = get_predictionMean(predictions, config_base)
plt.scatter(X_test, y_test)
plt.scatter(X_test[0:500], mean_values[0:500])
