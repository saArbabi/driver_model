import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
from models.core.train_eval import config_generator

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from models.core.tf_models.utils import nll_loss
from importlib import reload
from tensorflow import keras
import random
import json
reload(am)

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


X_train, X_test, y_train, y_test = build_toy_dataset()

model = am.FFMDN(config)

model.compile(loss=nll_loss(config), optimizer=model.optimizer)

model.fit(x=X_train, y=y_train,epochs=30, validation_data=(X_test, y_test),
                    verbose=2, batch_size=1280, callbacks=model.callback)

model.save(model.exp_dir+'/trained_model')
