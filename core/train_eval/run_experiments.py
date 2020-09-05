
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


seed_value = 2020
np.random.seed(seed_value)

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.set_random_seed(seed_value)



# %%

import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
# %%


def modelTrain(config):
    X_train, X_test, y_train, y_test = build_toy_dataset()

    model = am.FFMDN(config)

    model.compile(loss=nll_loss(config), optimizer=model.optimizer)

    model.fit(x=X_train, y=y_train, model.epochs_n=30, validation_data=(X_test, y_test),
                        verbose=2, model.batch_n=1280, callbacks=model.callback)

    model.save(model.exp_dir+'/trained_model')


def modelEvaluate(config):
    """
    Function for evaluating the model.
    Performance metrics are:
        - nll loss, training and validation
        - RWSE
        -
    """
    model = keras.models.load_model(model.exp_dir+'/trained_model',
                                        custom_objects={'loss': nll_loss(config)})

    pass

def run_trainingSeries():
    explogs = utils.loadExplogs()
    undone_exp = utils.get_undoneExpIDs(explogs)

    for exp_id in undone_exp:
        config = utils.loadConfig(exp_id)
        utils.updateExpstate(explogs, exp_id, 'in progress')
        modelTrain(config)
        modelEvaluate(config)
        utils.updateExpstate(explogs, exp_id, 'complete')


# %%

# %%
test_variables = {'param_name':'hidden_size', 'param_values': [1,2,3]} # variables being tested
from importlib import reload
reload(utils)
reload(config_generator)

config_base = utils.loadConfigBase('baseline_test.json')
config_generator.genExpSeires(config_base, test_variables=None)

run_trainingSeries()
