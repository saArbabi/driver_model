import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
from models.core.train_eval.model_evaluation import modelEvaluate
from sklearn.model_selection import train_test_split
import numpy as np
from models.core.tf_models.utils import nll_loss
from tensorflow import keras
import random

# seed_value = 2020
# np.random.seed(seed_value)
#
# random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.set_random_seed(seed_value)



def modelTrain(config, explogs):
    X_train, X_test, y_train, y_test = build_toy_dataset()
    model = am.FFMDN(config)
    exp_id = config['exp_id']
    model.compile(loss=nll_loss(config), optimizer=model.optimizer)

    utils.updateExpstate(explogs, exp_id, 'in progress')
    validation_data=(X_test, y_test)
    history = model.fit(x=X_train, y=y_train, epochs=model.epochs_n, validation_data=validation_data,
                        verbose=0, batch_size=model.batch_n, callbacks=model.callback)
    model.saveGraph(y_test[0])
    modelEvaluate(model, validation_data, config)
    explogs[exp_id]['train_loss'] = round(history.history['loss'][-1], 1)
    explogs[exp_id]['val_loss'] = round(history.history['val_loss'][-1], 1)

    model.save(model.exp_dir+'/trained_model')
    utils.updateExpstate(explogs, config['exp_id'], 'complete')

def runSeries():
    explogs = utils.loadExplogs()
    undone_exp = utils.get_undoneExpIDs(explogs)

    for exp_id in undone_exp:
        config = utils.loadConfig(exp_id)
        history = modelTrain(config, explogs)

def build_toy_dataset(nsample=10000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.8)
