import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
from models.core.preprocessing.data_obj import DataObj

import numpy as np
from models.core.tf_models.utils import nll_loss
from tensorflow import keras
import random


def modelTrain(config, explogs):
    model = am.FFMDN(config)
    x_train, y_train, x_val, y_val = DataObj(config).loadData()
    exp_id = config['exp_id']

    model.compile(loss=nll_loss(config['model_type']), optimizer=model.optimizer)

    utils.updateExpstate(explogs, exp_id, 'in progress')
    validation_data=(x_val, y_val)
    history = model.fit(x=x_train, y=y_train, epochs=model.epochs_n, validation_data=validation_data,
                        verbose=0, batch_size=model.batch_n, callbacks=model.callback)
    # modelEvaluate(model, validation_data, config)
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
