import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
from models.core.train_eval.model_evaluation import modelEvaluate
import numpy as np
from models.core.tf_models.utils import nll_loss
from tensorflow import keras
import random


def modelTrain(config, explogs):
    X_train, X_test, y_train, y_test = build_toy_dataset()
    model = am.FFMDN(config)
    exp_id = config['exp_id']
    model.compile(loss=nll_loss(config), optimizer=model.optimizer)

    utils.updateExpstate(explogs, exp_id, 'in progress')
    validation_data=(X_test, y_test)
    history = model.fit(x=X_train, y=y_train, epochs=model.epochs_n, validation_data=validation_data,
                        verbose=0, batch_size=model.batch_n, callbacks=model.callback)
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
