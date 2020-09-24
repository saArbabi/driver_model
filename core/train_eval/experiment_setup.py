import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
from models.core.preprocessing.data_obj import DataObj
import tensorflow as tf

import numpy as np
from models.core.tf_models.utils import nll_loss
from tensorflow import keras
import random
from datetime import datetime

def modelTrain(config, explogs):
    model = am.FFMDN(config)

    x_train, y_train, x_val, y_val = DataObj(config).loadData()
    exp_id = config['exp_id']

    train_ds = model.batch_data(x_train, y_train)
    test_ds = model.batch_data(x_val, y_val)

    utils.updateExpstate(explogs, exp_id, 'in progress')
    i = tf.constant([0], dtype='int64')

    for epoch in tf.range(1, dtype='int64'):
    # for epoch in tf.range(model.epochs_n, dtype='int64'):
        # Reset the metrics at the start of the next epoch
        for xs, targets in train_ds:
            model.train_step(xs, targets)
            # model.save_graph()
            model.save_batch_metrics(xs, targets, i, metric_name='train_loss_batch')
            model.save_batch_metrics(xs, targets, i, metric_name='cov_det')
            i += 1
        model.save_epoch_metrics(xs, targets, epoch, metric_name='train_loss')

        for xs, targets in test_ds:
            model.test_step(xs, targets)
        model.save_epoch_metrics(xs, targets, epoch, metric_name='validation_loss')

    # modelEvaluate(model, validation_data, config)
    explogs[exp_id]['train_loss'] = round(model.train_loss.result().numpy().item(), 2)
    explogs[exp_id]['val_loss'] = round(model.test_loss.result().numpy().item(), 2)
    # model.save(model.exp_dir+'/trained_model')
    utils.updateExpstate(explogs, exp_id, 'complete')


def runSeries():
    explogs = utils.loadExplogs()
    undone_exp = utils.get_undoneExpIDs(explogs)

    for exp_id in undone_exp:
        config = utils.loadConfig(exp_id)
        history = modelTrain(config, explogs)
