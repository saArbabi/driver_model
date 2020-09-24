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

    batch_i = tf.constant([0], dtype='int64')
    write_graph = 'no'

    for epoch in tf.range(model.epochs_n, dtype='int64'):
        for xs, targets in train_ds:
            if write_graph == 'yes':
                graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
                tf.summary.trace_on()
                model.train_step(xs, targets)
                with graph_write.as_default():
                    tf.summary.trace_export(name='graph', step=0)
                write_graph == 'no'
            else:
                model.train_step(xs, targets)
            model.save_batch_metrics(xs, targets, batch_i)
            batch_i += 1

        for xs, targets in test_ds:
            model.test_step(xs, targets)

        model.save_epoch_metrics(epoch)

    # modelEvaluate(model, validation_data, config)
    explogs[exp_id]['train_loss'] = round(model.train_loss.result().numpy().item(), 2)
    explogs[exp_id]['val_loss'] = round(model.test_loss.result().numpy().item(), 2)

    utils.updateExpstate(explogs, exp_id, 'complete')


def runSeries():
    explogs = utils.loadExplogs()
    undone_exp = utils.get_undoneExpIDs(explogs)

    for exp_id in undone_exp:
        config = utils.loadConfig(exp_id)
        modelTrain(config, explogs)
