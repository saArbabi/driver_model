import models.core.tf_models.abstract_model as am
from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
from models.core.preprocessing.data_obj import DataObj
import tensorflow as tf
import os
import numpy as np
from models.core.tf_models.utils import nll_loss
from tensorflow import keras
import random
from datetime import datetime

def modelTrain(exp_id, explogs):
    config = utils.loadConfig(exp_id)
    model = am.FFMDN(config)
    optimizer = tf.optimizers.Adam(model.learning_rate)

    # for more on checkpointing model see: https://www.tensorflow.org/guide/checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model.exp_dir+'/model_dir', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    x_train, y_train, x_val, y_val = DataObj(config).loadData()
    train_ds = model.batch_data(x_train, y_train)
    test_ds = model.batch_data(x_val, y_val)

    write_graph = 'False'
    batch_i = 0
    start_epoch = explogs[exp_id]['epoch']
    end_epoch = start_epoch +  model.epochs_n

    for epoch in range(start_epoch, end_epoch):
        for xs, targets in train_ds:
            if write_graph == 'True':
                graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
                tf.summary.trace_on(graph=True, profiler=False)
                model.train_step(xs, targets, optimizer)
                with graph_write.as_default():
                    tf.summary.trace_export(name='graph', step=0)
                write_graph = 'False'
            else:
                model.train_step(xs, targets, optimizer)
            model.save_batch_metrics(xs, targets, batch_i)
            batch_i += 1

        for xs, targets in test_ds:
            model.test_step(xs, targets)

        model.save_epoch_metrics(epoch)
        utils.updateExpstate(model, explogs, exp_id, 'in progress')

        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()

    utils.updateExpstate(model, explogs, exp_id, 'complete')
    # modelEvaluate(model, validation_data, config)
    # model.save(model.exp_dir+'/model_dir',save_format='tf')

def runSeries(exp_ids=None):
    explogs = utils.loadExplogs()

    if exp_ids:
        for exp_id in exp_ids:
            modelTrain(exp_id, explogs)
    else:
        # complete any undone experiments
        undone_exp = utils.get_undoneExpIDs(explogs)
        for exp_id in undone_exp:
            modelTrain(exp_id, explogs)
