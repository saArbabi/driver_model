from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
from models.core.preprocessing.data_obj import DataObj
import tensorflow as tf
from models.core.tf_models.cae_model import CAE

def modelTrain(exp_id, explogs):
    config = utils.loadConfig(exp_id)

    # (1) Load model and setup checkpoints
    model = CAE(config, model_use='training')

    # for more on checkpointing model see: https://www.tensorflow.org/guide/checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model) # no need for optimizer for now
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model.exp_dir+'/model_dir', max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    start_epoch = explogs[exp_id]['epoch']
    end_epoch = start_epoch + config['model_config']['epochs_n']

    # (2) Load data
    data_objs = DataObj(config).loadData()

    # (3) Run experiment
    write_graph = 'True'
    for epoch in range(start_epoch, end_epoch):
        model.dec_model.model_use = 'training'
        model.train_loop(data_objs[0:3])
        model.dec_model.model_use = 'validating'
        model.test_loop(data_objs[3:], epoch)
        utils.updateExpstate(model, explogs, exp_id, 'in progress')

        ckpt.step.assign_add(1)
        if int(ckpt.step) % 5 == 0:
            save_path = manager.save()

    utils.updateExpstate(model, explogs, exp_id, 'complete')
    # modelEvaluate(model, validation_data, config)

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
