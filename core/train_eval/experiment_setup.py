from models.core.train_eval import utils
# from models.core.train_eval.model_evaluation import modelEvaluate
from models.core.preprocessing.data_obj import DataObj
import tensorflow as tf
from models.core.tf_models.cae_model import CAE

def modelTrain(exp_id, explogs):
    config = utils.loadConfig(exp_id)

    # (1) Load model and setup checkpoints
    model = CAE(config, model_use='training')
    optimizer = tf.optimizers.Adam(model.learning_rate)

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
    end_epoch = start_epoch + model.epochs_n

    # (2) Load data
    data =  DataObj(config).loadData()
    train_ds = model.batch_data(data[0:3])
    test_ds = model.batch_data(data[3:])

    # (3) Run experiment
    write_graph = 'True'
    for epoch in range(start_epoch, end_epoch):
        for states, targets, conditions in train_ds:

            if write_graph == 'True':
                graph_write = tf.summary.create_file_writer(model.exp_dir+'/logs/')
                tf.summary.trace_on(graph=True, profiler=False)
                model.train_step(states, [targets[:, :, :2], targets[:, :, 2], targets[:, :, 3],
                                            targets[:, :, 4]], conditions, optimizer)
                with graph_write.as_default():
                    tf.summary.trace_export(name='graph', step=0)
                write_graph = 'False'
            else:
                model.train_step(states, [targets[:, :, :2], targets[:, :, 2], targets[:, :, 3],
                                            targets[:, :, 4]], conditions, optimizer)

        for states, targets, conditions in test_ds:
            model.test_step(states, [targets[:, :, :2], targets[:, :, 2], targets[:, :, 3],
                                                        targets[:, :, 4]], conditions)

        model.save_epoch_metrics(states, conditions, epoch)
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
