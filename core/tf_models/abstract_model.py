import numpy as np
np.random.seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from datetime import datetime
from models.core.tf_models.utils import loss_merge, loss_other, covDet_min
from tensorflow.keras.layers import Masking

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.exp_dir = './models/experiments/'+config['exp_id']
        self.optimizer = tf.optimizers.Adam(self.config['learning_rate'])
        self.batch_size = config['data_config']['batch_size']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.callback_def()

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'covdet')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            gmm_m, _, _, _ = self([states, conditions], training=True)
            covdet_min = covDet_min(gmm_m)
            tf.summary.scalar('covdet_min', covdet_min, step=epoch)
        self.writer_2.flush()

    def train_loop(self, data_objs):
        for seq_len in range(3, self.pred_horizon): # 3 is minimum step_n
            train_seq_data = [data_objs[0][seq_len], data_objs[1][seq_len], data_objs[2][seq_len]]
            train_ds = self.batch_data(train_seq_data)

            for states, targets, conditions in train_ds:
                targs = [targets[:, :, :2], targets[:, :, 2], \
                                                targets[:, :, 3], targets[:, :, 4]]

                self.train_step(states, targs, conditions)

    def test_loop(self, data_objs):
        for seq_len in range(3, self.pred_horizon):
            test_seq_data = [data_objs[0][seq_len], data_objs[1][seq_len], data_objs[2][seq_len]]
            test_ds = self.batch_data(test_seq_data)

            for states, targets, conditions in test_ds:
                targs = [targets[:, :, :2], targets[:, :, 2], \
                                                targets[:, :, 3], targets[:, :, 4]]

                self.test_step(states, targs, conditions)


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targs, conditions):
        batch_shape = tf.shape(conditions)
        with tf.GradientTape() as tape:
            gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions], training=True)
            loss = loss_merge(targs[0], gmm_m, batch_shape) + \
                    loss_other(targs[1], gmm_y, batch_shape) + \
                    loss_other(targs[2], gmm_f, batch_shape) + \
                    loss_other(targs[3], gmm_fadj, batch_shape)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targs, conditions):
        batch_shape = tf.shape(conditions)
        gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions], training=False)
        loss = loss_merge(targs[0], gmm_m, batch_shape) + \
                loss_other(targs[1], gmm_y, batch_shape) + \
                loss_other(targs[2], gmm_f, batch_shape) + \
                loss_other(targs[3], gmm_fadj, batch_shape)

        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        st, targ, cond = [tf.cast(set, dtype='float32') for set in sets]
        dataset = tf.data.Dataset.from_tensor_slices((st, targ, cond)).batch(self.batch_size)
        return dataset
