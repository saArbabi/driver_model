import numpy as np
np.random.seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from datetime import datetime
from models.core.tf_models.utils import loss_merge, loss_other, covDet_mean
from tensorflow.keras.layers import Masking

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.exp_dir = './models/experiments/'+config['exp_id']
        self.optimizer = tf.optimizers.Adam(self.config['learning_rate'])
        self.batch_size = self.config['batch_size']
        self.pred_horizon = config['data_config']['pred_horizon']
        self.batch_count = None
        self.epochs_n = self.config['epochs_n']
        self.callback_def()

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'epoch_metrics')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, targs, conditions, epsilon, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            gmm_m, gmm_y = self([states, conditions, epsilon], training=True)
            covDet = covDet_mean(gmm_m)
            tf.summary.scalar('covDet_mean', covDet, step=epoch)
            tf.summary.scalar('loss_m', loss_merge(targs[0], gmm_m), step=epoch)
            tf.summary.scalar('loss_y', loss_other(targs[1], gmm_y), step=epoch)
        self.writer_2.flush()

    def schedule_sampling_def(self, data_objs):
        self.batch_count = 0
        self.mini_batch_i = 0

        for seq_len in range(3, self.pred_horizon + 1): # 3 is minimum step_n
            train_seq_data = [data_objs[0][seq_len], data_objs[1][seq_len], data_objs[2][seq_len]]
            train_ds = self.batch_data(train_seq_data)

            for states, targets, conditions in train_ds:
                targs = [targets[:, :, :2], targets[:, :, 2]]
                self.batch_count += 1

        self.batch_count *= self.epochs_n
        x = np.linspace(0, self.batch_count, self.batch_count+1)
        k = 0.9995

        # if decay_type == 'exponential':
        self.epsilons = k**x

    def train_loop(self, data_objs):
        """Covers one epoch
        """
        if not self.batch_count:
            self.schedule_sampling_def(data_objs)

        for seq_len in range(3, self.pred_horizon + 1): # 3 is minimum step_n
            train_seq_data = [data_objs[0][seq_len], data_objs[1][seq_len], data_objs[2][seq_len]]
            train_ds = self.batch_data(train_seq_data)

            for states, targets, conditions in train_ds:
                targs = [targets[:, :, :2], targets[:, :, 2]]
                self.train_step(states, targs, conditions, self.epsilons[self.mini_batch_i])
                self.mini_batch_i += 1

    def test_loop(self, data_objs, epoch):
        for seq_len in range(3, self.pred_horizon + 1):
            test_seq_data = [data_objs[0][seq_len], data_objs[1][seq_len], data_objs[2][seq_len]]
            test_ds = self.batch_data(test_seq_data)

            for states, targets, conditions in test_ds:
                targs = [targets[:, :, :2], targets[:, :, 2]]
                self.test_step(states, targs, conditions, self.epsilons[self.mini_batch_i])
        self.save_epoch_metrics(states, targs, conditions, self.epsilons[self.mini_batch_i], epoch)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targs, conditions, epsilon):
        with tf.GradientTape() as tape:
            gmm_m, gmm_y = self([states, conditions, epsilon])
            loss = loss_merge(targs[0], gmm_m) + \
                    loss_other(targs[1], gmm_y)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targs, conditions, epsilon):
        gmm_m, gmm_y = self([states, conditions, epsilon])
        loss = loss_merge(targs[0], gmm_m) + \
                loss_other(targs[1], gmm_y)

        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        st, targ, cond = [tf.cast(set, dtype='float32') for set in sets]
        dataset = tf.data.Dataset.from_tensor_slices((st, targ, cond)).batch(self.batch_size)
        return dataset
