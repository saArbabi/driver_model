from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from datetime import datetime
from models.core.tf_models.utils import loss_merge, loss_other, covDet_min

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.exp_dir = './models/experiments/'+config['exp_id']
        self.learning_rate = self.config['learning_rate']
        self.epochs_n = self.config['epochs_n']
        self.batch_n = self.config['batch_n']
        self.callback = self.callback_def()

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'covdet')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, targs, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            gmm_m, _, _, _ = self([states, conditions], training=True)
            covdet_min = covDet_min(gmm_m)
            tf.summary.scalar('covdet_min', covdet_min, step=epoch)
        self.writer_2.flush()

    @tf.function
    def train_step(self, states, targs, conditions, optimizer):
        with tf.GradientTape() as tape:
            gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions], training=True)
            loss = loss_merge(targs[0], gmm_m) + loss_other(targs[1], gmm_y) \
                            loss_other(targs[2], gmm_f) + loss_other(targs[3], gmm_fadj)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function
    def test_step(self, states, targs, conditions):
        gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions], training=False)
        loss = loss_merge(targs[0], gmm_m) + loss_other(targs[1], gmm_y) \
                        loss_other(targs[2], gmm_f) + loss_other(targs[3], gmm_fadj)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        a, b, c, d = sets
        a, b, c, d = a.astype("float32"), b.astype("float32"), \
                                            c.astype("float32"), d.astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((a,b,c,d)).batch(self.batch_n)
        return dataset
