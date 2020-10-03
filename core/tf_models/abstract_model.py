from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from datetime import datetime
from models.core.tf_models.utils import nll_loss, varMin

# %%
class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.model_type = config['model_type']

        self.exp_dir = './models/experiments/'+config['exp_id']
        self.learning_rate = self.config['learning_rate']
        self.epochs_n = self.config['epochs_n']
        self.batch_n = self.config['batch_n']
        self.components_n = self.config['components_n'] # number of Mixtures
        self.callback = self.callback_def()

    def architecture_def(self, X):
        raise NotImplementedError()

    def callback_def(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'covdet')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, targets, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            predictions = self([states, conditions], training=True)
            covdet_min = covDet_min(predictions, self.model_type)
            tf.summary.scalar('covdet_min', covdet_min, step=epoch)
        self.writer_2.flush()

    @tf.function
    def train_step(self, states, targets, conditions, optimizer):
        with tf.GradientTape() as tape:
            predictions = self([states, conditions], training=True)
            loss = self.nll_loss(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function
    def test_step(self, states, targets, conditions):
        predictions = self([states, conditions], training=False)
        loss = self.nll_loss(targets, predictions)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, states, targets, conditions):
        dataset = tf.data.Dataset.from_tensor_slices((states.astype("float32"),
                    targets.astype("float32"), conditions.astype("float32"))).batch(self.batch_n)
        return dataset
