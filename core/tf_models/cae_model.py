
from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM
from datetime import datetime
from models.core.tf_models.utils import nll_loss
# from models.core.tf_models.abstract_model import  AbstractModel
class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.model_type = config['model_type']

        self.exp_dir = './models/experiments/'+config['exp_id']
        self.learning_rate = self.config['learning_rate']
        self.epochs_n = self.config['epochs_n']
        self.batch_n = self.config['batch_n']
        self.callback = self.callback_def()
        self.nll_loss = nll_loss

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
            loss = self.nll_loss(targets, predictions, self.model_type)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function
    def test_step(self, states, targets, conditions):
        predictions = self([states, conditions], training=False)
        loss = self.nll_loss(targets, predictions, self.model_type)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, states, targets, conditions):
        dataset = tf.data.Dataset.from_tensor_slices((states.astype("float32"),
                    targets.astype("float32"), conditions.astype("float32"))).batch(self.batch_n)
        return dataset

class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['model_config']['enc_units']
        self.architecture_def(config)

    def architecture_def(self, config):
        self.lstm_layers = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        _, state_h, state_c = self.lstm_layers(inputs)
        return [state_h, state_c]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.dec_units = config['model_config']['dec_units']
        self.model_type = config['model_type']
        self.architecture_def(config)

    def architecture_def(self, config):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layers = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.alphas = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long = Dense(self.components_n, name="mus_long")
        self.sigmas_long = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        if self.model_type == 'merge_policy':
            self.mus_lat = Dense(self.components_n, name="mus_lat")
            self.sigmas_lat = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
            self.rhos = Dense(self.components_n, activation=K.tanh, name="rhos")

    def call(self, inputs):
        # input[0] = conditions
        # input[1] = encoder states
        outputs, state_h, state_c = self.lstm_layers(inputs[0], initial_state=inputs[1])
        self.state = [state_h, state_c]
        alphas = self.alphas(outputs)
        mus_long = self.mus_long(outputs)
        sigmas_long = self.sigmas_long(outputs)
        if self.model_type == 'merge_policy':
            mus_lat = self.mus_lat(outputs)
            sigmas_lat = self.sigmas_lat(outputs)
            rhos = self.rhos(outputs)
            parameter_vector = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
        else:
            parameter_vector = self.pvector([alphas, mus_long, sigmas_long])

        return parameter_vector

class CAE(AbstractModel):
    def __init__(self, encoder_model, decoder_model, config):
        super(CAE, self).__init__(config)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def architecture_def(self, X):
        pass

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.encoder_model(inputs[0])
        return self.decoder_model([inputs[1], encoder_states])
