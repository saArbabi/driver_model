
from numpy.random import seed # keep this at top
seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM, Masking
from datetime import datetime
from models.core.tf_models.utils import loss_merge, loss_yield, covDet_min, get_pdf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# from models.core.tf_models.abstract_model import  AbstractModel
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

    def save_epoch_metrics(self, states, targets_m, targets_y, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            gmm_m, _ = self([states, conditions], training=True)
            covdet_min = covDet_min(gmm_m)
            tf.summary.scalar('covdet_min', covdet_min, step=epoch)
        self.writer_2.flush()

    @tf.function
    def train_step(self, states, targets_m, targets_y, conditions, optimizer):
        with tf.GradientTape() as tape:
            gmm_m, gmm_y = self([states, conditions], training=True)
            loss = loss_merge(targets_m, gmm_m) + loss_yield(targets_y, gmm_y)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function
    def test_step(self, states, targets_m, targets_y, conditions):
        gmm_m, gmm_y = self([states, conditions], training=False)
        loss = loss_merge(targets_m, gmm_m) + loss_yield(targets_y, gmm_y)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        a, b, c, d = sets
        a, b, c, d = a.astype("float32"), b.astype("float32"), \
                                            c.astype("float32"), d.astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((a,b,c,d)).batch(self.batch_n)
        return dataset

class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = config['model_config']['enc_units']
        self.architecture_def()

    def architecture_def(self):
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
        self.pred_horizon = config['data_config']['pred_horizon']

        self.architecture_def()

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layers = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.masking = Masking(mask_value=0., input_shape=(self.pred_horizon, None))
        """Merger vehicle
        """
        self.alphas_m = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_m = Dense(self.components_n, name="mus_long")
        self.sigmas_long_m = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_m = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_m = Dense(self.components_n, activation=K.exp, name="sigmas_lat")
        self.rhos_m = Dense(self.components_n, activation=K.tanh, name="rhos")
        """Yielder vehicle
        """
        self.alphas_y = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.mus_long_y = Dense(self.components_n, name="mus_long")
        self.sigmas_long_y = Dense(self.components_n, activation=K.exp, name="sigmas_long")

    def call(self, inputs):
        # input[0] = conditions
        # input[1] = encoder states
        conditions = self.masking(inputs[0])
        outputs, state_h, state_c = self.lstm_layers(conditions, initial_state=inputs[1])
        self.state = [state_h, state_c]
        """Merger vehicle
        """
        alphas = self.alphas_m(outputs)
        mus_long = self.mus_long_m(outputs)
        sigmas_long = self.sigmas_long_m(outputs)
        mus_lat = self.mus_lat_m(outputs)
        sigmas_lat = self.sigmas_lat_m(outputs)
        rhos = self.rhos_m(outputs)
        param_vec_m = self.pvector([alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos])
        gmm_m = get_pdf(param_vec_m, 'merge_vehicle')
        """Yielder vehicle
        """
        alphas = self.alphas_y(outputs)
        mus_long = self.mus_long_y(outputs)
        sigmas_long = self.sigmas_long_y(outputs)
        param_vec_y = self.pvector([alphas, mus_long, sigmas_long])
        gmm_y = get_pdf(param_vec_y, 'yield_vehicle')

        return gmm_m, gmm_y

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
