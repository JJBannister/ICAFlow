import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class GIN():

    def __init__(self):
        pass

    def build_model(self, input_shape, n_coupling_layers, n_hidden_layer_dimensions):

        self.base_distribution = tfd.MultivariateNormalDiag(
                loc=tf.zeros([input_shape]),
                scale_diag = tf.ones([input_shape]))

        def _init_once(x, name):
            return tf.Variable(x, name=name, trainable=False)

        bijector_chain = []
        for i in range(n_coupling_layers):
            bijector_chain.append(tfp.bijectors.BatchNormalization())

            bijector_chain.append(tfp.bijectors.Permute(permutation=_init_once( 
                np.random.permutation(input_shape).astype('int32'), 
                name='permutation')))

            bijector_chain.append(
                    GINBijector(
                        input_shape=input_shape,
                        hidden_layer_dim=n_hidden_layer_dimensions,
                        name="Gin_"+str(i)))

        bijector = tfb.Chain(list(bijector_chain))

        self.transformed_distribution = tfd.TransformedDistribution(
            distribution=self.base_distribution,
            bijector=bijector)

        x = Input(shape=input_shape, dtype=tf.float32)
        log_prob = self.transformed_distribution.log_prob(x)

        self.model = Model(x, log_prob)

        self.model.compile(optimizer=Adam(),
                      loss=lambda _, log_prob: -log_prob)


    def train_model(self, data, n_epochs=100, lr=1e-3, batch_size=128):
        self.model.fit(
            x=data,
            y=np.zeros((data.shape[0], 0), dtype=np.float32), 
            batch_size=batch_size,
            shuffle=True,
            verbose=True)


class GINBijector(tfb.Bijector):
    def __init__(self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="GIN",
            hidden_layer_dim=256):
        super(GINBijector, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        assert input_shape % 2 == 0
        self.input_shape = input_shape

        nn = ScaleAndTranslateNetwork(
            input_shape // 2,
            hidden_layer_dim=hidden_layer_dim)

        
        x = tf.keras.Input(input_shape // 2)
        log_s, t = nn(x)
        self.scale_and_translate_network = Model(x, [log_s, t], name=self.name + "/scale_and_translate")

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        log_s, t = self.scale_and_translate_network(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        log_s, t = self.scale_and_translate_network(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        _, x_b = tf.split(x, 2, axis=-1)
        log_s, t = self.scale_and_translate_network(x_b)
        return tf.reduce_sum(log_s)


class ScaleAndTranslateNetwork(Layer):

    def __init__(self,
            input_shape,
            hidden_layer_dim, 
            activation='relu',
            name=None):
        super(ScaleAndTranslateNetwork, self).__init__()

        self.layer_list = []
        for i in range(2):
            self.layer_list.append(
                Dense(
                    hidden_layer_dim,
                    activation=activation,
                    name="dense_{}".format(i)))

        log_s_shape = input_shape - 1

        self.log_s_layer = Dense(
            log_s_shape,
            kernel_initializer="zeros",
            name="log_s")

        def volume_preserving_log_s(x):
            neg_sum = tf.reduce_sum(x, axis=-1, keepdims=True)
            return K.concatenate([x,neg_sum], axis=-1)

        self.vp_log_s_layer = Lambda(volume_preserving_log_s)

        self.t_layer = Dense(
            input_shape,
            kernel_initializer="zeros",
            activation="tanh",
            name="t")

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        vp_log_s = self.vp_log_s_layer(log_s)
        t = self.t_layer(y)
        return vp_log_s, t
