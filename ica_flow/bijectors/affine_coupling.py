import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda, Input
from tensorflow.keras import Model, regularizers, initializers
import tensorflow_probability as tfp
tfb = tfp.bijectors


class AffineCoupling(tfb.Bijector):
    def __init__(self,
                 input_shape,
                 volume_preserve=True,
                 forward_min_event_ndims=1,
                 validate_args: bool = False,
                 name="Affine_Coupling",
                 hidden_layer_dim=128):
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        assert input_shape % 2 == 0
        self.input_shape = input_shape

        nn = ScaleAndTranslateNetwork(
            input_shape // 2,
            hidden_layer_dim=hidden_layer_dim,
            volume_preserve=volume_preserve)
        
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
            n_hidden_layers=2, 
            volume_preserve=True,
            activation='elu',
            name=None):
        super().__init__()

        self.volume_preserve=volume_preserve
        self.layer_list = []

        for i in range(n_hidden_layers):
            self.layer_list.append(
                Dense(
                    hidden_layer_dim,
                    activation=activation,
                    kernel_regularizer = regularizers.l2(0.001),
                    bias_regularizer = regularizers.l2(0.001),
                    bias_initializer="zeros",
                    name="dense_{}".format(i)))

        self.log_s_layer = Dense(
            input_shape,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer = regularizers.l2(0.001),
            bias_regularizer = regularizers.l2(0.001),
            activation="tanh",
            name="log_s")

        def mean_subtract(x):
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            return tf.subtract(x, mean)

        self.mean_subtract_layer = Lambda(mean_subtract)

        self.t_layer = Dense(
            input_shape,
            kernel_initializer='zeros',
            activation="linear",
            name="t")

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        if self.volume_preserve:
            log_s = self.mean_subtract_layer(log_s)
        t = self.t_layer(y)
        return log_s, t

