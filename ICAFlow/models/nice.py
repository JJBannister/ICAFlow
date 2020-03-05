import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer, Dense, Lambda, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class NICE():

    def __init__(self):
        pass

    def build_model(self, input_shape, n_coupling_layers, n_hidden_layer_dimensions):
        self.base_distribution = tfd.MultivariateNormalDiag(
                loc=tf.zeros([input_shape]),
                scale_diag = tf.ones([input_shape]))

        bijector_chain = []

        self.scale_values = tf.Variable(tf.ones([input_shape]), trainable=True)
        self.scale_bijector = tfb.AffineLinearOperator(
                shift=None,
                scale=tf.linalg.LinearOperatorDiag(self.scale_values))

        bijector_chain.append(self.scale_bijector)

        def _init_once(x, name):
            return tf.Variable(x, name=name, trainable=False)

        for i in range(n_coupling_layers):
            bijector_chain.append(tfb.BatchNormalization())

            bijector_chain.append(tfb.Permute(permutation=_init_once( 
                np.random.permutation(input_shape).astype('int32'), 
                name='permutation')))

            bijector_chain.append(
                    NICEBijector(
                        input_shape=input_shape,
                        hidden_layer_dim=n_hidden_layer_dimensions,
                        name="Nice_"+str(i)))

        bijector = tfb.Chain(list(bijector_chain))

        self.transformed_distribution = tfd.TransformedDistribution(
            distribution=self.base_distribution,
            bijector=bijector)

        x = Input(shape=input_shape, dtype=tf.float32)
        log_prob = self.transformed_distribution.log_prob(x)

        self.model = Model(x, log_prob)

        self.model.compile(optimizer=Adam(),
                      loss=lambda _, log_prob: -log_prob)

        self.model.summary()
        plot_model(self.model, to_file='NICE_model.png', 
                show_shapes=True, show_layer_names=True)


    def train_model(self, data, n_epochs=100, lr=1e-3, batch_size=128):
        self.model.fit(
            x=data,
            y=np.zeros((data.shape[0], 0), dtype=np.float32), 
            epochs=n_epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=True)


class NICEBijector(tfb.Bijector):
    def __init__(self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="NICE",
            hidden_layer_dim=256):
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        assert input_shape % 2 == 0
        self.input_shape = input_shape

        nn = self.TranslateNetwork(
            input_shape // 2,
            hidden_layer_dim=hidden_layer_dim)

        x = tf.keras.Input(input_shape // 2)
        t = nn(x)

        self.translate_network = Model(x, t, name=self.name + "/translate")

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        t = self.translate_network(x_b)
        y_a = x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        t = self.translate_network(y_b)
        x_a = y_a - t
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., dtype=x.dtype)


    class TranslateNetwork(Layer):
        def __init__(self,
                input_shape,
                hidden_layer_dim, 
                activation='relu',
                name=None):
            super().__init__()

            self.layer_list = []
            for i in range(2):
                self.layer_list.append(
                    Dense(
                        hidden_layer_dim,
                        activation=activation,
                        name="dense_{}".format(i)))

            self.t_layer = Dense(
                input_shape,
                kernel_initializer="zeros",
                activation="tanh", #?
                name="t")

        def call(self, x):
            y = x
            for layer in self.layer_list:
                y = layer(y)
            t = self.t_layer(y)
            return t



