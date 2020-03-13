import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda, Input
from tensorflow.keras import Model, regularizers, initializers
import tensorflow_probability as tfp
tfb = tfp.bijectors


class ScaleDiag(tfb.Bijector):
    def __init__(self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="Scale_Diag"):
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        x = Input([None])
        self.log_s_layer = TrainableVector(input_shape)
        log_s = self.log_s_layer(x)
        self.log_s_model = Model(x, log_s, name=self.name + "/log_s", trainable=True)

    def _forward(self, x):
        log_s = self.log_s_model(x)
        s = tf.exp(log_s)
        return x*s

    def _inverse(self, y):
        log_s = self.log_s_model(y)
        s = tf.exp(log_s)
        return y/s

    def _forward_log_det_jacobian(self, x):
        log_s = self.log_s_model(x)
        return tf.reduce_sum(log_s)


class TrainableVector(Layer):
    def __init__(self, output_dim):
       self.output_dim = output_dim
       super().__init__()

    def build(self, input_shape):
       self.kernel = self.add_weight(
               name='kernel',
               shape=self.output_dim,
               initializer='zeros', 
               trainable=True)

    def call(self, inputs):
       return tf.multiply(self.kernel, tf.ones(self.output_dim))

    def compute_output_shape(self):
       return self.output_dim

