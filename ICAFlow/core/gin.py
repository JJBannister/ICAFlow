import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from .bijectors import GIN, ScaleDiag


def gin_flow(input_shape, n_coupling_layers, n_hidden_layer_dimensions, batch_norm=False):

    base_distribution = tfd.MultivariateNormalDiag(
            loc=tf.zeros([input_shape]),
            scale_diag = tf.ones([input_shape]))

    def _init_once(x, name):
        return tf.Variable(x, name=name, trainable=False)

    bijector_chain = []
    bijector_chain.append(ScaleDiag(input_shape=input_shape))

    for i in range(n_coupling_layers):
        if(batch_norm):
            bijector_chain.append(tfb.BatchNormalization())

        bijector_chain.append(
                GIN(
                    input_shape=input_shape,
                    hidden_layer_dim=n_hidden_layer_dimensions,
                    name="Gin_"+str(i)))

        bijector_chain.append(tfb.Permute(permutation=_init_once( 
            np.random.permutation(input_shape).astype('int32'), 
            name='permutation')))

    bijector = tfb.Chain(list(bijector_chain))

    return tfd.TransformedDistribution(
        distribution=base_distribution,
        bijector=bijector)


