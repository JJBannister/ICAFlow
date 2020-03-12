import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from .bijectors import AffineCoupling, ScaleDiag

class GIN():

    def __init__(self, 
            input_shape, 
            n_coupling_layers, 
            hidden_layer_dim=128, 
            batch_norm=False):

        self.base_distribution = tfd.MultivariateNormalDiag(
                loc=tf.zeros([input_shape]),
                scale_diag = tf.ones([input_shape]))

        self.scale_bijector = ScaleDiag(input_shape=input_shape)

        def _init_once(x, name):
            return tf.Variable(x, name=name, trainable=False)

        bijector_chain = [tfb.Identity()]
        bijector_chain.append(self.scale_bijector)

        for i in range(n_coupling_layers):
            if(batch_norm):
                bijector_chain.append(tfb.BatchNormalization())

            bijector_chain.append(
                    AffineCoupling(
                        input_shape=input_shape,
                        incompressible=True,
                        hidden_layer_dim=hidden_layer_dim,
                        name="Gin_"+str(i)))

            bijector_chain.append(tfb.Permute(permutation=_init_once( 
                np.random.permutation(input_shape).astype('int32'), 
                name='permutation')))

        self.bijector = tfb.Chain(list(bijector_chain))

        self.transformed_distribution = tfd.TransformedDistribution( 
                distribution=self.base_distribution,
                bijector=self.bijector)


    def latent_log_stddev(self): 
        pass
        #return self.scale_bijector.scale_layer.get_weights()


