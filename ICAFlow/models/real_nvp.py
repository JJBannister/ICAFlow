from .ica import ICA

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class RealNVP(ICA):

    def __init__(self):
        super().__init__()


    def build_model(self, n_dimensions, n_coupling_layers, n_hidden_layer_dimensions):
        self.base_distribution = tfd.MultivariateNormalDiag(
                loc=tf.zeros([n_dimensions]),
                scale_diag = tf.ones([n_dimensions]))

        def _init_once(x, name):
            return tf.Variable(x, name=name, trainable=False)

        bijector_chain = []
        for i in range(n_coupling_layers):

            # random permutation
            bijector_chain.append(tfp.bijectors.Permute(permutation=_init_once( 
                np.random.permutation(n_dimensions).astype('int32'), 
                name='permutation')))

            # affine coupling layer
            bijector_chain.append(
                    tfb.RealNVP(
                        num_masked=n_dimensions//2,
                        shift_and_log_scale_fn=self.coupling_nn(n_hidden_layer_dimensions)))

            # batch norm
            # bijector_chain.append(tfp.bijectors.BatchNormalization())

        bijector = tfb.Chain(list(reversed(bijector_chain)))

        self.transformed_distribution = tfd.TransformedDistribution(
            distribution=self.base_distribution,
            bijector=bijector)


    def coupling_nn(self, hidden_layer_dimensions):
        return tfb.real_nvp_default_template(hidden_layers=
                [hidden_layer_dimensions, hidden_layer_dimensions])


    def train_model(self, data, lr=1e-3, n_epochs=300):
        @tf.function
        def loss(x):
            return -tf.reduce_mean(self.transformed_distribution.log_prob(x))

        optimizer = tf.optimizers.Adam(learning_rate=lr)
        log = tf.summary.create_file_writer('checkpoints')
        avg_loss = tf.keras.metrics.Mean(name='neg_log_prob', dtype=tf.float32)

        for epoch in range(n_epochs):
            for x in data:
                    with tf.GradientTape() as tape:
                        neg_log_prob = loss(x)

                    grads = tape.gradient(neg_log_prob, 
                        self.transformed_distribution.trainable_variables)

                    optimizer.apply_gradients(zip(grads, 
                        self.transformed_distribution.trainable_variables))

                    avg_loss.update_state(neg_log_prob)

                    if tf.equal(optimizer.iterations % 100, 0):
                        with log.as_default():
                            tf.summary.scalar("neg_log_prob", avg_loss.result(), step=optimizer.iterations)
                            print("Step {} Loss {:.6f}".format( optimizer.iterations, avg_loss.result()))
                            avg_loss.reset_states()



    def get_transformed_distribution(self):
        return self.transformed_distribution

