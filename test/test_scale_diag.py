import numpy as np
import unittest
import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('../')

from ica_flow.bijectors.scale_diag import ScaleDiag, TrainableVector


class TestScaleDiag(unittest.TestCase):

    def setUp(self):
        self.shape = 2
        self.bijector = ScaleDiag(self.shape)
        self.vec = TrainableVector(self.shape)


    def test_trainable_vector(self):
        x = tf.keras.Input([None])
        y = self.vec(x)
        model = tf.keras.Model(x, y, name="trainable_vector_test")
        model.summary()
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss="mean_squared_error")

        model.fit(
            x=np.zeros([128]),
            y=np.ones([128,self.shape]),
            epochs=10,
            batch_size=128,
            verbose=True)

    def test_bijector_forward(self):
        x = tf.keras.Input(self.shape)
        y = self.bijector.forward(x)
        model = tf.keras.Model(x, y, name="forward_test")
        model.summary()
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss="mean_squared_error")

        sample = np.zeros([128,self.shape])
        model.fit(
            x=sample,
            y=sample,
            epochs=1,
            batch_size=128,
            verbose=True)


    def test_bijector_log_prob(self):
        flow = tfp.distributions.TransformedDistribution(
            event_shape=[self.shape],
            distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
            bijector=self.bijector)

        x = tf.keras.Input(self.shape)
        log_prob = flow.log_prob(x)
        model = tf.keras.Model(x, log_prob, name="log_prob_test")
        model.summary()
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss="mean_squared_error")

        model.fit(
            x=np.zeros([128, self.shape]),
            y=np.zeros([128]),
            epochs=1,
            batch_size=128,
            verbose=True)


if __name__ == '__main__':

    unittest.main()




