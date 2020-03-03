from . import ICA

"""
class FFJORD(ICA):
    def __init__(self):
        self.derivative_net = self._derivative_net()
        self.bijector = tfb.FFJORD(state_time_derivative_fn=self.derivative_net)

        self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([28*28]))
        self.target = tfd.TransformedDistribution(self.base, self.bijector)

        model_in = k.layers.Input(shape=(28*28,), dtype=tf.float32)
        model_out = self.target.log_prob(model_in)

        self.model = k.models.Model(model_in, model_out)
        self.model.compile(
                optimizer = tf.optimizers.Adam(),
                loss = lambda _, log_prob: -log_prob)
    

    def _derivative_net(self):
        model_in = k.layers.Input(shape=(28*28,), dtype=tf.float32)
        x = k.layers.Dense(28*28, activation='relu')(model_in)
        x = k.layers.Dense(28*28, activation='relu')(x)
        model_out = x
        return k.models.Model(model_in, model_out)

main()

"""
