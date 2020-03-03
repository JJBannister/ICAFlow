import numpy as np
import tensorflow as tf
import tensorflow.keras as k

from ..models.real_nvp import RealNVP

def mnist():
    x = get_training_data()

    print(x.dtype)
    real_nvp = RealNVP()
    real_nvp.build_model(x.shape[1], 1)
    real_nvp.train_model(x, n_epochs=100)

def get_training_data():
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    return x_train.reshape(x_train.shape[0], 28*28).astype(np.float32)

