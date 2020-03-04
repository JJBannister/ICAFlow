import numpy as np
import tensorflow as tf
import tensorflow.keras as k

from ..models.gin import GIN

def mnist():
    x = get_training_data()

def get_training_data():
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    return x_train.reshape(x_train.shape[0], 28*28).astype(np.float32)

