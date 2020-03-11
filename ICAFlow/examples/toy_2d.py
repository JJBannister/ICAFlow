import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

from ..core.gin import gin_flow

def toy_2d():

    # Data
    x = get_training_data("Small Shifted Normal", 5000)
    viz_2d(x, "Training Samples")

    # Flow
    flow = gin_flow( x.shape[-1], 12, 8)
    print(flow.batch_shape)

    viz_2d(flow.distribution.sample(5000), 
            "Base samples")
    viz_2d(flow.sample(5000), 
            "Transformed samples (Pre-train)")

    # Train
    i = Input(shape=x.shape[-1], dtype=tf.float32)
    log_prob = flow.log_prob(i)
    model = Model(i, log_prob)
    model.compile(optimizer=Adam(lr=1e-1),
            loss=lambda _, log_prob: -log_prob)

    model.summary()

    model.fit( 
            x=x, 
            y=np.zeros((x.shape[0], 0), dtype=np.float32), 
            epochs=10, 
            batch_size=128, 
            shuffle=True, 
            verbose=True)

    # Test
    viz_2d(flow.sample(5000), 
            "Transformed samples (Post-train)")


def viz_2d(data, title=""):
    sns.scatterplot(x=data[:,0], y=data[:,1])
    plt.title(title)
    plt.show()


def get_training_data(data_type, n_samples):
    x = np.zeros([n_samples,2], dtype=np.float32)

    if data_type == "Small Shifted Normal":
        for i in range(n_samples):
            x[i,0] = np.random.normal(0.5, 0.4)
            x[i,1] = np.random.normal(0, 0.2)

    elif data_type == "Ring":
        for i in range(n_samples):
            r = np.random.normal(1,0.1)
            theta = np.random.uniform(0,2*np.pi)
            x[i,0] = r*np.cos(theta)
            x[i,1] = r*np.sin(theta)

    elif data_type == "Moons":
        for i in range(n_samples):
            if np.random.choice([False, True]):
                theta = np.random.uniform(0,np.pi)
                shift = 0.5
            else:
                theta = np.random.uniform(np.pi,2*np.pi)
                shift = -0.5
                
            r = np.random.normal(1,0.1)
            x[i,0] = shift + r*np.cos(theta)
            x[i,1] = r*np.sin(theta)

    else:
        print("Invalid Data Type")

    return x
