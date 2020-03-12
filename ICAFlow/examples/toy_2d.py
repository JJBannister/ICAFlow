import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

from ..core.gin import GIN

def toy_2d():

    # Data
    x = get_training_data(data_type=None, n_samples=5000, stddev=[2,4])
    viz_2d(x, "Training Samples")

    # Flow
    flow = GIN(
            input_shape=x.shape[-1], 
            n_coupling_layers=0, 
            hidden_layer_dim=8,
            batch_norm=False)

    print(flow.transformed_distribution.trainable_variables)


    viz_2d(flow.base_distribution.sample(5000), 
            "Base Samples")
    viz_2d(flow.transformed_distribution.sample(5000), 
            "Target samples (pre-train)")

    print(flow.transformed_distribution.log_prob(x))


    # Train
    i = Input(shape=x.shape[-1])
    log_prob = flow.transformed_distribution.log_prob(i)
    model = Model(i, log_prob) 
    model.compile(optimizer=Adam(lr=1e-1),
            loss=lambda _, log_prob: -log_prob)

    print(model.summary())

    model.fit( 
            x=x, 
            y=np.zeros((x.shape[0], 0), dtype=np.float32), 
            epochs=10, 
            batch_size=128, 
            shuffle=True, 
            verbose=True)
            
    # Test
    print("Latent Variable Log Stddevs: ", flow.latent_log_stddev())

    viz_2d(flow.transformed_distribution.sample(5000), 
            "Transformed samples (Post-train)")



def viz_2d(data, title=""):
    sns.scatterplot(x=data[:,0], y=data[:,1])
    plt.title(title)
    plt.show()


def get_training_data(data_type=None, n_samples=1000, stddev=[1.,1.]):
    x = np.zeros([n_samples,2], dtype=np.float32)
    for i in range(n_samples):
        x[i,0] = np.random.normal(0, stddev[0])
        x[i,1] = np.random.normal(0, stddev[1])

        if data_type == "squared":
            x[i,1] = x[i,1] + x[i,0]**2 - 2

        elif data_type == "cubed":
            x[i,1] = x[i,1] + x[i,0]**3

    return x
