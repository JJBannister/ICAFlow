import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../')
from ica_flow.affine_coupling_ica import AffineCouplingICA as ICA

def toy_2d():

    # Synthetic Data
    #x = get_training_data(data_transformation='None', n_samples=500, stddev=[3, 0.3])
    x = get_training_data(data_transformation='sinusoid', n_samples=5000, stddev=[3, 0.1])
    #x = get_training_data(data_transformation='square', n_samples=5000, stddev=[1, 0.2])

    # Flow
    ica = ICA(
            input_shape=x.shape[-1], 
            n_coupling_layers=4,
            hidden_layer_dim=8)

    viz_2d(ica.transformed_distribution.sample(5000), data2=x,
           title="Train + Target samples (Pre-train)")

    # Train
    i = Input(shape=x.shape[-1])

    log_prob = ica.transformed_distribution.log_prob(i)
    model = Model(i, log_prob) 
    model.compile(optimizer=Adam(lr=1e-2),
            loss=lambda _, log_prob: -log_prob)

    model.fit( 
            x=x, 
            y=np.zeros((x.shape[0], 0), dtype=np.float32), 
            epochs=300,
            batch_size=256,
            shuffle=True)

    print("Latent Variable Stddevs: ", ica.latent_stddev())

    viz_2d(ica.transformed_distribution.sample(5000), data2=x,
            title="Train + Target samples (Post-train)")



def viz_2d(data1, data2=None, title=""):
    plt.scatter(x=data1[:,0], y=data1[:,1],c='red')
    if not data2 is None:
        plt.scatter(x=data2[:,0], y=data2[:,1],c='blue')
    plt.title(title)
    plt.show()


def get_training_data(data_transformation=None, n_samples=1000, stddev=[1.,1.]):
    x = np.zeros([n_samples,2], dtype=np.float32)
    for i in range(n_samples):
        x[i,0] = np.random.normal(0, stddev[0])
        x[i,1] = np.random.normal(0, stddev[1])

        if data_transformation == "square":
            x[i,1] = x[i,1] + x[i,0]**2 - 1


        if data_transformation == "sinusoid":
            x[i,1] = x[i,1] + np.sin(x[i,0])
    return x


if __name__ == "__main__":
    toy_2d()
