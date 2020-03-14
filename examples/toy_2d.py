import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Nadam
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../')
from ica_flow.affine_coupling_ica import AffineCouplingICA as ICA

def toy_2d():

    # Synthetic Data
    #x = get_training_data(data_transformation='None', n_samples=1000, stddev=[3, 0.1])
    x = get_training_data(data_transformation='sinusoid', n_samples=5000, stddev=[3, 0.2])
    #x = get_training_data(data_transformation='square', n_samples=3000, stddev=[1, 0.1])

    # Flow
    ica = ICA(
            input_shape=x.shape[-1], 
            n_coupling_layers=8,
            hidden_layer_dim=8)

    viz_2d(ica.transformed_distribution.sample(5000), data2=x,
           title="Train + Target samples (Pre-train)")

    # Train
    i = Input(shape=x.shape[-1])

    log_prob = ica.transformed_distribution.log_prob(i)
    model = Model(i, log_prob) 
    optimizer = Nadam(lr=1e-3)
    model.compile(optimizer=optimizer,
            loss=lambda _, log_prob: -log_prob)

    model.fit( 
            x=x, 
            y=np.zeros((x.shape[0], 0), dtype=np.float32), 
            epochs=300,
            batch_size=512,
            shuffle=True)

    print("Latent Variable Stddevs: ", ica.latent_stddev())

    # Target Samples
    viz_2d(ica.transformed_distribution.sample(5000), data2=x,
            title="Train + Target samples (Post-train)")

    # Grid
    samples = get_grid_samples(-5, 5)
    viz_2d(samples, title="Grid")
    y = ica.bijector.forward(samples)
    viz_2d(y, data2=x, title="Transformed_Grid")

    # Iso-Probability Lines 
    samples = get_ring_samples()
    viz_2d(samples, title="Rings")
    y = ica.bijector.forward(samples)
    viz_2d(y, data2=x, title="Transformed_Rings")


def viz_2d(data1, data2=None, title=""):
    plt.scatter(x=data1[:,0], y=data1[:,1],c='red', s=1)
    if not data2 is None:
        plt.scatter(x=data2[:,0], y=data2[:,1],c='blue', s=1)
    plt.title(title)
    plt.show()


def get_grid_samples(minimum, maximum, n_lines=10, n_samples=5000):
    line = np.linspace(minimum, maximum, n_samples, dtype=np.float32)
    grid = np.linspace(minimum, maximum, n_lines, dtype=np.float32)

    x0 = np.transpose([np.tile(line, len(grid)), np.repeat(grid, len(line))])
    x1 = np.transpose([np.tile(grid, len(line)), np.repeat(line, len(grid))])

    return np.concatenate([x0,x1],axis=0)


def get_ring_samples(n_samples=5000, n_rings=6):
    thetas = np.linspace(0, 2*np.pi, n_samples)
    rhos = np.logspace(-3, 1, num=n_rings, base=np.e)

    x = [np.zeros([n_samples,2], dtype=np.float32) for _ in range(n_rings)]
    for i in range(n_rings):
        for j in range(n_samples):
            x[i][j,0] = rhos[i]*np.cos(thetas[j])
            x[i][j,1] = rhos[i]*np.sin(thetas[j])

    return np.concatenate(x,axis=0)


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
