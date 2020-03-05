import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from ..models.nice import NICE

def toy_2d():
    x = get_training_data("Small Normal", 5000)
    viz_2d(x, "Training Samples")

    nice = NICE()
    nice.build_model(x.shape[-1], 0, 1)

    x = nice.base_distribution.sample(1000)
    viz_2d(x, "Base samples")

    x = nice.transformed_distribution.sample(1000)
    viz_2d(x, "Target samples (Pre-Training)")

    nice.train_model(x, lr=1e-3, n_epochs=100, batch_size=256)

    x_hat = nice.transformed_distribution.sample(1000)
    viz_2d(x_hat, "Target samples (Post-Training)")

    tf.print(nice.scale_values)


def viz_2d(data, title=""):
    sns.scatterplot(x=data[:,0], y=data[:,1])
    plt.title(title)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()


def get_training_data(data_type, n_samples):
    x = np.zeros([n_samples,2], dtype=np.float32)

    if data_type == "Square":
        for i in range(n_samples):
            x[i,0] = np.random.uniform(-1,1)
            x[i,1] = np.random.uniform(-1,1)

    if data_type == "Small Normal":
        for i in range(n_samples):
            x[i,0] = np.random.normal(0, 0.2)
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
