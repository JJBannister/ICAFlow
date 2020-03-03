import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..models.real_nvp import RealNVP

def toy_2d():
    x = get_training_data("Moons", 1000)
    print(x.shape)
    viz_2d(x)

    real_nvp = RealNVP()
    real_nvp.build_model(x.shape[1], 10, 5)

    x_hat = real_nvp.transformed_distribution.sample(1000)
    viz_2d(x_hat)

    real_nvp.train_model(x, lr=1e-3, n_epochs=1)

    x_hat = real_nvp.transformed_distribution.sample(1000)
    viz_2d(x_hat)

def viz_2d(data):
    sns.scatterplot(x=data[:,0], y=data[:,1])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

def get_training_data(data_type, n_samples):
    x = np.zeros([n_samples,2], dtype=np.float32)

    if data_type == "Square":
        for i in range(n_samples):
            x[i,0] = np.random.uniform(-1,1)
            x[i,1] = np.random.uniform(-1,1)

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