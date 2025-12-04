#%%
import numpy as np


def generate_ellipsoidal_data(ell_par, type, num):
    # ellipsoidal points
    center = ell_par[0:3]
    a = ell_par[3]
    b = ell_par[4]
    c = ell_par[5]
    alpha = ell_par[6]
    beta = ell_par[7]
    gamma = ell_par[8]

    # scale matrix
    A = np.diag([1 / a ** 2, 1 / b ** 2, 1 / c ** 2])

    # rotation matrices
    invRx = np.array([[1, 0, 0],
                      [0, np.cos(-alpha), np.sin(-alpha)],
                      [0, -np.sin(-alpha), np.cos(-alpha)]])

    invRy = np.array([[np.cos(-beta), 0, np.sin(-beta)],
                      [0, 1, 0],
                      [-np.sin(-beta), 0, np.cos(-beta)]])

    invRz = np.array([[np.cos(-gamma), np.sin(-gamma), 0],
                      [-np.sin(-gamma), np.cos(-gamma), 0],
                      [0, 0, 1]])

    R = invRz @ invRy @ invRx
    M = R.T @ A @ R
    U, S, V = np.linalg.svd(M)
    L = np.real(V.T @ np.diag(1.0 / np.sqrt(S)) @ V)

    Dimension = 3
    NumSamples = num

    # Obtain random samples evenly distributed on the surface of the unit hypersphere
    Samples = np.random.randn(Dimension, NumSamples)
    SampleNorms = np.sqrt(np.sum(Samples ** 2, axis=0))
    Samples = Samples / SampleNorms

    # Add some noise
    Samples += 0.05 * np.random.randn(*Samples.shape)  # noise: 0.01-0.05-0.1-0.15-0.2-0.25

    # Transform the data into the desired ellipsoid
    Samples = L @ Samples + np.array(center)[:, np.newaxis]

    Outliers = np.empty((0, 3))
    NumOut = int(0.3 * num)
    if type:
        Ox = np.random.rand(NumOut) * 200 - 100
        Oy = np.random.rand(NumOut) * 200 - 100
        Oz = np.random.rand(NumOut) * 150 - 50
        Outliers = np.vstack((Ox, Oy, Oz)).T

    Samples1 = {'sample': Samples.T, 'outlier': Outliers}

    return Samples1, L


