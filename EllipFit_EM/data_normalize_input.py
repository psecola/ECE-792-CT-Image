import numpy as np


def data_normalize_input(x):
    n = x.shape[0]

    pre_normal = {}
    pre_normal['xd'] = np.mean(x, axis=0)

    x = x - np.tile(pre_normal['xd'], (n, 1))

    pre_normal['xscale'] = np.sqrt(np.sum(np.sum(x ** 2, axis=1)) / n)

    X = x / pre_normal['xscale']

    return X, pre_normal