import numpy as np
from parameter_solving import parameter_solving


def ellipsoid_fit_EM(X, Y, outlierness, normal):
    opt = {
        #'outliers': 0.5,  # You can use "outlierness" to set this parameter, but 0.5 works well in most cases.
        'outliers': outlierness,
        'max_it': 200,  # max number of iterations
        'tol': 1e-8,  # tolerance
        'normal': normal
    }

    # Convert to double type, save Y
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    # Volume of the bounding box
    MaxX = np.max(X, axis=0)
    MinX = np.min(X, axis=0)
    L = MaxX - MinX
    V = L[0] * L[1] * L[2]

    opt['v'] = V
    normal = opt['normal']

    # Start fitting
    R, t, iter, spend = parameter_solving(X, Y, opt['tol'], opt['outliers'], opt['v'])



    Transform = {
        'R': R,
        't': t,
        's': normal['xscale'] / normal['yscale']
    }
    Transform['t'] = normal['xscale'] * t + normal['xd'] - Transform['s'] * (Transform['R'] @ normal['yd'].T)
    Transform['R'] = Transform['s'] * Transform['R']

    return Transform, iter, spend