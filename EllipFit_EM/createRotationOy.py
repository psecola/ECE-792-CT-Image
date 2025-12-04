import numpy as np


def create_rotation_oy(*args):
    # default values
    dx = 0
    dy = 0
    dz = 0
    theta = 0

    # get input values
    if len(args) == 1:
        # only one argument -> rotation angle
        theta = args[0]

    elif len(args) == 2:
        # origin point (as array) and angle
        var = args[0]
        dx = var[0]
        dy = var[1]
        dz = var[2]
        theta = args[1]

    elif len(args) == 3:
        # origin (x and y) and angle
        dx = args[0]
        dy = args[1]
        dz = 0
        theta = args[2]

    elif len(args) == 4:
        # origin (x and y) and angle
        dx = args[0]
        dy = args[1]
        dz = args[2]
        theta = args[3]

    # compute coefs
    cot = np.cos(theta)
    sit = np.sin(theta)

    # create transformation
    trans = np.array([
        [cot, 0, sit, 0],
        [0, 1, 0, 0],
        [-sit, 0, cot, 0],
        [0, 0, 0, 1]
    ])

    # add the translation part
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    trans = np.dot(t, trans) @ np.linalg.inv(t)

    return trans