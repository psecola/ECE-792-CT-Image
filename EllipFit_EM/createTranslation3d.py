import numpy as np

def create_translation_3d(*args):
    if len(args) == 0:
        # assert translation with null vector
        dx = 0
        dy = 0
        dz = 0
    elif len(args) == 1:
        # translation vector given in a single argument
        var = args[0]
        dx = var[0]
        dy = var[1]
        dz = var[2]
    else:
        # translation vector given in 3 arguments
        dx = args[0]
        dy = args[1]
        dz = args[2]

    # create the translation matrix
    trans = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
    return trans