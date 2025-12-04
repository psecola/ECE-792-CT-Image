import numpy as np


def create_scaling_3d(*args):
    # default arguments
    sx = 1
    sy = 1
    sz = 1
    center = np.array([0, 0, 0])

    # process input parameters
    if len(args) == 1:
        # only one argument -> scaling factor
        sx, sy, sz = parse_scaling_factors(args[0])

    elif len(args) == 2:
        # 2 arguments, giving center and uniform scaling
        center = np.array(args[0])
        sx, sy, sz = parse_scaling_factors(args[1])

    elif len(args) == 3:
        # 3 arguments, giving scaling in each direction
        sx = args[0]
        sy = args[1]
        sz = args[2]

    elif len(args) == 4:
        # 4 arguments, giving center and scaling in each direction
        center = np.array(args[0])
        sx = args[1]
        sy = args[2]
        sz = args[3]

    # create the scaling matrix
    trans = np.array([
        [sx, 0, 0, center[0] * (1 - sx)],
        [0, sy, 0, center[1] * (1 - sy)],
        [0, 0, sz, center[2] * (1 - sz)],
        [0, 0, 0, 1]
    ])

    return trans


def parse_scaling_factors(var):
    if len(var) == 1:
        # same scaling factor in each direction
        sx = var
        sy = var
        sz = var
    elif len(var) == 3:
        # scaling is a vector, giving different scaling in each direction
        sx = var[0]
        sy = var[1]
        sz = var[2]
    else:
        raise ValueError('wrong size for first parameter of "create_scaling_3d"')

    return sx, sy, sz