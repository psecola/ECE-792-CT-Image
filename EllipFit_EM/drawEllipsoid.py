import numpy as np
import matplotlib as plt
from createRotationOx import create_rotation_ox
from createRotationOy import create_rotation_oy
from createRotationOz import create_rotation_oz
from createScaling3d import create_scaling_3d
from createTranslation3d import create_translation_3d


def draw_ellipsoid(elli, type, num):

    # number of meridians
    nPhi = num

    # number of parallels
    nTheta = num

    # Parse numerical inputs
    xc = elli[0]
    yc = elli[1]
    zc = elli[2]
    a = elli[3]
    b = elli[4]
    c = elli[5]
    ellPhi = elli[6]
    ellTheta = elli[7]
    ellPsi = elli[8]

    # Coordinates computation

    # convert unit basis to ellipsoid basis
    sca = create_scaling_3d(a, b, c)
    rotZ = create_rotation_oz(ellPhi)
    rotY = create_rotation_oy(ellTheta)
    rotX = create_rotation_ox(ellPsi)
    tra = create_translation_3d(np.array([xc, yc, zc]).T)

    # concatenate transforms
    trans = tra @ rotZ @ rotY @ rotX @ sca
    # parameterization of ellipsoid

    # spherical coordinates
    theta = np.linspace(0, np.pi, int(nTheta) + 1)
    phi = np.linspace(0, 2 * np.pi, int(nPhi) + 1)

    # convert to cartesian coordinates
    sintheta = np.sin(theta)
    x = np.cos(phi[:, np.newaxis]) * sintheta
    y = np.sin(phi[:, np.newaxis]) * sintheta
    z = np.ones((len(phi), 1)) * np.cos(theta)
    NP = x.size

    for i in range(NP):
        res = np.array([x.flat[i], y.flat[i], z.flat[i], 1]) @ trans.T
        x.flat[i] = res[0]
        y.flat[i] = res[1]
        z.flat[i] = res[2]

    if type == 0:
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        outIntensity = 100
        x_out = np.random.randn(56) * outIntensity
        y_out = np.random.randn(56) * outIntensity
        z_out = np.random.randn(56) * outIntensity

        plt.figure()
        plt.hold(True)
        plt.plot3D(x, y, z, 'r.')
        plt.plot3D(x_out, y_out, z_out, 'k+')
        plt.axis('equal')
        plt.grid(True)
        plt.view_init(azim=3)
        plt.hold(False)

        plt.figure()
        plt.hold(True)
        plt.plot3D(x, y, z, 'r.')
        plt.plot3D(x_out, y_out, z_out, 'k+')
        plt.axis('equal')
        plt.grid(True)
        plt.view_init(azim=3)
        plt.hold(False)

        x = np.concatenate((x, x_out))
        y = np.concatenate((y, y_out))
        z = np.concatenate((z, z_out))
    elif type == 1:

        x = x.T.reshape(-1,1)
        y = y.T.reshape(-1,1)
        z = z.T.reshape(-1,1)
    elif type == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, cmap='viridis')  # you can pick any colormap
        ax.view_init(elev=30, azim=-60)  # similar to MATLAB "view(3)"


    Pt = np.column_stack((x, y, z))
    return Pt

