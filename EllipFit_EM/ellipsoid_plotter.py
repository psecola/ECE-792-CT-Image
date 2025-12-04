#%%
import pyvista as pv
import matplotlib
import numpy as np

def create_ellipsoid_from_9_parameters(center, radii, rotation_angles):
    """
    Creates a PyVista mesh for an ellipsoid using 9 parameters.

    Parameters
    ----------
    center : tuple or list
        The (x, y, z) coordinates of the ellipsoid's center.
    radii : tuple or list
        The (x, y, z) radii of the ellipsoid.
    rotation_angles : tuple or list
        The (alpha, beta, gamma) rotation angles in degrees for X, Y, and Z axes.

    Returns
    -------
    pyvista.PolyData
        The ellipsoid mesh.
    """
    # Create a canonical ellipsoid with the given radii, centered at the origin
    # The pyvista.ParametricEllipsoid function creates an axis-aligned shape.
    ellipsoid = pv.ParametricEllipsoid(
        xradius=radii[0], yradius=radii[1], zradius=radii[2]
    )

    # Create a transform object to apply rotation and translation
    transform = pv.Transform()

    # Apply rotations in the order of Z, Y, X for intuitive Euler angles
    transform.rotate_z(rotation_angles[2]) # Yaw
    transform.rotate_y(rotation_angles[1]) # Pitch
    transform.rotate_x(rotation_angles[0]) # Roll

    # Apply the final translation to move the ellipsoid to its center
    transform.translate(center)

    # Apply the combined transform to the ellipsoid mesh
    transformed_ellipsoid = ellipsoid.transform(transform)

    return transformed_ellipsoid

# Main function to generate ellipsoid
def generate_ellipsoid_3d(params):

    # Define the 9 parameters
    ellipsoid_center = params[0:3]
    ellipsoid_radii = params[3:6]
    ellipsoid_rotation_angles = params[6:9] # Degrees for roll, pitch, yaw

    # Create the ellipsoid mesh
    my_ellipsoid = create_ellipsoid_from_9_parameters(
        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation_angles
    )

    return my_ellipsoid
