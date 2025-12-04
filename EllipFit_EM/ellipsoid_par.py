import numpy as np


def ellipsoid_par(init_center: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Get the final ellipsoidal parameters using the solved affine transformation.

    Input:
    --------------
    init_center : np.ndarray
        The initial spherical center (e.g., a 1D array of shape (3,)).
    R : np.ndarray
        The solved affine matrix (e.g., a 3x3 matrix).
    t : np.ndarray
        The translation vector (e.g., a 1D array of shape (3,)).

    Output:
    -------------
    P : np.ndarray
        A 1D array containing the nine geometric parameters of the fitted
        ellipsoidal surface: (x_c, y_c, z_c, a, b, c, alpha, beta, gamma).
        (x_c, y_c, z_c) are the ellipsoid's center coordinates.
        (a, b, c) are the three semi-axis lengths.
        (alpha, beta, gamma) are the rotation angles (Euler angles).
    """

    # Ensure init_center and t are treated as column vectors for matrix multiplication if they are 1D.
    # NumPy handles this correctly for @ operator with 1D arrays, resulting in a 1D array.
    # If inputs are (3,1) column vectors, they will remain so and the result will be (3,1).

    # Center of the ellipsoid
    # MATLAB: center = t + R * init_center';
    # In NumPy, if init_center is (3,) and R is (3,3), R @ init_center results in (3,).
    # If init_center is (3,1) and R is (3,3), R @ init_center results in (3,1).
    # Assuming standard NumPy vector/matrix operations where 1D arrays are treated as vectors.
    center = t + R @ init_center

    # Three semi-axis lengths
    # MATLAB: B = R * R';
    B = R @ R.T


    # MATLAB: [V,D] = eig(B);
    # np.linalg.eigh returns eigenvalues in ascending order.
    # V are eigenvectors as columns.
    eigenvalues, V = np.linalg.eigh(B)


    # MATLAB: a=sqrt(D(1,1)); b=sqrt(D(2,2)); c=sqrt(D(3,3));
    # In MATLAB, D(i,i) corresponds to the i-th eigenvalue.
    # In NumPy, eigenvalues[i] is the i-th eigenvalue (0-indexed).
    # Given that np.linalg.eigh sorts eigenvalues in ascending order,
    # and the MATLAB code doesn't explicitly sort them further,
    # we take them as they come.
    # If a,b,c are conventionally major, medium, minor axes (a >= b >= c),
    # then we'd need to sort eigenvalues in descending order and map them.
    # However, the original MATLAB code takes them directly, so we follow that.
    # Therefore, eigenvalues[0] corresponds to D(1,1), etc.
    a = np.sqrt(eigenvalues[0])
    b = np.sqrt(eigenvalues[1])
    c = np.sqrt(eigenvalues[2])


    # Rotation angles (Euler angles)
    denominator_rx1 = np.sqrt(V[0, 0] ** 2 + V[1, 0] ** 2)
    rx1 = np.arctan2(-V[2, 0], denominator_rx1)

    # ry1 (alpha): Rotation around Z-axis (yaw)
    ry1 = np.arctan2(V[1, 0], V[0, 0])

    # rx1 (beta): Rotation around Y'-axis (pitch)
    # The denominator sqrt(V(1,1)^2+V(2,1)^2) handles potential gimbal lock issues
    # (where the pitch angle is +/- 90 degrees).


    # rz1 (gamma): Rotation around X''-axis (roll)
    rz1 = np.arctan2(V[2, 1], V[2, 2])

    # Construct the output array P
    # MATLAB: P = [center' a b c ry1 rx1 rz1];
    # If center is a 1D array (3,), center.flatten() is center itself.
    # If center is a 2D array (3,1), center.flatten() makes it (3,).
    # np.concatenate joins the arrays.
    P = np.concatenate((center.flatten(), np.array([a, b, c, ry1, rx1, rz1])))

    return P
