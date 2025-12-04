import numpy as np
from precompute import cpd_comp


def parameter_solving(X, Y, tol, outliers, vol):

    # Inputs:
    # X: Nx3 array of fitted data points
    # Y: Mx3 array of generated spherical data
    # tol: tolerance (default 1e-8)
    # outliers: outlier weight
    # vol: volume of the bounding box X

    # Outputs:
    # B: 3x3 solved affine matrix
    # t: 3x1 translation vector

    N, _ = X.shape
    M, D = Y.shape

    # Initialization of the variance
    sigma2 = (M*np.trace(X.T@X)+N*np.trace(Y.T@Y)-2*np.dot(np.sum(X, axis=0), np.sum(Y, axis=0))) / (M * N * D)

    T = Y.copy()

    # Optimization
    iter_num = 0
    ntol = tol + 10
    L = 1

    # Start iteration by the EM
    s1 = np.datetime64('now')
    while ntol > tol:
        L_old = L

        # X and vol are fixed
        P1, Pt1, PX, PM, L = cpd_comp(X, T, sigma2, outliers, vol)
        #print("P1", P1)
        # print("Pt1", Pt1)
        # print("Px", PX)
        # print("PM", PM)
        # print("L", L)

        ntol = abs((L - L_old) / L)

        # Intermediate parameter
        Np = np.sum(P1)

        mu_x = X.T @ Pt1 / Np
        mu_y = Y.T @ P1 / Np

        # Solve for parameters
        P1_rep = np.repeat(np.array(P1).reshape(-1, 1), D, axis=1)
        B1 = (PX.T @ Y) - (Np * np.outer(mu_x, mu_y.T))
        B2 = ((Y*P1_rep).T @ Y) - (Np * np.outer(mu_y, mu_y.T))
        B = np.linalg.solve(B2.T, B1.T).T # Numerically stable way to compute B1/B2

        t = mu_x - B @ mu_y

        sigma2 = abs(np.sum(X**2 * Pt1[:, np.newaxis]) - Np * (mu_x.T @ mu_x) - np.trace(B1 @ B.T)) / (Np * D)


        # Update weight
        temp = np.sum(PM)
        outliers = temp / (temp + Np)

        iter_num += 1

        # Update centroids positions
        T = (Y @ B.T) + np.tile(t.T, (M, 1))

    e1 = np.datetime64('now')
    iter_time = (e1 - s1).astype('timedelta64[s]').astype(int)

    return B, t, iter_num, iter_time
