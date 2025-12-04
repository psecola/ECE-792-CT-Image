import numpy as np

def cpd_comp(x, y, sigma2, outlier, V):
    """
    Python / NumPy translation of the provided C/MEX cpd_comp function.

    Parameters
    ----------
    x : ndarray, shape (N, D)
    y : ndarray, shape (M, D)
    sigma2 : float
    outlier : float
    V : float

    Returns
    -------
    P1 : ndarray, shape (M,)
    Pt1: ndarray, shape (N,)
    Px : ndarray, shape (M, D)
    PM : ndarray, shape (N,)
    E  : float
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma2 = float(sigma2)
    outlier = float(outlier)
    V = float(V)

    N, D = x.shape
    M = y.shape[0]

    # outputs (initialized to zero)
    P1 = np.zeros(M, dtype=float)     # corresponds to OUT_P1 (M x 1)
    Pt1 = np.zeros(N, dtype=float)    # OUT_Pt1 (N x 1)
    Px = np.zeros((M, D), dtype=float)# OUT_Px (M x D)
    PM = np.zeros(N, dtype=float)     # OUT_PM (N x 1)
    E = 0.0

    ksig = -2.0 * sigma2

    # Note: matches your C line:
    # outlier_tmp = (*outlier*M*pow (-ksig*pi,0.5*D))/((1-*outlier)**V);
    # outlier_tmp2 = (*V*(1-*outlier))/(*outlier*M);
    outlier_tmp = (outlier * M * ((-ksig * 3.14159265358979) ** (0.5 * D))) / ((1.0 - outlier) * V)

    try:
        outlier_tmp2 = (V * (1.0 - outlier)) / (outlier * M) # possibly undefined

    except ZeroDivisionError:
        if (V * (1.0 - outlier)) > 0:
            outlier_tmp2 = float('inf')
        elif (V * (1.0 - outlier)) < 0:
            outlier_tmp2 = float('-inf')

    # Loop over n (kept similar structure; inner operations vectorized)
    for n in range(N):
        # squared distances between x[n] and all y: shape (M,)
        diff = y - x[n]           # shape (M, D)
        razn = np.sum(diff * diff, axis=1)  # shape (M,)

        P = np.exp(razn / ksig)    # same as exp(-razn/(2*sigma2))
        sp = np.sum(P)                 # sum over m

        PM[n] = 1.0 / (1.0 + outlier_tmp2 * sp)

        sp_with_outlier = sp + outlier_tmp
        Pt1[n] = 1.0 - outlier_tmp / sp_with_outlier

        # temp_x[d] = x[n,d] / sp_with_outlier  (C used sp which already included outlier_tmp only after adding it)
        temp_x = x[n] / sp_with_outlier  # shape (D,)

        # accumulate P1 and Px
        P_over_sp = P / sp_with_outlier  # shape (M,)
        P1 += P_over_sp                   # elementwise add to length M

        # Px[m,d] += temp_x[d] * P[m]  -> vectorized outer product
        # shape (M, D) += (M,1) * (1,D)
        Px += np.outer(P, temp_x)

        # E contribution
        E += -np.log(sp_with_outlier)

    # final E adjustment (matches C: *E += D*N*log(*sigma2)/2 - N*log(1-*outlier);)
    E += (D * N * np.log(sigma2)) / 2.0 - N * np.log(1.0 - outlier)

    return P1, Pt1, Px, PM, E

