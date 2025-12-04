import numpy as np

def kernel_comp(data, total_inx, inx, point_num, h=1e-1):
    """
    Compute kernel density estimate for a single point.

    data: (M, D) array
    total_inx: list of neighbor indices (1-based)
    inx: index of the current point (0-based)
    point_num: total number of points (M)
    h: bandwidth
    """
    N = len(total_inx)
    temp = 1 / ((N + 1) * ((2 * np.pi * h * h) ** 1.5))

    # convert neighbor indices to 0-based for NumPy access
    neighbors = np.array(total_inx) - 1

    diffs = data[inx] - data[neighbors]  # shape (N, D)
    razn = np.sum(diffs ** 2, axis=1)
    sum_val = np.sum(np.exp(-0.5 * razn / (h * h)))

    # add self
    sum_val += 1.0

    return temp * sum_val


def score_comp(data, knn_map_id, K):
    """
    data: (M, D) numpy array
    knn_map_id: (M, C) numpy array of indices (1-based, same as MATLAB)
    K: number of neighbors
    returns: score array (M,)
    """
    M, D = data.shape
    knn_map_id = knn_map_id.astype(int)

    point_pdf = []
    all_total_inx = []

    for i in range(M):
        total_inx = []

        j = i

        # find all m such that j is in knn_map_id[m, :]
        mask = np.any(knn_map_id == j, axis=1)
        total_inx.extend(np.where(mask)[0].tolist())

        # add i’s own neighbors
        for k in range(K):
            l = knn_map_id[i,k]
            total_inx.append(l)

            # add neighbors of l -- Breaks here
            for k2 in range(K):
                l1 = knn_map_id[l, k2]
                total_inx.append(l1)


        # unique & sorted (still 0-based)
        total_inx = sorted(set(total_inx))
        all_total_inx.append(total_inx)
        # kernel density for point i (passing 0-based neighbor list)
        pdf = kernel_comp(data, total_inx, i, M)
        point_pdf.append(pdf)


    # compute scores
    score = np.zeros(M)
    for i in range(M):
        pdf_sum = 0
        num = len(all_total_inx[i])
        for j in range(num):
            temp = all_total_inx[i][j]-1  # convert to 0-based
            if temp == -1:
                pass
            else:
                pdf_sum += point_pdf[temp]


        ave = pdf_sum / (num - 1)
        score[i] = ave / point_pdf[i]

    return score


def knn_function(data, knn_map_id, K):

    # Output

    # Do the actual computations in the subroutine
    OUT_x = score_comp(data, knn_map_id, K)

    return OUT_x

