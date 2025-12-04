#%%
from data_normalize_input import data_normalize_input
from knn_cpp_p import *

def outlier_det(data):
    """
    Outlier detection using rdos_outlier
    Input
    -------------
    data   input data Nx3 array

    Output
    ------------
    rdos_score               outlierness
    data                     normalized data
    data_normal              record the mean and std of the normal
    """

    global knn_map_id

    # Normalization of the input data---> zero mean, unit std
    data, data_normal = data_normalize_input(data)

    num_data = data.shape[0]

    # the highest k values to examine
    K = min(100, num_data - 1)

    # Build the KNN graph to be used later
    knn_map_id = np.zeros((num_data, K), dtype=int)  # KNN label graph
    knn_map_dis = np.zeros((num_data, K))  # KNN distance graph

    for i in range(num_data):
        dataTmp = data[i, :]
        dis = np.round(np.sqrt(np.sum((np.ones((num_data, 1)) * dataTmp - data) ** 2, axis=1)), 4)
        c = np.where(dis == 0)[0]
        dis[c] = 999999
        sortix = np.argsort(dis)
        knn_map_id[i, :] = sortix[:K]
        knn_map_dis[i, :] = dis[sortix[:K]]


    rdos_score = knn_function(data, knn_map_id, K)

    return knn_map_id, rdos_score, data, data_normal

