#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.proj3d import proj_transform_clip
from ellipsoid_fit_EM import ellipsoid_fit_EM
from ellipsoid_par import ellipsoid_par
from outlier_det import outlier_det
from data_normalize_input import data_normalize_input
from drawEllipsoid import draw_ellipsoid
from generate_ellipsoidal_data import generate_ellipsoidal_data
from ellipsoid_plotter import generate_ellipsoid_3d
import pyvista as pv
import time
import scipy.io as sio

def execution_timer(start_time, message):
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    #print(f"{message} {exec_time:.4f} seconds")


def chamber_fitting(annotation_dict : dict, threshold, scaling_params = (1,1,1), pixel_measurement = 1):

    chamber_params = {}

    for chamber in annotation_dict:

        # capture the 3D masked tensor of a chamber
        foram_array = annotation_dict[chamber]['3D Tensor']

        #Create a temporary dictionary to store computed params and scaled params
        params_temp_dict = {}

        # Start time of algorithm
        start_time = time.perf_counter()

        # Set the number of samples and find all non-zero values
        threshold = threshold
        indices_thres_array = np.argwhere(foram_array > threshold)
        num_samples = min(len(indices_thres_array), 200)


        # Randomly sample the desired number of coordinates
        sampled_indices = np.random.choice(len(indices_thres_array), size=num_samples, replace=False)
        sampled_coords = indices_thres_array[sampled_indices]

        # Center data to perform PCA
        ptFit = sampled_coords

        #Visualize sampled point cloud
        #viz_sampled_pc(ptFit)


        # Compute outlierness
        knn_map_id, rdos_score, X, X_normal = outlier_det(ptFit)
        inlier = ptFit[np.array(rdos_score) <= 2, :]
        inlier_num = inlier.shape[0]
        init_center = np.mean(inlier, axis=0)  # mass center
        outlierness = 1 - inlier_num / ptFit.shape[0]

        # Initialization the ellipsoid parameter
        #ellParInit = [*init_center, 1, 1, 1, roll_init, pitch_init, yaw_init]  # init_center
        ellParInit = [*init_center, 1, 1, 1, 0, 0, 0]  # init_center
        ptInit = draw_ellipsoid(ellParInit, 1, np.sqrt(inlier_num))

        # Normalization
        Y, Y_normal = data_normalize_input(ptInit)


        # Start fitting
        normal = {
            'xd': X_normal['xd'],
            'yd': Y_normal['xd'],
            'xscale': X_normal['xscale'],
            'yscale': Y_normal['xscale']
        }


        transform, iter, spend = ellipsoid_fit_EM(X, Y, outlierness, normal)

        # Get final ellipsoid parameters (center cord, axis lengths (x,y,z lengths), euler angles (roll, pitch, yaw))
        FitEllipsoid = ellipsoid_par(init_center, transform['R'], transform['t'])


        params_temp_dict['computed params'] = FitEllipsoid.tolist()
        params_temp_dict['scaled params'] = ([x * y * pixel_measurement for x, y in zip(FitEllipsoid[0:3], scaling_params)] +
                                             [x * y * pixel_measurement for x, y in zip(FitEllipsoid[3:6], scaling_params)] +
                                             FitEllipsoid.tolist()[6:9])

        chamber_params[chamber] = params_temp_dict

        #execution_timer(start_time, f'{chamber} execution time: ')


    return chamber_params
