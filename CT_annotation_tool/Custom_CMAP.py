#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%
def create_cmap(threshold : int):

    # Create a custom colormap controlling voxel color and transparency
    # White pixel value (0,0,0) with full transparency (0) for pixel values up to 89
    colors1 = np.tile(np.array([0,0,0,0]), (threshold,1))

    # pull color value from autumn colormap when voxels are equal to threshold and use .25 transparency
    colors2 = plt.cm.autumn(np.arange(threshold,threshold+2))
    colors2[:, 3] = .25

    # Use viridis colors for rest of voxel values
    colors3 = plt.cm.viridis(np.arange(threshold+2,256))

    # Create custom colormap by stacking above, individual colormaps
    colors = np.vstack((colors1, colors2, colors3))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)


    return custom_cmap
