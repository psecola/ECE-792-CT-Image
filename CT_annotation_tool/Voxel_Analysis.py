#%%

from CT_Import_Helper_PC import *
import sklearn
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

#%%
# Specify the directory, get .tiff metadata, and create tiff stack numpy object
# directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\\Proj-Forams\Data_Files\716-4L-HEX-1'
directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\Proj-Forams\Data_Files\716-6L-HEX-2'

tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path = directory, condense = True, im_pooling = True,
                                                        kernel_size = 3, stride = 4, verbose = False)


#%%

voxel_vals = condensed_stack.ravel()
sampled_voxel_vals = np.random.choice(voxel_vals, size=20000, replace=False)

#%%

linked = linkage(sampled_voxel_vals.reshape(-1, 1), method='ward', metric='euclidean')
max_clusters = 6
labels_maxclust = fcluster(linked, t=max_clusters, criterion='maxclust')

# create a DataFrame
vox_dist = pd.DataFrame({'Dist Label': labels_maxclust, 'values': sampled_voxel_vals}).query('values != 0')

# group data & plot histogram
bin_num = 100
ax = vox_dist.pivot(columns='Dist Label', values='values').plot.hist(bins=bin_num)
min_val = vox_dist['values'].min()
max_val = vox_dist['values'].max()
tick_interval = np.floor((max_val - min_val)/bin_num)
custom_ticks = np.arange(np.floor(min_val), np.ceil(max_val) + tick_interval, tick_interval)
plt.xticks(ticks=custom_ticks, rotation=-45)
plt.show()

#%%
from sklearn.neighbors import NearestCentroid

# Create and fit the NearestCentroid classifier
clf = NearestCentroid()
clf.fit(sampled_voxel_vals.reshape(-1, 1), labels_maxclust)

predicted_class = clf.predict(condensed_stack.reshape(-1, 1))
class_voxels = np.hstack((predicted_class.reshape(-1, 1), condensed_stack.reshape(-1, 1)))

#%%

mask1 = class_voxels[:, 0] == 1
test_voxs1 = np.where(mask1.reshape(condensed_stack.shape), condensed_stack, 0)

mask2 = class_voxels[:, 0] == 2
test_voxs2 = np.where(mask2.reshape(condensed_stack.shape), condensed_stack, 0)

mask3 = class_voxels[:, 0] == 3
test_voxs3 = np.where(mask3.reshape(condensed_stack.shape), condensed_stack, 0)

mask4 = class_voxels[:, 0] == 4
test_voxs4 = np.where(mask4.reshape(condensed_stack.shape), condensed_stack, 0)

mask5 = class_voxels[:, 0] == 5
test_voxs5 = np.where(mask5.reshape(condensed_stack.shape), condensed_stack, 0)

mask6 = class_voxels[:, 0] == 6
test_voxs6 = np.where(mask6.reshape(condensed_stack.shape), condensed_stack, 0)

#Create Pyvista plotter object and plot to annotate object
grid = pv.ImageData()
grid.dimensions = np.array(test_voxs1.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = test_voxs1.ravel(order="F")  # Use Fortran ordering for flattening

grid2 = pv.ImageData()
grid2.dimensions = np.array(test_voxs2.shape) + 1
grid2.origin = (0, 0, 0)
grid2.spacing = (1, 1, 1)
grid2.cell_data["values"] = test_voxs2.ravel(order="F")  # Use Fortran ordering for flattening

grid3 = pv.ImageData()
grid3.dimensions = np.array(test_voxs3.shape) + 1
grid3.origin = (0, 0, 0)
grid3.spacing = (1, 1, 1)
grid3.cell_data["values"] = test_voxs3.ravel(order="F")  # Use Fortran ordering for flattening

grid4 = pv.ImageData()
grid4.dimensions = np.array(test_voxs4.shape) + 1
grid4.origin = (0, 0, 0)
grid4.spacing = (1, 1, 1)
grid4.cell_data["values"] = test_voxs4.ravel(order="F")  # Use Fortran ordering for flattening

grid5 = pv.ImageData()
grid5.dimensions = np.array(test_voxs5.shape) + 1
grid5.origin = (0, 0, 0)
grid5.spacing = (1, 1, 1)
grid5.cell_data["values"] = test_voxs5.ravel(order="F")  # Use Fortran ordering for flattening

grid6 = pv.ImageData()
grid6.dimensions = np.array(test_voxs6.shape) + 1
grid6.origin = (0, 0, 0)
grid6.spacing = (1, 1, 1)
grid6.cell_data["values"] = test_voxs6.ravel(order="F")  # Use Fortran ordering for flattening

p = pv.Plotter(shape=(3,2))

p.subplot(0,0)
p.add_volume(grid, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs1[test_voxs1 != 0])} max vox values: {np.max(test_voxs1)}', font_size=10, position="upper_left")

p.subplot(0,1)
p.add_volume(grid2, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs2[test_voxs2 != 0])} max vox values: {np.max(test_voxs2)}', font_size=10, position="upper_left")

p.subplot(1,0)
p.add_volume(grid3, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs3[test_voxs3 != 0])} max vox values: {np.max(test_voxs3)}', font_size=10, position="upper_left")

p.subplot(1,1)
p.add_volume(grid4, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs4[test_voxs4 != 0])} max vox values: {np.max(test_voxs4)}', font_size=10, position="upper_left")

p.subplot(2,0)
p.add_volume(grid5, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs5[test_voxs5 != 0])} max vox values: {np.max(test_voxs5)}', font_size=10, position="upper_left")


p.subplot(2,1)
p.add_volume(grid6, opacity='sigmoid', cmap='viridis', show_scalar_bar=False)
p.add_text(text=f'min vox values: {np.min(test_voxs6[test_voxs6 != 0])} max vox values: {np.max(test_voxs6)}', font_size=10, position="upper_left")


p.show()


