#%%
import pyvista as pv
from CT_Import_Helper_PC import *

#%%
# Specify the directory, get .tiff metadata, and create tiff stack numpy object
# directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\\Proj-Forams\Data_Files\716-4L-HEX-1'
directory = r'C:\Users\pseco\Documents\Micro_CT_Dissertation\Proj-Forams\Data_Files\716-6L-HEX-2'

tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path = directory, condense = True, im_pooling = True,
                                                        kernel_size = 3, stride = 4, plot_image = False,
                                                        image_type = 'volume slicer', verbose = False)

#%%
#Create Pyvista plotter object and plot to annotate object
grid = pv.ImageData()
grid.dimensions = np.array(condensed_stack.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = condensed_stack.ravel(order="F")  # Use Fortran ordering for flattening
plane_cords = {i : None for i in ["x", "y", "z"]}  # Used to track past plane coordinates
slices_dict = {}  # Used to track 2D slices
masked_slices = {}

p = pv.Plotter()
p.add_volume_clip_plane(grid, normal='x', opacity='sigmoid', cmap='viridis', invert = False,
                                  assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12)
p.show()