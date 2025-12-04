#%%
import pyvista as pv
import pickle
from pv_helper_PC import *

#%%
annotation_dict = open_3d_annotations(r'C:\Users\pseco\Documents\Micro_CT_Dissertation\Proj-Forams\Data_Files\Annotated_Foram_Dicts\716-4L-HEX-1_dict.pkl')
foram_array = annotation_dict['Chamber2']['3D Tensor']

#%%

#Create Pyvista plotter object and plot to annotate object
grid = pv.ImageData()
grid.dimensions = np.array(foram_array.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = foram_array.ravel(order="F")  # Use Fortran ordering for flattening

p = pv.Plotter()
p.add_volume_clip_plane(grid, normal='x', opacity='sigmoid', cmap='viridis', invert = False,
                                  assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12)
p.show()