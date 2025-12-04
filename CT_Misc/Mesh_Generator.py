#%%
# Import

import sys
import os
from importlib import reload
import numpy as np
import pyvista as pv
import time
from collections import defaultdict
from stl import mesh # from numpy-stl library
from CT_Import_Helper import *

sys.path.append(r'C:\Users\pseco\Documents\Mirco CT Dissertation\Proj-Forams\Foram_Mirco_CT_Code\Packages')
import SurfaceNet #This is a .py function downloaded from https://github.com/mjoppich/surfacenet_python/blob/master/SurfaceNet.py

def mesh_generator(tensor, threshold = 120):

    # Generate surface mesh vertices and face coordinates. Face coordinates are tuples of vertex combinations to create a face
    sn = SurfaceNet.SurfaceNets()

    # 125 is the upper threshold of pixels used above; it is the pixels that are assumed to be solid
    start_time = time.perf_counter()
    verts,faces = sn.surface_net(tensor, threshold)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.4f} seconds") #~8 mins
    print("done")

    return verts, faces


def mesh_to_stl(vertices, faces):
    # Save mesh as STL file
    mesh_obj = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):

        for j in range(3):
            mesh_obj.vectors[i][j] = vertices[f[j]]

    posElem = defaultdict(list)

    for i in range(3):
        for vert in mesh_obj.vectors:
            posElem[i].append(vertices[i])

    # this prints the min/max value per dimension
    for posidx in posElem:
        print(posidx, np.min(posElem[posidx]), np.max(posElem[posidx]))

    print("Mesh Complete")

    return mesh_obj

#%%

directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\Proj-Forams\Data_Files'
mesh_directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\Proj-Forams\Data_Files\Foram_Meshes'
for i, subdirectory in enumerate(os.listdir(directory)):
    if subdirectory != 'Foram_Meshes' and i > 4:
        print(subdirectory)

        # Specify the directory, get .tiff metadata, and create tiff stack numpy object
        files = os.path.join(directory, subdirectory)

        # generates a NumPy tensor for the Foram CT information and a downsampled (condensed) version of the NumPy tensor
        tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path = files, condense = True, im_pooling = True,
                                                               kernel_size = 3, stride = 1, verbose = False)

        # Convert Mesh object to a stl file format and export

        vertices, faces = mesh_generator(condensed_stack, threshold=120)
        foram = mesh_to_stl(vertices, faces)

        foram.save(os.path.join(mesh_directory, ".".join((subdirectory, 'stl'))))

        del foram




