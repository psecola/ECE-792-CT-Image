#%%
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
import numpy as np
from matplotlib.path import Path
import os
from PIL import Image, ImageDraw
import copy


matplotlib.use('TkAgg') # or 'Qt5Agg'


'''
function: 
    Extracts user defined vertices from the one of the CT cross-sections within the image segmentor window. 
    Once a polygon has been defined on one of the CT cross sections a max is generated using a meshgrid and saved as 
    2D numpy mask

input: 
    verts (list) - a list of vertex coordinates that is continually appended to as the user clicks on the CT cross-sections
                   in the segmentor window

output:
    selected_verts (list) - the final list of vertex coordinates defined by the user clicking within the segmentor window
    mask# (array) - A 2D bool. numpy array that is a mask of all the points between the outline created by the vertices in selected_verts
'''

# on_select callback for CT cross-section 1
def on_select1(verts):
    global selected_verts1, mask1

    selected_verts1 = [(float(x), float(y)) for x, y in verts]

    # Update the filled polygon
    filled_polygon1.set_xy(selected_verts1)
    filled_polygon1.set_visible(True)
    fig.canvas.draw_idle()

    # Generate mask
    height, width = img1.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x.flatten(), y.flatten())).T

    poly_path = Path(selected_verts1)
    mask1 = poly_path.contains_points(points).reshape((height, width))


# on_select callback for CT cross-section 2
def on_select2(verts):
    global selected_verts2, mask2

    selected_verts2 = [(float(x), float(y)) for x, y in verts]

    # Update the filled polygon
    filled_polygon2.set_xy(selected_verts2)
    filled_polygon2.set_visible(True)
    fig.canvas.draw_idle()

    # Generate mask
    height, width = img2.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x.flatten(), y.flatten())).T

    poly_path = Path(selected_verts2)
    mask2 = poly_path.contains_points(points).reshape((height, width))


# on_select callback for CT cross-section 3
def on_select3(verts):
    global selected_verts3, mask3

    selected_verts3 = [(float(x), float(y)) for x, y in verts]

    # Update the filled polygon
    filled_polygon3.set_xy(selected_verts3)
    filled_polygon3.set_visible(True)
    fig.canvas.draw_idle()

    # Generate mask
    height, width = img3.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x.flatten(), y.flatten())).T

    poly_path = Path(selected_verts3)
    mask3 = poly_path.contains_points(points).reshape((height, width))

'''
function: 
    Produces an interactive window that creates 3 segments corresponding to a cross-section of the inputted CT slices
    from the im_dict.  
    
input: 
    im_dict (dict) - dictionary containing the 2D arrays for each x,y,z cross-section of the CT scan data
    center_dict (dict) - dictionary that captures the coordinate intersection (list) of the 2d planes used to segment data

output:
    fig - matplotlib figure containing 3 subplots
    ax - axis of the matplotlib figure where each axis corresponds to each of the subplots
    img# - 2D numpy array of specified cross-section from im_dict
    filled_polygon# - initializes a N,2 array for a cross-section segment that corresponds to the vertices chosen by the user (used in on_select func)
    selector# - initializes widget function for a cross-section segment that tracks users clicks (dependent on on_select function)
    Note: variables are global so they can used by the on_select# function above
    
    mask# (list) - list of tuples containing the array of 2D bool. masks corresponding to the polygons drawn by users for each of the cross-sections
    vertices (list) - list of tuples containing the coordinates of all the vertices for each polygon for each cross-section
'''

# Load and process multiple images
def load_multi_img(im_dict, center_dict):
    global fig, ax, img1, img2, img3, filled_polygon1, filled_polygon2, filled_polygon3, selector1, selector2, selector3

    vertices = []
    masks = []

    # Create Subplot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 6))

    # Define coordinates and dot size
    coordinate1a = center_dict['Slices Center'][0][0]
    coordinate2a = center_dict['Slices Center'][0][1]

    # Save the images with a dot to a variable
    img1 = np.array(list(im_dict.values())[0])

    # Plot X-axis slice with center dot (coordinates are flipped because going from Pyvista to Matplot)
    ax[0].imshow(img1, cmap='viridis', origin = 'lower')
    ax[0].scatter(coordinate2a, coordinate1a, color='red', s=100, marker='o')
    ax[0].set_title(f"Click to define a polygon for slice:{0}")

    # Set up the polygon patch
    filled_polygon1 = Polygon([[0, 0], [0, 0]], closed=True,
                              facecolor='yellow', edgecolor='black', alpha=0.4)
    filled_polygon1.set_visible(False)
    ax[0].add_patch(filled_polygon1)

    # Create the PolygonSelector
    selector1 = PolygonSelector(ax[0], on_select1, useblit=True,
                                props=dict(color='yellow', linestyle='-', linewidth=2, alpha=0.8))

    # Define coordinates and dot size
    coordinate1b = center_dict['Slices Center'][1][0]
    coordinate2b = center_dict['Slices Center'][1][1]

    # Save the images with a dot to a variable
    img2 = np.array(list(im_dict.values())[1])

    # Plot Y-axis slice with center dot (coordinates are flipped because going from Pyvista to Matplot)
    ax[1].imshow(img2, cmap='viridis', origin = 'lower')
    ax[1].scatter(coordinate2b, coordinate1b, color='red', s=100, marker='o')
    ax[1].set_title(f"Click to define a polygon for slice:{1}")

    # Set up the polygon patch
    filled_polygon2 = Polygon([[0, 0], [0, 0]], closed=True,
                              facecolor='yellow', edgecolor='black', alpha=0.4)
    filled_polygon2.set_visible(False)
    ax[1].add_patch(filled_polygon2)

    # Create the PolygonSelector
    selector2 = PolygonSelector(ax[1], on_select2, useblit=True,
                                props=dict(color='yellow', linestyle='-', linewidth=2, alpha=0.8))

    # Define coordinates and dot size
    coordinate1c = center_dict['Slices Center'][2][0]
    coordinate2c = center_dict['Slices Center'][2][1]

    # Save the images with a dot to a variable
    img3 = np.array(list(im_dict.values())[2])

    # Plot Z-axis slice with center dot (coordinates are flipped because going from Pyvista to Matplot)
    ax[2].imshow(img3, cmap='viridis', origin = 'lower')
    ax[2].scatter(coordinate2c, coordinate1c, color='red', s=100, marker='o')
    ax[2].set_title(f"Click to define a polygon for slice:{2}")

    # Set up the polygon patch
    filled_polygon3 = Polygon([[0, 0], [0, 0]], closed=True,
                              facecolor='yellow', edgecolor='black', alpha=0.4)
    filled_polygon3.set_visible(False)
    ax[2].add_patch(filled_polygon3)

    # Create the PolygonSelector
    selector3 = PolygonSelector(ax[2], on_select3, useblit=True,
                                props=dict(color='yellow', linestyle='-', linewidth=2, alpha=0.8))

    # Show plot but block remaining code until window is closed so the following variables are created
    plt.show(block = True)

    # Append vertices and 2D segmentation masks to their respective lists
    vertices.append((selected_verts1, selected_verts2, selected_verts3))
    masks.append((mask1, mask2, mask3))

    return vertices, masks


'''
function: 
    Applies the functions created directly above into a single process and feeds them the 2D array cross-sections from 
    a data dictionary. It produces a window where the user can see the xyz cross-sections with identifying dots from 
    the segmenting planes of the pyvista GUI. Allows user to isolate certain polygons within the 2D cross-section image
    in order to annotate/mask the 2D views (x,y,z) of the CT object.

input: 
    data_dict (dict) - dictionary containing the 2D arrays for each x,y,z cross-section of the CT scan data
    dataset_name (str) - name provided by the user to label the set of 2D array cross-section masks

output:
    selected_vertices (list) - list of tuples containing the coordinates of all the vertices for each polygon for each cross-section
    mask_list - list of tuples containing the array of 2D masks corresponding to the polygons drawn by users for each of the cross-sections
'''

def pv_to_image_mask(data_dict : dict, dataset_name : str):

    # Initialize image dictionary; length corresponds to number of slices in data dictionary
    image_dict = {i : None for i in range(0, len(data_dict[dataset_name]['Slices']))}

    #Convert Numpy array 'slice' from data dictionary dataset to image format
    for i, slice in enumerate(data_dict[dataset_name]['Slices']):
        image_dict[i] = Image.fromarray((slice * 255).astype(np.uint8)).convert("RGBA")

    # Call the function with the correct group name
    selected_vertices, selected_masks = load_multi_img(image_dict, data_dict[dataset_name])

    #apply masks to slice arrays
    mask_list = []
    for j, slice in enumerate(data_dict[dataset_name]['Slices']):
        mask_list.append(selected_masks[0][j]) #accesses mask list and inner tuple

    return  selected_vertices, mask_list

'''
function: 
    A function that takes the 2D masks for each CT cross-section and repeats them across the entire 3D space occupied by
    the 3D CT object. It then combines each one of these repeated 2D masks to form a complete 3D mask that isolates the 
    polyhedra within the 3D CT object annotated by the user

input: 
    mask_dict (dict) - dictionary containing the 2D arrays for each x,y,z cross-section of the CT scan data
    dataset_name (str) - name provided by the user to label the set of 2D array cross-section masks
    threshold (int) - the value to which existing voxels should be changed to in correspondence to the 3D mask generated
    ct_stack (array) - the 3D array which the 3D CT rendering is based off of (defines the resolution of 3D object)
    invert_mask (bool) - True/False value that determines if the 3D mask should be inverted or not. This is helpful when 
                         the user wants to apply the mask to the 3D rendering to overwrite the existing values (False) or
                         isolate the polyhedra and zero out the rest of the existing voxel values (True)
    
output:
   ct_mask - 3D mask (normal or inverted) to be applied to the 3D array and subsequently the 3D rendering of CT object
'''

def mask_ct_vol(masks_dict : dict, dataset_name : str,  threshold : int, ct_stack = None, invert_mask = False):

    #Combine masks from each axis to make an overall mask
    mask_3d_x = np.repeat(masks_dict[dataset_name]['Slices'][0][np.newaxis, :, :], ct_stack.shape[0], axis=0)
    mask_3d_y = np.repeat(masks_dict[dataset_name]['Slices'][1][:, np.newaxis, :], ct_stack.shape[1], axis=1)
    mask_3d_z = np.repeat(masks_dict[dataset_name]['Slices'][2][:, :, np.newaxis], ct_stack.shape[2], axis=2)
    combined3d_mask = mask_3d_x & mask_3d_y & mask_3d_z

    # apply mask to 3d volume
    if invert_mask:
        ct_mask = np.where(combined3d_mask, threshold, ct_stack)

    else:
        ct_mask = np.where(combined3d_mask, ct_stack, 0)


    return ct_mask