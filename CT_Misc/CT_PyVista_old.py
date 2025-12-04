#%%

import importlib
from CT_Import_Helper import *
from pv_helper import *
import Custom_CMAP

# Create a custom_cmap
importlib.reload(Custom_CMAP) #reload any changes


threshold = 80
custom_cmap = Custom_CMAP.create_cmap(threshold = threshold)


'''
data = np.arange(0, 256).reshape(16,16)

# Plotting with the custom colormap
plt.imshow(data, cmap=custom_cmap)
plt.colorbar(label='Data Value')
plt.title('Plot with Custom Colormap')
plt.show()
'''


# Switch renderer to plot 3d
matplotlib.use('TkAgg') # or 'Qt5Agg'

# V3 (Working)

def execution_timer(start_time, message):
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    print(f"{message} {exec_time:.4f} seconds")

# Render the volume image, check rendering time (works)
def pyrender(grid):
    global plane_cords, slice_x, slice_y, slice_z

    # Set plotter shape, col and row weighs control size of window and plotter output size
    plotter = pv.Plotter(shape=(3,2), col_weights=[1, 1], row_weights=[1,1,1], groups = [([0,2], 0), (0, 1), (1, 1), (2, 1)])

    plotter.subplot(0,0)  # 23 Seconds - Array Size: 474,449, 362
    start_time = time.perf_counter()
    plotter.add_volume_clip_plane(grid, normal='x', cmap=custom_cmap, opacity = 'sigmoid', invert = False,
                                  assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12)


    # Control Initial Zoom
    plotter.camera.zoom(2)

    # Update all plane's coordinates
    plane_cords['x'] = plotter.plane_widgets[0].GetOrigin()[0]
    plane_cords['y'] = plotter.plane_widgets[0].GetOrigin()[1]
    plane_cords['z'] = plotter.plane_widgets[0].GetOrigin()[2]

    # Print plotting time
    execution_timer(start_time, 'Plot 1 Execution time: ')

    # Add text directions for functionality
    plotter.add_text(text="Press r to re-render images after adjusting planes", font_size=12, position="upper_left")
    plotter.add_text(text="Press s to save slices", font_size=12, position="lower_left")

    #Compute Origin Coordinates
    orig = [i for i in plotter.plane_widgets[0].GetOrigin()]

    # Create subplots
    plotter.subplot(0,1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    def clip_with_planex(normal, origin):
        # Define the plane widget's origin and normal vector direction base on the normal and origin inputs
        clipped = single_slice_x.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)

        # Remove previous clipped mesh
        plotter.remove_actor("mesh_clip", render=False)
        plotter.render()

    single_slice_x = grid.slice(normal='x', origin=orig)
    slice_x = condensed_stack[int(orig[0]), :, :]
    plotter.add_mesh(single_slice_x, cmap=custom_cmap, opacity = 'sigmoid', name='x view', show_scalar_bar=False)
    plotter.add_plane_widget(clip_with_planex, normal='y', origin = orig, normal_rotation=False)
    plotter.add_plane_widget(clip_with_planex, normal='z', origin = orig, normal_rotation=False)
    plotter.add_text(text=f"X Axis Slice (Y >, Z ^)", font_size=10, position="upper_left")


    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[2].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    prop = plotter.plane_widgets[3].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    # Sets camera view and initial zoom
    plotter.camera_position = 'yz'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 2 Execution time: ')


    plotter.subplot(1,1)  # 35 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    def clip_with_planey(normal, origin):
        clipped = single_slice_y.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)
        plotter.remove_actor("mesh_clip", render=False)  # Remove previous clipped mesh
        plotter.render()

    single_slice_y = grid.slice(normal='y', origin=orig)
    slice_y = condensed_stack[:, int(orig[1]), :]
    plotter.add_mesh(single_slice_y, cmap=custom_cmap, opacity = 'sigmoid', name='y view', show_scalar_bar=False)
    plotter.add_plane_widget(clip_with_planey, normal='x', normal_rotation=False, origin=orig)
    plotter.add_plane_widget(clip_with_planey, normal='z', normal_rotation=False, origin=orig)
    plotter.add_text(text="Y Axis Slice (X >, Z ^)", font_size=10, position="upper_left")

    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[4].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    prop = plotter.plane_widgets[5].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    # Sets camera view and initial zoom
    plotter.camera_position = 'xz'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 3 Execution time: ')


    plotter.subplot(2,1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    def clip_with_planez(normal, origin):
        clipped = single_slice_z.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)
        plotter.remove_actor("mesh_clip", render=False)  # Remove previous clipped mesh
        plotter.render()

    single_slice_z = grid.slice(normal='z', origin=orig)
    slice_z = condensed_stack[:, :, int(orig[2])]
    plotter.add_mesh(single_slice_z, cmap=custom_cmap, opacity = 'sigmoid', name='z view', show_scalar_bar=False)
    plotter.add_plane_widget(clip_with_planez, normal='x', normal_rotation=False, origin=orig)
    plotter.add_plane_widget(clip_with_planez, normal='y', normal_rotation=False, origin=orig)
    plotter.add_text(text=f"Z Axis Slice (X >, Y ^)", font_size=10, position="upper_left")

    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[6].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    prop = plotter.plane_widgets[7].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    # Sets camera view and initial zoom
    plotter.camera_position = 'xy'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 4 Execution time: ')


    return plotter


# Define a callback function to re-render
def re_render():
    global plane_cords, slice_x, slice_y, slice_z

    # Record current plane coordinates; each sublist are the coordinates of x, y, z in each of the planes displayed
    # When using add_volume_clip 2 planes are added instead of 1 with other methods
    curr_cords = [list(plotter.plane_widgets[0].GetOrigin()), list(plotter.plane_widgets[1].GetOrigin()),
                  list(plotter.plane_widgets[2].GetOrigin()), list(plotter.plane_widgets[3].GetOrigin()),
                  list(plotter.plane_widgets[4].GetOrigin()), list(plotter.plane_widgets[5].GetOrigin()),
                  list(plotter.plane_widgets[6].GetOrigin()), list(plotter.plane_widgets[7].GetOrigin())]

    # Checks for changes in any of the x, y, z coordinates between planes displayed to re-adjust them when re-rendering
    plane_cords = cord_compare(curr_cords, plane_cords)

    # Create new origin point for each slice
    orig = [max(i, 0) for i in plane_cords.values()]
    print(orig)

    # Clear all plane widgets and graphs
    plotter.clear_plane_widgets()

    # Re-render plane widgets and graphs
    plotter.subplot(0,0)

    # plotter.add_volume(grid, opacity='sigmoid')
    start_time = time.perf_counter()
    plotter.clear()
    plotter.add_volume_clip_plane(grid, normal='x', origin = orig, cmap=custom_cmap, opacity = 'sigmoid', invert = False,
                                  assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12)


    # Print plotting time
    execution_timer(start_time, 'Plot 1 Execution time: ')



    plotter.subplot(0,1)
    start_time = time.perf_counter()

    def clip_with_planex(normal, origin):
        clipped = single_slice_x.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)
        plotter.remove_actor("mesh_clip", render=False)  # Remove previous clipped mesh
        #plotter.render()

    single_slice_x = grid.slice(normal='x', origin=orig)
    slice_x = condensed_stack[int(plane_cords['x']), :, :]
    plotter.add_mesh(single_slice_x, cmap=custom_cmap, opacity = 'sigmoid', name='x view', show_scalar_bar=False)
    plotter.add_plane_widget(clip_with_planex, normal='y', origin = orig, normal_rotation=False)
    plotter.add_plane_widget(clip_with_planex, normal='z', origin = orig, normal_rotation=False)
    plotter.add_text(text=f"X Axis Slice (Y >, Z ^)", font_size=10, position="upper_left")


    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[2].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    prop = plotter.plane_widgets[3].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    # Sets camera view
    plotter.camera_position = 'yz'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 2 Execution time: ')


    plotter.subplot(1,1)
    start_time = time.perf_counter()

    def clip_with_planey(normal, origin):
        clipped = single_slice_y.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)
        plotter.remove_actor("mesh_clip", render=False)  # Remove previous clipped mesh
        plotter.render()

    single_slice_y = grid.slice(normal='y', origin=orig)
    slice_y = condensed_stack[:, int(plane_cords['y']), :]
    plotter.add_mesh(single_slice_y, cmap=custom_cmap, opacity = 'sigmoid', name='y view', show_scalar_bar=False)
    plotter.add_plane_widget(clip_with_planey, normal='x', origin = orig, normal_rotation=False)
    plotter.add_plane_widget(clip_with_planey, normal='z', origin = orig, normal_rotation=False)
    plotter.add_text(text="Y Axis Slice (X >, Z ^)", font_size=10, position="upper_left")

    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[4].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 7 (adjust as needed)

    prop = plotter.plane_widgets[5].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 7 (adjust as needed)

    # Sets camera view
    plotter.camera_position = 'xz'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 3 Execution time: ')


    plotter.subplot(2,1)
    start_time = time.perf_counter()

    def clip_with_planez(normal, origin):
        clipped = single_slice_z.clip(normal=normal, origin=origin)
        plotter.add_mesh(clipped, name="clipped_mesh", opacity=0, render=False, show_scalar_bar=False)
        plotter.remove_actor("mesh_clip", render=False)  # Remove previous clipped mesh
        plotter.render()

    single_slice_z = grid.slice(normal='z', origin=orig)
    execution_timer(start_time, 'Single Slice Execution time: ')
    slice_z = condensed_stack[:, :, int(plane_cords['z'])]
    execution_timer(start_time, 'Array Execution time: ')
    plotter.add_mesh(single_slice_z, cmap=custom_cmap, opacity = 'sigmoid', name='z view', show_scalar_bar=False)
    execution_timer(start_time, 'Mesh Execution time: ')
    plotter.add_plane_widget(clip_with_planez, normal='x', origin = orig, normal_rotation=False)
    execution_timer(start_time, 'x plane Execution time: ')
    plotter.add_plane_widget(clip_with_planez, normal='y', origin = orig, normal_rotation=False)
    execution_timer(start_time, 'y plane Execution time: ')
    plotter.add_text(text=f"Z Axis Slice (X >, Y ^)", font_size=10, position="upper_left")

    # Increase Plane Widget Line Width
    prop = plotter.plane_widgets[6].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    prop = plotter.plane_widgets[7].GetEdgesProperty()  # Get the property object
    prop.SetLineWidth(7)  # Set the line width to 5 (adjust as needed)

    # Sets camera view
    plotter.camera_position = 'xy'
    plotter.camera.zoom(2)

    # Print plotting time
    execution_timer(start_time, 'Plot 4 Execution time: ')

    #Update and Rerender Plot
    plotter.update()
    #plotter.render()


def save_slices():
    global slices_dict, masked_slices, condensed_stack_copy

    # Capture plane widget locations from GUI
    orig = [max(i, 0) for i in plane_cords.values()]

    # User input for group of slices name
    while True:
        slices_name = diag_box()

        if slices_name.lower() not in [key.lower() for key in slices_dict.keys()]:
            break
        else:
            show_info_message("That name is taken. Please provide new name")


    slices_dict[slices_name] = {'Slices' : [slice_x, slice_y, slice_z],
                                'Slices Center' : [[orig[1], orig[2]],[orig[0], orig[2]],[orig[0], orig[1]]]}

    show_info_message("Slices saved")

    # Apply mask of saved chamber; update and rerender plot
    masked_slices[slices_name] = {'Slices': None, 'Vertices': None}
    vertices, masks = pv_to_image_mask(slices_dict, slices_name)
    masked_slices[slices_name]['Slices'] = masks
    masked_slices[slices_name]['Vertices'] = vertices


    # Apply masks to the underlying grid data in the Pyvista renderer
    condensed_stack_masked_inv = mask_ct_vol(masks_dict=masked_slices, dataset_name=slices_name,
                                         ct_stack=condensed_stack_copy, invert_mask = True, threshold= threshold)

    grid.cell_data["values"] = condensed_stack_masked_inv.ravel(order="F")
    condensed_stack_copy = condensed_stack_masked_inv

    print((grid.cell_data["values"] == threshold).sum())

    re_render()


#%%

# Specify the directory, get .tiff metadata, and create tiff stack numpy object
# directory = r'C:\Users\pseco\Documents\Mirco CT Dissertation\\Proj-Forams\Data_Files\716-4L-HEX-1'
directory = r'C:\Users\pseco\Documents\Micro_CT_Dissertation\Proj-Forams\Data_Files\716-6L-HEX-2'

tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path = directory, condense = True, im_pooling = True,
                                                        kernel_size = 3, stride = 4, verbose = False)


#%%


#Create copy of the condensed stack to perform augmentations to the underlying data during segmentation process
condensed_stack_copy = copy.deepcopy(condensed_stack)

# Delete Plotter object if exists

if 'plotter' in globals():
    plotter.close()
    print('Plotter Cleared')


#Create Pyvista plotter object and plot to annotate object
grid = pv.ImageData()
grid.dimensions = np.array(condensed_stack_copy.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = condensed_stack_copy.ravel(order="F")  # Use Fortran ordering for flattening
plane_cords = {i : None for i in ["x", "y", "z"]}  # Used to track past plane coordinates
slices_dict = {}  # Used to track 2D slices
masked_slices = {}

print((grid.cell_data["values"]  == threshold).sum())


# Generate Plotter
plotter = pyrender(grid)

# Add a key event to the 'r' key
plotter.add_key_event("r", re_render)
plotter.add_key_event("s", save_slices)
#plotter.add_key_event("m", toggle_on_distance_tool)
#plotter.add_key_event("f", toggle_off_distance_tool)

# Display plot
plotter.show()


#%%
# About 33 min to annotate one Foram
fname = directory.split('\\')[-1]+'_dict'
direct = r'C:\Users\pseco\Documents\Mirco CT Dissertation\Proj-Forams\Data_Files\Annotated_Foram_Dicts'
save_3d_annotations(dict = masked_slices, ct_array = condensed_stack, directory=direct, filename=fname)



