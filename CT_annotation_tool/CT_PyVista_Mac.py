#%%

import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(repo_root, 'CT_annotation_tool'))
sys.path.append(os.path.join(repo_root, 'EllipFit_EM'))

import importlib
import csv
import matplotlib
import pyvista as pv
from CT_annotation_tool.CT_Import_Helper_Mac import *
from CT_annotation_tool.pv_helper_Mac import *
from EllipFit_EM.ellipsoid_fitting import *
import Custom_CMAP

#%%

# Switch renderer to plot 3d
matplotlib.use('TkAgg') # 'TkAgg' or 'Qt5Agg'

# Function that assigns callbacks to buttons to Pyvista GUI for segmenting the 3D foram
def top_buttons():
    plotter.add_checkbox_button_widget(callback=re_render, value=False, position=(10, 1950), color_on='blue', color_off='blue')
    plotter.add_text("Rerender", position=(75, 1960), color="black", font_size=12, shadow=True)

    plotter.add_checkbox_button_widget(callback=save_slices, value=False, position=(200, 1950), color_on='blue', color_off='blue')
    plotter.add_text("Save Slices", position=(265, 1960), color="black", font_size=12, shadow=True)

    plotter.add_checkbox_button_widget(callback=save_slice_collection, value=False, position=(420, 1950), color_on='blue',
                                       color_off='blue')
    plotter.add_text("Save Annotations", position=(480, 1960), color="black", font_size=12, shadow=True)

    plotter.add_checkbox_button_widget(callback=ellipsoid_fit, value=False, position=(700, 1950), color_on='blue',
                                       color_off='blue')
    plotter.add_text("Fit Chambers", position=(765, 1960), color="black", font_size=12, shadow=True)

# Function that assigns callbacks to buttons to Pyvista GUI for adjusting 3D foram chambers
def top_buttons2():
    plotter.add_checkbox_button_widget(callback=ellipsoid_fit, value=True, position=(10, 1950), color_on='blue',
                                       color_off='blue')
    plotter.add_text("Fit Chambers", position=(75, 1960), color="black", font_size=12, shadow=True)

    plotter.add_checkbox_button_widget(callback=update_ellipsoid_dict, value=True, position=(250, 1950), color_on='blue',
                                       color_off='blue')
    plotter.add_text("Update Ellipsoid Parameters", position=(315, 1960), color="black", font_size=12, shadow=True)

    plotter.add_checkbox_button_widget(callback=export_chamber_params, value=True, position=(650, 1950), color_on='blue',
                                       color_off='blue')
    plotter.add_text("Export Chambers Parameters", position=(715, 1960), color="black", font_size=12, shadow=True)

# Function that creates a series of radio button that toggle the view of the GUI between 3D foram and Chambers
def radio_buttons():

    # Creates list depending on if the 3D chambers have been fitted or not
    if 'ellip_params' in globals() and len(masked_slices) > 0:
        labels = ['3D Foram'] + list(ellip_params.keys())
    else:
        labels = ['3D Foram']

    # Iteratively creates radio buttons that is dependent on the list above
    for i, chamber in enumerate(labels):

        _ = plotter.add_radio_button_widget(
            set_annotation(str(labels[i])),
            'annotations',
            position=(1675, 1945 - (i*60)),
            size=50,
            color_on='blue',
            color_off='grey',
            title=chamber,
        )

# Callback that determines the rendering and buttons of the pyvista GUI depending on what view option is chosen
def set_annotation(annotation):
    def wrapped_callback():
        global selected_annotation

        #Update global variable to determine which button is active (each button associated with a chamber)
        selected_annotation = annotation

        # Check the value of the radio button currently selected and executes renders the object in the GUI accordingly
        if annotation == "3D Foram":

            selected_annotation = annotation

            # Clear all actors from all subplots
            plotter.clear()
            plotter.clear_actors()
            plotter.clear_slider_widgets()
            plotter.clear_button_widgets()

            #Create recreate all the initial plotter objects
            pyrender(grid)

        else:

            # Clear all actors from all subplots
            plotter.clear_actors()
            plotter.clear_slider_widgets()
            plotter.clear_button_widgets()

            #Initiate buttons and create meshes to show chamber fitting
            plotter.subplot(0, 0)
            top_buttons2()
            radio_buttons()

            # Gather chamber voxel data and ellipsoid parameter data
            voxel_data = masked_slices[annotation]['3D Tensor']
            ellip = generate_ellipsoid_3d(ellip_params[annotation]['computed params'])

            # Plot the ellipsoid
            plotter.add_mesh(
                ellip,
                color="skyblue",
                show_edges=True,
                edge_color="black",
                smooth_shading=True,
                name='ellipsoid',
            )

            # Plot actual chamber
            grid_temp = pv.ImageData()
            grid_temp.dimensions = np.array(voxel_data.shape) + 1
            grid_temp.origin = (0, 0, 0)
            grid_temp.spacing = (1, 1, 1)
            grid_temp.cell_data["values"] = voxel_data.ravel(order="F")  # Use Fortran ordering for flattening

            plotter.add_volume(grid_temp, show_scalar_bar=False, opacity='sigmoid', cmap=custom_cmap)

            chamber_sliders(ellip_params[annotation]['computed params'])

            #Add real world measurements to adjacent subplot given current saved ellipsoid parameters
            plotter.subplot(0, 1)
            chamber_measurements(annotation)


    return wrapped_callback

# Define callback to save current annotated slices (User Input)
def save_slices(state):
    global slices_dict, masked_slices, condensed_stack_copy, locked_state

    if not locked_state:

        # Lock buttons
        locked_state = True

        # Capture plane widget locations from GUI
        orig = [max(i, 0) for i in plane_cords.values()]

        # User input for group of slices name
        slices_name = chamber_diag_box_mac()

        # Checks if the inputted name is viable; if not cancels process
        if slices_name is not None and slices_name != '':
            if slices_name.lower() not in [key.lower() for key in slices_dict.keys()]:

                # this is here in case the user prematurely closes out of the manual annotation window
                try:

                    # create new annotation key
                    slices_dict[slices_name] = {'Slices': [slice_x, slice_y, slice_z],
                                                'Slices Center': [[orig[1], orig[2]], [orig[0], orig[2]],
                                                                  [orig[0], orig[1]]]}

                    # Apply mask of saved chamber; update and rerender plot
                    masked_slices[slices_name] = {'Slices': None, 'Vertices': None}
                    vertices, masks = pv_to_image_mask(slices_dict, slices_name)
                    masked_slices[slices_name]['Slices'] = masks
                    masked_slices[slices_name]['Vertices'] = vertices

                    # Apply masks to the underlying grid data in the Pyvista renderer
                    condensed_stack_masked_inv = mask_ct_vol(masks_dict=masked_slices, dataset_name=slices_name,
                                                             ct_stack=condensed_stack_copy, invert_mask=True,
                                                             threshold=threshold)

                    # Update values in the grid that controls the 3D CT object rendering
                    grid.cell_data["values"] = condensed_stack_masked_inv.ravel(order="F")
                    condensed_stack_copy = condensed_stack_masked_inv

                    show_info_message_mac("Slices saved")

                    re_render(1)

                    # Unlock buttons
                    locked_state = False

                except Exception as e:
                    show_info_message_mac(f"Error {e}! Ensure all segmentations are completed before exiting. Slices not saved")

                    # Remove key that was added dictionary because of code failure.
                    del slices_dict[slices_name]

                    # Unlock buttons
                    locked_state = False

            else:
                show_info_message_mac("That name is taken. Please provide new name")

                # Unlock buttons
                locked_state = False

        else:
            show_info_message_mac("Invalid name provided. Process cancelled")

            # Unlock buttons
            locked_state = False

# Define callback to save all annotated slices (User Input)
def save_slice_collection(state):
    global locked_state

    if not locked_state:
        # Lock buttons
        locked_state = True

        # Threshold parameters is provided by code via a voxel analysis
        if len(masked_slices) != 0:
            try:
                # Using the directory from the foram_import function to create a file name and prompt user where to save it
                fname = directory.split('/')[-1] + '_dict'
                direct = filepath_diag_box_mac("Input folder location annotations should be saved to")
                save_3d_annotations(im_dict=masked_slices, ct_array=condensed_stack, directory=direct, filename=fname, threshold=threshold)
                locked_state = False

                show_info_message_mac("Annotations saved")

            except FileNotFoundError:
                show_info_message_mac("Directory path not found. Save cancelled, please retry.")
                locked_state = False

            except Exception as e:
                show_info_message_mac(f"An issue occurred: {e}. Save cancelled, please retry.")
                locked_state = False

        else:
            show_info_message_mac("Please Save at least one set of annotation slices before saving collection")
            locked_state = False

# Define callback to perform initial fitting of ellipsoids to chambers (User Input)
def ellipsoid_fit(state):
    global masked_slices, ellip_params, locked_state

    if not locked_state:

        # Lock buttons
        locked_state = True

        file = filepath_diag_box_mac("Input .pkl file location of annotations that should be used")

        # Check to see if filename is viable
        if file != '' and file is not None:
            try:

                # Import the pkl file provided that holds the 3D masks of the annotations
                masked_slices = open_3d_annotations(file)

                # Fit ellipsoids to the 3D arrays of the annotations
                ellip_params = chamber_fitting(masked_slices, threshold=80, scaling_params = scale_xyz, pixel_measurement = RW_pixel_size)
                show_info_message_mac("Chambers fitted")

                plotter.subplot(0,0)
                radio_buttons()

                # Unlock buttons
                locked_state = False

            except FileNotFoundError:
                show_info_message_mac("No PKL file found. Please try again")

                # Unlock buttons
                locked_state = False

            except Exception as e:
                show_info_message_mac(f"An error {e} occurred while loading the PKL file. Please Ensure the annotations files is a PKL file")

                # Unlock buttons
                locked_state = False

#Function to iteratively create the chamber parameter sliders
def chamber_sliders(params):
    global is_initialized

    # Create lists for min-max range for sliders; last 3 are roll, pitch, yaw, respectively
    lower_range_params = [0, 0, 0] + [0, 0, 0] + [-180, -90, -180]
    upper_range_params = list(condensed_stack.shape) + list(condensed_stack.shape) + [180, 90, 180]

    # Create a list of titles for sliders
    titles = ['X Axis Center', 'Y Axis Center', 'Z Axis Center', 'X Radius Width', 'Y Radius Width', 'Z Radius Width', 'Roll',
              'Pitch', 'Yaw']

    for i, param in enumerate(params):

        # Define the flag variable to prevent function when the slider widgets are being initialized
        is_initialized = False

        # Create a bottom row of slider widgets for several of the ellipsoid parameters
        if i < 4:

            # Iteratively create a slider widget based on the provided range parameters, initial value, title, etc.
            slider_widget = plotter.add_slider_widget(
                callback=refit_chamber,
                rng=[lower_range_params[i], upper_range_params[i]],
                value=param,
                title=titles[i],
                pointa=(.00 + (i * .2), .1),  # (x-axis, y-axis)
                pointb=(.00 + ((i + 1) * .2), .1),
                slider_width=.01,
                tube_width=.01,
                title_height=0.015,
                fmt='%.2f',
                style='modern',
                interaction_event='always',
            )

            slider_widget.GetRepresentation().SetLabelHeight(0.015)

        # Create a top row of slider widgets for several of the ellipsoid parameters
        else:

            # Iteratively create a slider widget based on the provided range parameters, initial value, title, etc.
            slider_widget = plotter.add_slider_widget(
                callback=refit_chamber,
                rng=[lower_range_params[i], upper_range_params[i]],
                value=param,
                title=titles[i],
                pointa=(.00 + ((i - 4) * .2), .035),  # (x-axis, y-axis)
                pointb=(.00 + ((i - 4 + 1) * .2), .035),
                slider_width=.01,
                tube_width=.01,
                title_height=0.015,
                fmt='%.2f',
                style='modern',
                interaction_event='always',
            )

            slider_widget.GetRepresentation().SetLabelHeight(0.015)

#Function to add real word measurements of chambers to GUI
def chamber_measurements(annotation):

    # Create titles for the text information
    titles = ["Chamber Center: ", "X Diameter: ", "Y Diameter: ", "Z Diameter: ", "Roll: ", "Pitch: ", "Yaw: ",
              "Chamber Volume: "]

    plotter.add_text("Real World Measurements", position=(10, 625), color="black", font_size=12, shadow=True)

    # Iteratively create estimated real world measurements of fitted ellipsoids using ellipsoid parameters and CT imaging metadata
    for i, title in enumerate(titles):
        if i == 0:
            val = np.round(ellip_params[annotation]['scaled params'][i:i + 3], 3)
            plotter.add_text(title + str(tuple(val)), position=(10, 575 - (i * 50)), color="black",
                             font_size=12, shadow=True)

        elif i < 4:
            val = round(ellip_params[annotation]['scaled params'][i] * 2, 3)
            plotter.add_text(title + str(val) + "μm", position=(10, 575 - (i * 50)), color="black",
                             font_size=12, shadow=True)

        elif i < 7:
            val = round(ellip_params[annotation]['scaled params'][i], 3)
            plotter.add_text(title + str(val), position=(10, 575 - (i * 50)), color="black", font_size=12,
                             shadow=True)

        else:
            val = round(((4 / 3) * np.pi * ellip_params[annotation]['scaled params'][3] *
                         ellip_params[annotation]['scaled params'][4] *
                         ellip_params[annotation]['scaled params'][5]), 3)
            plotter.add_text(title + str(val) + "μm\u00b3", position=(10, 575 - (i * 50)), color="black",
                             font_size=12, shadow=True)

#Callback for the chamber parameter sliders when the values are updated
def refit_chamber(val):
    global new_ellip_params

    # Check if widget has been created yet
    global is_initialized

    if not is_initialized:
        is_initialized = True
        return


    plotter.subplot(0,0)

    # Clear current ellipsoid
    plotter.remove_actor("my sphere")

    # Create list to store new ellipsoid parameters according to the current value of widget sliders
    new_ellip_params = []

    for widget in plotter.slider_widgets:
        new_ellip_params.append(widget.GetSliderRepresentation().GetValue())

    # Generate new ellipsoid mesh
    ellipsoid = generate_ellipsoid_3d(new_ellip_params)

    # Plot the new ellipsoid
    plotter.add_mesh(
        ellipsoid,
        color="skyblue",
        show_edges=True,
        edge_color="black",
        smooth_shading=True,
        name = "ellipsoid",
    )

# Callback that updates the ellipsoid params values in the ellipsoid params dictionary (User Input)
def update_ellipsoid_dict(state):
    global locked_state

    if not locked_state:

        # Lock buttons
        locked_state = True

        user_input = filepath_diag_box_mac("Overwrite ellipsoid parameters (Y/N)?")

        # Check if user wants to update ellipsoid parameters and check to make sure the ellipsoid parameters dictionary exists
        if user_input == "Y":
            if 'ellip_params' in globals() and len('ellip_params') > 0:

                # Update ellipsoid parameter dictionary with new parameters based on current ellipsoid widget values
                ellip_params[selected_annotation]['computed params'] = new_ellip_params
                ellip_params[selected_annotation]['scaled params'] = (
                            [x * y * RW_pixel_size for x, y in zip(new_ellip_params[0:3], scale_xyz)] +
                            [x * y * RW_pixel_size for x, y in zip(new_ellip_params[3:6], scale_xyz)] +
                            new_ellip_params[6:9])

                show_info_message_mac("Parameters updated")

                # Unlock buttons
                locked_state = False

            else:
                show_info_message_mac("Please annotate chambers first")

                # Unlock buttons
                locked_state = False

        else:

            # Unlock buttons
            locked_state = False

# Callback to save the chamber parameters (User Input)
def export_chamber_params(state):

    global locked_state

    if not locked_state:

        # Lock buttons
        locked_state = True

        # Prompt user for a folder location
        folder = filepath_diag_box_mac("Input folder location you would like to save chamber ellipsoid parameters CSV")

        # Check is user inputs are valid
        if folder != '' and folder is not None:

            try:

                file = os.path.join(folder, directory.split('/')[-1] + '_Params.csv')
                with open(file, "w", newline="") as f:

                    writer = csv.writer(f)

                    # Header: outer_key, inner_key, val1, val2, ...
                    header = ["Chamber", "Parameter Desc.", "Chamber Center X", "Chamber Center Y", "Chamber Center Z",
                              "X Radius", "Y Radius", "Z Radius", "Roll", "Pitch", "Yaw", "Chamber Volume"]

                    # Write header to CSV
                    writer.writerow(header)

                    # Write ellip_params rows (outer_key is the chamber name; inner_key is parameter Desc = [params, scaled_params]
                    for outer_key, inner_dict in ellip_params.items():
                        for inner_key, values in inner_dict.items():
                            if inner_key == 'scaled params':
                                vol = round(((4 / 3) * np.pi * values[3] * values[4] * values[5]), 3)
                                row = [outer_key, 'real world measurements'] + values + [vol]
                                writer.writerow(row)  # Write rows to CSV
                            else:
                                vol = 'N/A'
                                row = [outer_key, inner_key] + values + [vol]
                                writer.writerow(row)  # Write rows to CSV

                # Unlock buttons
                locked_state = False

            except FileNotFoundError:
                show_info_message_mac("Folder not found. Please try again.")

                # Unlock buttons
                locked_state = False

            except Exception as e:
                show_info_message_mac(f"An error occurred while saving the CSV file: {e}")

                # Unlock buttons
                locked_state = False

        else:
            show_info_message_mac("No file path entered. Please try again")

            # Unlock buttons
            locked_state = False

# Render the volume image, check rendering time (works)
def pyrender(grid):
    global plane_cords, slice_x, slice_y, slice_z

    plotter.subplot(0,0)  # 23 Seconds - Array Size: 474,449, 362
    start_time = time.perf_counter()

    # Create 3D rendering base on CT voxel data in the grid object
    foram_3d = plotter.add_volume_clip_plane(grid, normal='x', cmap=custom_cmap, opacity = 'sigmoid', invert = False,
                                  assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12, )

    # add buttons to the 3D object subplot
    top_buttons()
    radio_buttons()

    # Control Initial Zoom
    #plotter.camera.zoom(1.5)


    # Update all plane widget center coordinates
    plane_cords['x'] = plotter.plane_widgets[0].GetOrigin()[0]
    plane_cords['y'] = plotter.plane_widgets[0].GetOrigin()[1]
    plane_cords['z'] = plotter.plane_widgets[0].GetOrigin()[2]


    # plotting time
    execution_timer(start_time, 'Plot 1 Execution time: ')


    # Create subplot
    plotter.subplot(0,1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    # extract 3D tensor slice where the X slicer is currently located
    slice_x = condensed_stack[int(plane_cords['x']), :, :]

    # Create a matplotlib IM object (properly flip/rotate) and add it to the subplot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x_transform = np.fliplr(np.rot90(condensed_stack_copy[int(plane_cords['x']), :, :], k=-1))
    ax1.imshow(x_transform, cmap=custom_cmap)
    fig1.gca().invert_yaxis()
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.set_title(label=f"X Axis Slice (Y >, Z ^)", fontsize=10)

    vline = ax1.axvline(plane_cords['y'], color='r', linestyle='-')
    hline = ax1.axhline(plane_cords['z'], color='r', linestyle='-')

    chart1 = pv.ChartMPL(fig1)
    plotter.add_chart(chart1)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_y(val):
        vline.set_xdata([val])
        fig1.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_z(val):
        hline.set_ydata([val])
        fig1.canvas.draw_idle()

    # Add Y axis slider using above methods to adjust the 2D slicer planes
    y_slider = plotter.add_slider_widget(
        update_y,
        rng=[0, condensed_stack.shape[1]],
        value=plane_cords['y'],
        title='Y',
        pointa=(0.39, .05),  # (x-axis, y-axis)
        pointb=(0.635, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    # Add Z axis slider using above methods to adjust the 2D slicer planes
    z_slider = plotter.add_slider_widget(
        update_z,
        rng=[0, condensed_stack.shape[2]],
        value=plane_cords['z'],
        title='Z',
        pointa=(.71, .005),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.71, 1),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )

    plt.close()

    # Print plotting time
    execution_timer(start_time, 'Plot 2 Execution time: ')


    # Similar to above
    plotter.subplot(1,1)  # 35 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    slice_y = condensed_stack[:, int(plane_cords['y']), :]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    y_transform = np.fliplr(np.rot90(condensed_stack_copy[:, int(plane_cords['y']), :], k=-1))
    ax2.imshow(y_transform, cmap=custom_cmap)
    fig2.gca().invert_yaxis()
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.set_title(label=f"Y Axis Slice (X >, Z ^)", fontsize=10)

    vline2 = ax2.axvline(plane_cords['x'], color='r', linestyle='-')
    hline2 = ax2.axhline(plane_cords['z'], color='r', linestyle='-')

    chart2 = pv.ChartMPL(fig2)
    plotter.add_chart(chart2)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_x2(val):
        vline2.set_xdata([val])
        fig2.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_z2(val):
        hline2.set_ydata([val])
        fig2.canvas.draw_idle()

    x_slider2 = plotter.add_slider_widget(
        update_x2,
        rng=[0, condensed_stack.shape[0]],
        value=plane_cords['x'],
        title='X',
        pointa=(0.365, .05),  # (x-axis, y-axis)
        pointb=(0.66, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    z_slider2 = plotter.add_slider_widget(
        update_z2,
        rng=[0, condensed_stack.shape[2]],
        value=plane_cords['z'],
        title='Z',
        pointa=(.71, .125),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.71, .87),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )

    plt.close()


    # Print plotting time
    execution_timer(start_time, 'Plot 3 Execution time: ')

    # Similar to above
    plotter.subplot(2,1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    slice_z = condensed_stack[:, :, int(plane_cords['z'])]
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    z_transform = np.flipud(np.rot90(condensed_stack_copy[:, :, int(plane_cords['z'])]))
    ax3.imshow(z_transform, cmap=custom_cmap)
    fig3.gca().invert_yaxis()
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax3.set_title(label=f"Z Axis Slice (X >, Y ^)", fontsize=10)

    vline3 = ax3.axvline(plane_cords['x'], color='r', linestyle='-')
    hline3 = ax3.axhline(plane_cords['y'], color='r', linestyle='-')

    chart3 = pv.ChartMPL(fig3)
    plotter.add_chart(chart3)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_x3(val):
        vline3.set_xdata([val])
        fig3.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_y3(val):
        hline3.set_ydata([val])
        fig3.canvas.draw_idle()

    x_slider3 = plotter.add_slider_widget(
        update_x3,
        rng=[0, condensed_stack.shape[0]],
        value=plane_cords['x'],
        title='X',
        pointa=(0.3375, .05),  # (x-axis, y-axis)
        pointb=(0.688, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    y_slider3 = plotter.add_slider_widget(
        update_y3,
        rng=[0, condensed_stack.shape[1]],
        value=plane_cords['y'],
        title='Y',
        pointa=(.77, .125),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.77, .87),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )

    plt.close()

    # Print plotting time
    execution_timer(start_time, 'Plot 4 Execution time: ')

# Define a callback function to re-render
def re_render(state):

    global plane_cords, slice_x, slice_y, slice_z

    # Record current plane coordinates; each sublist are the coordinates of x, y, z in each of the planes displayed
    # When using add_volume_clip 2 planes are added instead of 1 with other methods
    curr_cords = [list(plotter.plane_widgets[0].GetOrigin()), list(plotter.plane_widgets[1].GetOrigin()), #Volume Planes
                  [plane_cords['x'], plotter.slider_widgets[0].GetRepresentation().GetValue(), plotter.slider_widgets[1].GetRepresentation().GetValue()], # Y Slider, Z Slider
                  [plotter.slider_widgets[2].GetRepresentation().GetValue(), plane_cords['y'] , plotter.slider_widgets[3].GetRepresentation().GetValue()], # X Slider, Z Slider
                  [plotter.slider_widgets[4].GetRepresentation().GetValue(), plotter.slider_widgets[5].GetRepresentation().GetValue(), plane_cords['z']]  # X Slider, Y Slider
                  ]

    # Checks for changes in any of the x, y, z coordinates between planes displayed to re-adjust them when re-rendering
    plane_cords = cord_compare(curr_cords, plane_cords)


    # Create new origin point for each slice
    orig = [max(i, 0) for i in plane_cords.values()]

    plotter.clear_slider_widgets()

    # Re-render plane widgets and graphs
    plotter.subplot(0,0)
    start_time = time.perf_counter()

    #Update any values to volume when slices are saved
    #plotter.clear()
    #foram = plotter.add_volume_clip_plane(grid, normal='x', cmap=custom_cmap, opacity='sigmoid', invert=False,
    #                                      assign_to_axis='x', implicit=True, show_scalar_bar=False, outline_opacity=12)

    # Update first plane in 3d voxel render
    plotter.plane_widgets[0].SetOrigin(orig)
    plotter.plane_widgets[0].UpdatePlacement()

    # Update 2nd plane in 3d voxel render (unseen in render)
    plotter.plane_widgets[1].SetOrigin(orig)
    plotter.plane_widgets[1].UpdatePlacement()

    execution_timer(start_time, 'Plot 1 Execution time: ')


    # Create subplots
    plotter.subplot(0, 1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    # Remove chart that currently exists on subplot
    plotter.renderer.remove_chart(0)

    # Create new MLP chart for a given 2D axis slice; Process same as pyrender(.) function
    slice_x = condensed_stack[int(plane_cords['x']), :, :]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x_transform = np.fliplr(np.rot90(condensed_stack_copy[int(plane_cords['x']), :, :], k=-1))
    ax1.imshow(x_transform, cmap=custom_cmap)
    fig1.gca().invert_yaxis()
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.set_title(label=f"X Axis Slice (Y >, Z ^)", fontsize=10)

    vline = ax1.axvline(plane_cords['y'], color='r', linestyle='-')
    hline = ax1.axhline(plane_cords['z'], color='r', linestyle='-')

    chart1 = pv.ChartMPL(fig1)
    plotter.add_chart(chart1)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_y(val):
        vline.set_xdata([val])
        fig1.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_z(val):
        hline.set_ydata([val])
        fig1.canvas.draw_idle()

    y_slider = plotter.add_slider_widget(
        update_y,
        rng=[0, condensed_stack.shape[1]],
        value=plane_cords['y'],
        title='Y',
        pointa=(0.39, .05),  # (x-axis, y-axis)
        pointb=(0.635, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    z_slider = plotter.add_slider_widget(
        update_z,
        rng=[0, condensed_stack.shape[2]],
        value=plane_cords['z'],
        title='Z',
        pointa=(.71, .125),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.71, .87),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )

    plt.close()

    # Print plotting time
    execution_timer(start_time, 'Plot 2 Execution time: ')

    # Process similar above
    plotter.subplot(1, 1)  # 35 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    plotter.renderer.remove_chart(0)

    slice_y = condensed_stack[:, int(plane_cords['y']), :]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    y_transform = np.fliplr(np.rot90(condensed_stack_copy[:, int(plane_cords['y']), :], k=-1))
    ax2.imshow(y_transform, cmap=custom_cmap)
    fig2.gca().invert_yaxis()
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.set_title(label=f"Y Axis Slice (X >, Z ^)", fontsize=10)

    vline2 = ax2.axvline(plane_cords['x'], color='r', linestyle='-')
    hline2 = ax2.axhline(plane_cords['z'], color='r', linestyle='-')

    chart2 = pv.ChartMPL(fig2)
    plotter.add_chart(chart2)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_x2(val):
        vline2.set_xdata([val])
        fig2.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_z2(val):
        hline2.set_ydata([val])
        fig2.canvas.draw_idle()

    x_slider2 = plotter.add_slider_widget(
        update_x2,
        rng=[0, condensed_stack.shape[0]],
        value=plane_cords['x'],
        title='X',
        pointa=(0.365, .05),  # (x-axis, y-axis)
        pointb=(0.66, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    z_slider2 = plotter.add_slider_widget(
        update_z2,
        rng=[0, condensed_stack.shape[2]],
        value=plane_cords['z'],
        title='Z',
        pointa=(.71, .125),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.71, .87),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )


    plt.close()

    #plotting time
    execution_timer(start_time, 'Plot 3 Execution time: ')

    # Process similar above
    plotter.subplot(2, 1)  # 33 Seconds - Array Size: 474,449, 362

    start_time = time.perf_counter()

    plotter.renderer.remove_chart(0)

    slice_z = condensed_stack[:, :, int(plane_cords['z'])]
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    z_transform = z_transform = np.flipud(np.rot90(condensed_stack_copy[:, :, int(plane_cords['z'])]))
    ax3.imshow(z_transform, cmap=custom_cmap)
    fig3.gca().invert_yaxis()
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax3.set_title(label=f"Z Axis Slice (X >, Y ^)", fontsize=10)

    vline3 = ax3.axvline(plane_cords['x'], color='r', linestyle='-')
    hline3 = ax3.axhline(plane_cords['y'], color='r', linestyle='-')

    chart3 = pv.ChartMPL(fig3)
    plotter.add_chart(chart3)
    #plotter.show(auto_close=False, interactive=True, interactive_update=True)

    # Method and slider to update all visuals based on the slide selection
    def update_x3(val):
        vline3.set_xdata([val])
        fig3.canvas.draw_idle()

    # Method and slider to update all visuals based on the slide selection
    def update_y3(val):
        hline3.set_ydata([val])
        fig3.canvas.draw_idle()

    x_slider3 = plotter.add_slider_widget(
        update_x3,
        rng=[0, condensed_stack.shape[0]],
        value=plane_cords['x'],
        title='X',
        pointa=(0.3375, .05),  # (x-axis, y-axis)
        pointb=(0.688, .05),
        slider_width=.02,
        tube_width=.02,
        style='modern',
        interaction_event='always',
    )

    y_slider3 = plotter.add_slider_widget(
        update_y3,
        rng=[0, condensed_stack.shape[1]],
        value=plane_cords['y'],
        title='Y',
        pointa=(.77, .125),  # (x-axis, y-axis) - x = .71, y = .125
        pointb=(.77, .87),  # x = .71, y = .87
        slider_width=.01,
        tube_width=.01,
        style='modern',
        interaction_event='always',
    )

    plt.close()

    # Print plotting time
    execution_timer(start_time, 'Plot 4 Execution time: ')


    #Update and Rerender Plot
    plotter.update()
    plotter.render()


#%%
# V3 (Working)
directory, tiff_stack, N, tiff_res, condensed_stack, metadata, scale_xyz, RW_pixel_size = foram_import()

del tiff_stack # This is to reduce memory requirements

#%%

# Create a custom_cmap
importlib.reload(Custom_CMAP) #reload any changes

# Set the threshold for voxels that get color
threshold = np.min(condensed_stack[condensed_stack != 0])
custom_cmap = Custom_CMAP.create_cmap(threshold = threshold)

'''
# Plotting with the custom colormap
data = np.arange(0, 256).reshape(16,16)
plt.imshow(data, cmap=custom_cmap)
plt.colorbar(label='Data Value')
plt.title('Plot with Custom Colormap')
plt.show()
'''

#Create copy of the condensed stack to perform augmentations to the underlying data during segmentation process
condensed_stack_copy = copy.deepcopy(condensed_stack)


# Delete Plotter object if exists
if 'plotter' in globals():
    plotter.close()


# Set plotter shape, col and row weighs control size of window and plotter output size
plotter = pv.Plotter(shape=(3,2), col_weights=[1, 1], row_weights=[1,1,1], groups = [([0,2], 0), (0, 1), (1, 1), (2, 1)])

#Create Pyvista plotter object and plot to annotate object
grid = pv.ImageData()
grid.dimensions = np.array(condensed_stack_copy.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = condensed_stack_copy.ravel(order="F")  # Use Fortran ordering for flattening
plane_cords = {i : None for i in ["x", "y", "z"]}  # Used to track past plane coordinates
slices_dict = {}  # Used to track 2D slices of annotations
masked_slices = {} # Used to track 3D annotations
selected_annotation = None # Used to track the label of the active radio button which is associated with foram dictionary
locked_state = False # Used to ensure multiple buttons cannot be used at the same time so no conflicts with diag. boxes happen


# Generate Plotter
pyrender(grid)

#plotter.add_key_event("m", toggle_on_distance_tool)
#plotter.add_key_event("f", toggle_off_distance_tool)

# Display plot
plotter.show()


