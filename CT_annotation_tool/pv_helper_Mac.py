
# Compares coordinates between sets to see if any have changed between iterations
import tkinter as tk
from tkinter import messagebox, font, simpledialog
import pickle
import time
from CT_PySegmentation import *

'''
function: 
    Takes in 2 dictionaries containing the 3D coordinates of the center of 2D slicing planes of the pyvista GUI. 
    The past_cords dictionary records where they planes were located prior to being moved by the user and the 
    rerender button being clicked. The current_cords dictionary records the current coordinates of the 2D plane slicers
    and the function compares the coordinates to see which coordinates have changed and updates accordingly

input: 
    current_cords (dict) - dictionary containing the 3D center coordinates of the current position of 2D slicer widgets in Pyvista GUI
    past_cords (dict) -  dictionary containing the 3D center coordinates of the past position of 2D slicer widgets in Pyvista GUI

output:
    past_cords (dict) - dictionary containing the updated 3D center coordinates of the of 2D slicer widgets in Pyvista GUI
'''


def cord_compare(current_cords, past_cords):

    #Finds different x,y,z coordinates from all possible plane widget
    x_cords = [current_cords[0] for current_cords in current_cords if current_cords[0] != past_cords['x']] # Checks if any x coordinates changed in planes
    y_cords = [current_cords[1] for current_cords in current_cords if current_cords[1] != past_cords['y']] # Checks if any y coordinates changed in planes
    z_cords = [current_cords[2] for current_cords in current_cords if current_cords[2] != past_cords['z']] # Checks if any z coordinates changed in planes

    if len(x_cords) != 0:
        past_cords['x'] = x_cords[0] # extract first different value of x plane widget coordinate
    if len(y_cords) != 0:
        past_cords['y'] = y_cords[0] # extract first different value of y plane widget coordinate
    if len(z_cords) != 0:
        past_cords['z'] = z_cords[0] # extract first different value of z plane widget coordinate

    return past_cords

'''
function: 
    Prompts user to input a name for their chamber annotation 2D slices and checks if the name has already been used in 
    the dictionary recording the 2D slice annotations

input: 
    None

output:
    None
'''

def chamber_diag_box_pc():
    """Asks user for a valid name using a Tkinter input dialog.
       macOS-safe version that avoids dialog freezing/spinning.
    """

    root = tk.Tk()
    root.withdraw()  # Hide main window
    root.geometry("1500x1000")

    # Custom font
    custom_font = font.Font(family="Arial", size=18, weight="bold")
    root.option_add("*Dialog.msg.font", custom_font)
    root.option_add("*Dialog.entry.font", custom_font)

    result = {"value": None}

    def ask():
        user_input = simpledialog.askstring(
            "Input",
            "Please enter the name you want for your slice set:",
            parent=root
        )

        if not user_input:
            result["value"] = None
            root.destroy()
            return

        user_input = user_input.strip()

        # Validate Python-like variable name
        if user_input[0].isalpha() and user_input.isalnum():
            result["value"] = user_input
        else:
            result["value"] = None
            show_info_message_pc(
                "Please provide a valid variable name (letters and numbers only; "
                "must begin with a letter)."
            )

        root.destroy()

    root.after(0, ask)  # Schedule dialog after Tk initializes
    root.mainloop()  # Allow macOS to process events

    return result["value"]


def chamber_diag_box_mac():
    """Asks user for a valid name using a Tkinter input dialog.
       macOS-safe version that avoids dialog freezing/spinning.
    """

    root = tk.Tk()
    root.withdraw()  # Hide main window
    root.geometry("1500x1000")

    # Custom font
    custom_font = font.Font(family="Arial", size=18, weight="bold")
    root.option_add("*Dialog.msg.font", custom_font)
    root.option_add("*Dialog.entry.font", custom_font)

    result = {"value": None}

    def ask():
        user_input = simpledialog.askstring(
            "Input",
            "Please enter the name you want for your slice set:",
            parent=root
        )

        if not user_input:
            result["value"] = None
            root.destroy()
            return

        user_input = user_input.strip()

        # Validate Python-like variable name
        if user_input[0].isalpha() and user_input.isalnum():
            result["value"] = user_input
        else:
            result["value"] = None
            show_info_message_mac(
                "Please provide a valid variable name (letters and numbers only; "
                "must begin with a letter)."
            )

        root.destroy()

    root.after(0, ask)  # Schedule dialog after Tk initializes
    root.mainloop()  # Allow macOS to process events

    return result["value"]

'''
function: 
    Prompts user to input a file/folder path to be used as an input for another function

input: 
    Message (str) - Message user wants to display in the pop-up

output:
    None
'''

def filepath_diag_box_mac(message: str):
    """Shows a Tkinter input dialog using a temporary hidden root.
       macOS-safe version that avoids hangs or spinning.
    """

    root = tk.Tk()
    root.withdraw()                # hide window
    root.geometry("1500x1000")

    # Custom font for dialogs
    custom_font = font.Font(family="Arial", size=24, weight="bold")
    root.option_add("*Dialog.msg.font", custom_font)
    root.option_add("*Dialog.entry.font", custom_font)

    result = {"value": None}

    def ask():
        user_input = simpledialog.askstring("Input", message, parent=root)
        if user_input:
            result["value"] = user_input.strip()
        root.destroy()             # close hidden root when done

    # Schedule dialog after Tk initializes
    root.after(0, ask)
    root.mainloop()                # allow dialog + events to run

    return result["value"]


'''
function: 
    Displays a custom message in a pop-up text box that is displayed when the function is called

input: 
    Message (str) - Message user wants to display in the pop-up

output:
    None
'''

def show_info_message_mac(message):
    """Displays an informational message box."""
    root = tk.Tk()
    root.withdraw()            # hide the main window
    root.after(0, lambda: messagebox.showinfo(title="", message=message))
    root.after(10, root.destroy)
    root.mainloop()

'''
function: 
    Takes in a dictionary with 3D arrays (tensors) and vertices of the annotated 2D array slices. It then converts the 
    2D slices into a 3D mask using the mask_ct_vol function and saves the 3D mask. Additionally, it records the vertices
    associated with the 2D slices and pairs them with the 3D mask associated with the 3D slices. The resulting dictionary
    is saved to a specified directory and filename

input: 
    dict (dict) - A dictionary of annotated 2D slices/polygons of the 3D rendered object and the vertices of where the user 
                  isolated a 2D polygon (from the 2D slices) within the 3D rendered object
    ct_array (array) - An array/tensor that stores the voxel values of the CT object
    directory (str) - A file path where the pickled version of the dictionary created by the function should be stored
    filename (str) - The filename of the dictionary that is created by the function
    threshold (int) - An integer that defines the voxel values threshold at which the voxel values should be zeroed out

output:
    None
'''

def save_3d_annotations(im_dict : dict, ct_array, directory, filename, threshold : int):

    annotated_chambers_dict = {}
    for key in im_dict:
        annotated_chambers_dict[key] = {'3D Tensor': None, 'Vertices': None}
        annotated_chambers_dict[key]['3D Tensor'] = mask_ct_vol(masks_dict=im_dict, dataset_name=key,
                                                                ct_stack=ct_array, invert_mask=False, threshold=threshold)
        annotated_chambers_dict[key]['Vertices'] = im_dict[key]['Vertices']

    # Save Dictionary to folder
    pickle_doc = filename + '.pkl'
    pickle_path = os.path.join(directory, pickle_doc)
    with open(pickle_path, 'wb') as f:
        pickle.dump(annotated_chambers_dict, f)


'''
function: 
    Reads a pickle file into python

input: 
    fielpath - filepath for picle file

output:
    loaded_object - python object stored in pickle file
'''

def open_3d_annotations(filepath):
    try:
        with open(filepath, 'rb') as f:
            loaded_object = pickle.load(f)
        show_info_message_mac("Object loaded successfully:")

        return loaded_object

    except FileNotFoundError:
        show_info_message_mac(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        show_info_message_mac(f"An error occurred while loading the pickle file: {e}")


'''
function: 
    Records and displays a function's execution time. Requires start time to be recorded prior to function execution

input: 
    start_time - time prior to function execution
    message - customer user defined message to be displayed

output:
    None
'''

def execution_timer(start_time, message):
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    #print(f"{message} {exec_time:.4f} seconds")


'''
# Function to toggle visibility (Not needed for current code but useful)
def toggle_plot_visibility(plotter_instance, visibility_setting):
    for name, actor in plotter_instance.actors.items():
        actor.SetVisibility(visibility_setting)

def measurement(a,b,dist):
    global distance
    distance = dist
    plotter.subplot(0, 0)
    plotter.add_text(f"distance is: {dist}",name="distance_text",  font_size=12, position="upper_right")

def toggle_on_distance_tool():
    show_info_message("Measurement tool is active. Click to start and stop line")
    plotter.add_measurement_widget(callback = measurement)

def toggle_off_distance_tool():
    plotter.clear_measure_widgets()
    plotter.remove_actor("distance_text")
    plotter.subplot(0, 0)
    plotter.renderers[0].ResetCameraClippingRange()
    plotter.renderers[1].ResetCameraClippingRange()
    plotter.renderers[2].ResetCameraClippingRange()
    plotter.renderers[3].ResetCameraClippingRange()
'''


