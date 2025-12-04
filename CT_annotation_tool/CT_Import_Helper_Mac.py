#%%

import os
import glob
import numpy as np
import vtk
import tifffile as tf
import torch
import pyvista as pv
import vtk
import time
from matplotlib import pyplot as plt
from pv_helper_Mac import *

'''
function: 
    Extracts all information from a directors .tiff files, places them into
    a stacked numpy array, and produces metadata about individual .tiff files
    and number of .tiff files. The .tiff files must be in order

input: 
    directory_path (str) - the filepath to the directory folder housing a set of .tiff files

output:
    tiff_stack (array) - stacks the numerical representation of a .tiff file scan
                         on to one another to create a 3D numpy array
    N (int) - number of .tiff files
    tiff_res (tuple) - resolution of each .tiff file (assuming uniform for all files)
'''


def get_tiff_metadata(directory_path):
    # Specify file type
    file_pattern = '*.tiff'

    # Extract metadata
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    N = len(file_paths)

    # Extract resolution of first .tiff tile
    reader = vtk.vtkTIFFReader()
    reader.SetFileName(file_paths[0])
    reader.Update()

    image = reader.GetOutput()
    tiff_res = list(image.GetExtent())
    tiff_res[-1] = N - 1

    tiff_stack = np.array([tf.imread(file) for file in file_paths], dtype=np.int64)

    return tiff_stack, N, tuple(tiff_res)


'''
function: 
    Rotates the 3D tiff stack array and removes submatrices whose elements do not
    meet a user-specified threshold. This is to shrink the array and remove groups
    of elements that may interfere with the rending or can be discarded before rendering.
    This function operates on the assumption that higher valued cells are more important
    than lower value cells in regards to rendering an image

input: 
    tensor (array) - 3D numpy array
    threshold (int) - the maximum value an element needs to reach for a submatrix to not be removed

output:
    condensed_stack (array) - reduced .tiff 3D numpy stack
'''


# Preprocess data to mask low values cells based on array value dist.
def shave_tensor(tensor, threshold_upper=None, threshold_lower=None):
    rows, cols, depth = tensor.shape
    rows_to_drop = []
    cols_to_drop = []
    depth_to_drop = []
    condensed_stack = tensor
    for row in range(0, rows):
        if np.any((condensed_stack[row, :, :] > threshold_upper)):
            continue
        else:
            rows_to_drop.append(row)

    condensed_stack = np.delete(condensed_stack, rows_to_drop, axis=0)

    rows, cols, depth = condensed_stack.shape
    for col in range(0, cols):
        if np.any((condensed_stack[:, col, :] > threshold_upper)):
            continue
        else:
            cols_to_drop.append(col)

    condensed_stack = np.delete(condensed_stack, cols_to_drop, axis=1)

    rows, cols, depth = condensed_stack.shape
    for dep in range(0, depth):
        if np.any((condensed_stack[:, :, dep] > threshold_upper)):
            continue
        else:
            depth_to_drop.append(dep)

    condensed_stack = np.delete(condensed_stack, depth_to_drop, axis=2)

    condensed_stack[condensed_stack <= threshold_lower] = 0


    return condensed_stack


'''
function: 
    Passes values identified by a n x m x h area (kernel) where n is a user specified parameter.
    This function averages the values within the user-defined n x m x h area and produces
    a single values for those selected values, condensing the 3D array. Note that there
    is no zero-padding used. 

    For more information about the function see:
        https://dev.to/hyperkai/avgpool2d-in-pytorch-2i25

input: 
    tensor (array) - 3D numpy array
    k_size (int/tup; Required) - Int/Tuple that defines the number of array elements selected by the n x m x h kernel (default: None)
    stride (int/tup) - Int/Tuple that determines the shift of the n x m x h kernel to compute the next pooling value (default: None)

output:
    numpy array - a pooled 3D numpy array
'''


def average_pool(tensor, k_size, stride):
    # Turns numpy array into tensor
    tensor = torch.from_numpy(tensor)

    # Add a batch dimension if the tensor doesn't have one
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    # Define pooling function using inputes
    pool_func = torch.nn.AvgPool3d(kernel_size=k_size, stride=stride)
    # pool_func = torch.nn.AvgPool2d(kernel_size=2)

    # Remove batching dimension
    output = pool_func(tensor).squeeze(0)


    return output.numpy()


'''
function: 
    Aggregates the get_tiff_metadata(), shave_tensor(), and average_pool() functions.
    Renders a volumetric or adjustable sliced 2d image of the 3D .tiff stack, depending
    on the parameters selected. Execution time can vary depending on the size of the 
    3d array and dimension reduction parameters chosen

input: 
    directory_path (str) - The filepath to the directory folder housing a set of .tiff files
    condense(Bool) - If true Implements condensing algorithm (via shave tensor) reduce .tiff stack 3D array size
    upper_threshold (int) - Determines maximum value of array elements of submatrices identified by the shave_tensor() function
    lower_threshold (int) - Determines the maximum value of array elements that will be zereod out before rendering
    im_pooling (Bool) - If true implements image pooling using 3D averaging
    k_size (Int/Tuple) - Defines the number of array elements selected by the n x m x h kernel (default: 2)
    stride (Int/Tuple) - Determines the shift of the n x m x h kernel to compute the next pooling value (default: 1)
    type - Determines type of rendering of CT scans:
                volume - renders surface volumetric 3D image of the scanned object that can be rotates and magnified
                mesh slicer - renders a 2D image of the scanned object that can be adjusted in real time (no reloading)
                mesh plane - renders a volumetric image whose depth can be adjusted in real time allowing the user to
                             see the 2D interior of the scanned object at certain depths
                volume slicer - renders a volumetric image of the object where the user can adjust the height and depth
                                of the slicer to see the inside of the 3D volumetric rendering
    verbose (Bool) - Returns a histogram of array values of the 3D numpy .tiff stack. Assists in identifying proper thresholds

output:
    Renders mesh sliced or volumetric image of the .tiff stack.

    tiff_stack (array) - Original 3D numpy array representation of the .tiff file stack
    N (int) - Number of .tiff files
    tiff_res (tuple) - Resolution of a single .tiff image (given as dimensions of array)
    condensed_stack (array) - Reduced representation of .tiff file stack (if proper arguments supplied to function)
'''


def import_tiff(directory_path=None, condense=False, im_pooling=False, kernel_size=2, stride=1, threshold = None,
                 plot_image=True, image_type='volume', verbose=False):
    start_time = time.perf_counter()
    # function being timed
    tiff_stack, N, tiff_res = get_tiff_metadata(directory_path)
    # Set tiff_stack to be condensed_stack, so code runs
    condensed_stack = tiff_stack
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    #print(f"Function execution time: {execution_time:.4f} seconds")

    if verbose:
        # Get distribution of array values
        plt.hist(tiff_stack.flatten(), bins=90, color='skyblue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Value')
        plt.xticks(ticks=np.arange(0, np.max(tiff_stack.flatten()), 5), rotation=-45)
        plt.ylabel('Frequency')
        plt.title('Histogram of Random Data')

        # Show the plot
        plt.show()

    if condense:

        if threshold is None:
            # Preprocess tiff_stack to condense tensor and eliminate any unnecessary pixels (turns them to 0)
            start_time = time.perf_counter()
            # function being timed
            vals = tiff_stack.flatten()
            mask = vals != 0  # Creates a boolean array where True indicates elements not equal to 0
            vals = vals[mask]
            lower_threshold = np.percentile(vals, 97)
            upper_threshold = 1.07 * lower_threshold  # 1.07 is just from iterations
            condensed_stack = shave_tensor(condensed_stack, threshold_upper=upper_threshold,
                                           threshold_lower=lower_threshold)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            #print(f"Function execution time: {execution_time:.4f} seconds")

        else:
            # Preprocess tiff_stack to condense tensor and eliminate any unnecessary pixels (turns them to 0)
            start_time = time.perf_counter()
            # function being timed
            vals = tiff_stack.flatten()
            mask = vals != 0  # Creates a boolean array where True indicates elements not equal to 0
            vals = vals[mask]
            lower_threshold = threshold
            upper_threshold = threshold
            condensed_stack = shave_tensor(condensed_stack, threshold_upper=upper_threshold,
                                           threshold_lower=lower_threshold)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            #print(f"Function execution time: {execution_time:.4f} seconds")

    if im_pooling:
        start_time = time.perf_counter()

        # average pool tensor and 0 out non-important voxel values
        try:
            condensed_stack = average_pool(condensed_stack, kernel_size, stride)
            condensed_stack = np.where(condensed_stack < lower_threshold, 0, condensed_stack)
        except NameError:
            vals = tiff_stack.flatten()
            mask = vals != 0  # Creates a boolean array where True indicates elements not equal to 0
            vals = vals[mask]
            lower_threshold = np.percentile(vals, 5)
            condensed_stack = average_pool(condensed_stack, kernel_size, stride)
            condensed_stack = np.where(condensed_stack < lower_threshold, 0, condensed_stack)


        end_time = time.perf_counter()
        execution_time = end_time - start_time
        #print(f"Function execution time: {execution_time:.4f} seconds")


    return tiff_stack, N, tiff_res, condensed_stack




'''
function: 
    Takes in .txt header file with metadata about the CT imaging parameters and coverts them into Python dictionary

input: 
    header_path (str) - file path from local directory of the .txt file containing CT scan metadata

output:
    metadata - python dictionary of dictionaries containing sectioned CT scan metadata
'''

def read_ct_header(header_path):
    """
    Reads a CT scan metadata header (.txt) and returns a structured dictionary.
    Example section headers: [Det Assembly Info], [Image Info]
    """
    metadata = {}
    current_section = None

    with open(header_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue
            # Detect section headers like [Image Info]
            if line.startswith('[') and line.endswith(']'):
                current_section = line.strip('[]')
                metadata[current_section] = {}
            elif '=' in line and current_section:
                key, value = map(str.strip, line.split('=', 1))
                # Try to convert value to a number (int or float)
                try:
                    if '.' in value or 'e' in value.lower():
                        num_value = float(value)
                        # If it’s an integer-like float, cast down to int
                        if num_value.is_integer():
                            num_value = int(num_value)
                        metadata[current_section][key] = num_value
                    else:
                        metadata[current_section][key] = int(value)
                except ValueError:
                    metadata[current_section][key] = value
    return metadata

'''
function: 
    Prompts the user to supply several file paths to run functions created to import CT images and metadata. 

input: 
    None - All inputs are file paths inputted into prompt boxes

output:
    directory (str) - file path of CT images 
    tiff_stack (array) - 3D numpy representation of the stacked CT images
    N (int) - Number of CT images in directory
    tiff_res (tuple) - Voxel resolution of (W,.,L,.,D) 
    condensed_stack - 3D numpy representation of stacked CT images with 2D slices containing voxels under a given threshold. Voxels then pooled.
    metadata (dict) - dictionary containing metadata of CT images. See github for format of file
    scale (tuple) - the scaling of the condensed 3D array compared to the original tiff stack 3D array
    RW_pixel_size (tuple) - the real world measurement of the initial pixel (voxel, really) of each CT image (before condensing)
'''


def foram_import():

    #Define variables to be returned
    directory = tiff_stack = N = tiff_res = condensed_stack = metadata = None


    # Specify the directory, get .tiff metadata, and create tiff stack numpy object
    directory = filepath_diag_box_mac("Please enter the CT scans directory:")


    if directory != '' and directory is not None:
        try:
            pickle_path = directory[:directory.rfind('/')] + directory[
            directory.rfind('/'):] + '_voxel_dict'  # No need to change this

            with open(pickle_path, 'rb') as f:
                loaded_object = pickle.load(f)
            show_info_message_mac("Object loaded successfully")

            tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path=directory, condense=False, im_pooling=False,
                                                                   kernel_size=3, stride=4, verbose=False)
            condensed_stack = loaded_object["condensed stack"]

            show_info_message_mac("CT scan import complete")

        except FileNotFoundError:
            show_info_message_mac("No PKL file found. Resorting to importing scans via .tiff files and import function")

            try:
                tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path=directory, condense=True, im_pooling=True,
                                                                       kernel_size=3, stride=4, verbose=False)

                voxel_dict = {"condensed stack": condensed_stack} #Only storing condensed stack for size purposes
                with open(pickle_path, 'wb') as f:
                    pickle.dump(voxel_dict, f)

                show_info_message_mac("CT scan import complete")

            except Exception:
                show_info_message_mac("An error occurred. Please ensure correct CT directory was entered")


        except Exception as e:
            show_info_message_mac(f"An error occurred while loading the PKL file: {e}. Resorting to importing scans via .tiff files and import function")

            try:
                tiff_stack, N, tiff_res, condensed_stack = import_tiff(directory_path=directory, condense=True, im_pooling=True,
                                                                       kernel_size=3, stride=4, verbose=False)

                voxel_dict = {"condensed stack": condensed_stack}  # Only storing condensed stack for size purposes
                with open(pickle_path, 'wb') as f:
                    pickle.dump(voxel_dict, f)

                show_info_message_mac("CT scan import complete")

            except Exception:
                show_info_message_mac("An error occurred. Please ensure correct CT directory was entered")

    else:
        show_info_message_mac("No directory provided")


    # Prompt user for CT scan metadata to compute chamber lengths
    user_answer = filepath_diag_box_mac("Do you have a file with image metadata to assist with computing measurements (Y/N):")

    if user_answer == "Y":
        fp = filepath_diag_box_mac("Please enter the CT metadata file path (see Github for file formatting):")

        if fp != '' and fp is not None:
            try:
                metadata = read_ct_header(fp)
                show_info_message_mac("Metadata import complete; import process complete")
                fp = None

            except FileNotFoundError:
                show_info_message_mac("No file found. Please try again or click OK to continue without inputting file")
                fp = filepath_diag_box_mac("Please enter the CT metadata file path (see Github for file formatting):")
                metadata = None

            except Exception as e:
                show_info_message_mac(f"An error occurred while loading the header file: {e}. Please try again or click OK to continue without loading a file")
                fp = filepath_diag_box_mac("Please enter the CT metadata file path (see Github for file formatting):")
                metadata = None
    else:
        metadata = None
        show_info_message_mac("Import Process complete")

    # Check is metadata exists and pull pertanent information to get real world measurements
    if metadata is not None:
        # compute real world size of each pixel based on metadata provided
        RW_pixel_size = metadata['Image Info']['Pixel Size'] / metadata['Image Info']['Optical Magnification']

        # Compute scaling for x, y, z axis due to condensing of CT imaging
        scale = (tiff_res[1] / condensed_stack.shape[0], tiff_res[3] / condensed_stack.shape[1],
                     tiff_res[5] / condensed_stack.shape[2])

    else:
        # Set RW_pixel_size to be 1 so there is no change in output of dependent functions
        RW_pixel_size = 1

        # Set scale_xyz to be (1,1,1) so there is no change in output of dependent functions
        scale = (1, 1, 1)

    return directory, tiff_stack, N, tiff_res, condensed_stack, metadata, scale, RW_pixel_size
