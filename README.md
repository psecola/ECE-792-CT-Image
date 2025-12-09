**Welcome to the CT Image analyzer tool**

**About the tool**

This tool allows the user to visualize and annotate 3D representations of objects that have been scanned into one or more .tiff files. The tool allows for the annotation
of segments of the CT object. Additionally, multivariate Gaussian can be fitted to each sement in an attempt to estimate the dimensions and volume of each annotation.
The voxelized segments and their dimensional information can be downloaded as a .pkl and .csv files, respectfully.

Unfortunately at this time (12/8) this system has only been tested on Windows systems. Future instructions will be written for Mac OS and Linux OS. 

**Programs that will need to be installed and configured:** <br/>
1.) Ubuntu (for windows) <br/>
2.) Docker Desktop for Windows <br/>
  - Enable WSL (Windows Subsystem for Linux) Integration for Ubuntu in Docker Desktop <br/>

**Quick Start Guide** <br/>
1.) Open Docker Desktop <br/>
2.) Open Microsoft Powershell. To confirm you are using WSL 2 run the following code <br/>
```
wsl -l -v
```
<br/>

You should see that Ubuntu is running version 2. If not run the following code: <br/>
```
wsl --set-version Ubuntu 2
```
<br/>
2.) Open Ubuntu terminal <br/>
3.) Install the proper GUI requirements: <br/>

```
sudo apt update
sudo apt install x11-apps
```
<br/>
4.) Change directory and load the docker image into Ubuntu (change [username] and [folder]): <br/>

```
cp /mnt/c/Users/[username]/Documents/[folder]/ct_pyvista.tar ~/
cd ~
docker load -i ct_pyvista.tar
```

You should see the message: **Loaded image: ct_pyvista:latest**

5.) To run docker image use the following code: <br/>
```
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e PULSE_SERVER=$PULSE_SERVER \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ct_pyvista
```
6.) Insert the folder path where your CT images into the first initial popup -> Answer Y/N if you have metadata file about the CT information (necessary for gathering real-world measurements of annotations) -> If you do have this file then provide the path to the document and ensure if follows the format given below <br/>
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/eff79e7f-1ed4-4270-a1e9-32232fcb0c3b" /> <br/>
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/78442d05-9f72-4879-814e-53b2b1b60aa4" /> <br/>
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/b99ab198-c38f-49f0-a022-4cde8c20adb3" /> <br/>

7.) Explore and Annotate!<br/>


Metadata file format (must be .txt file): <br/>
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/1f6e6820-ef3a-4fba-b73a-35b89d576f55" />






