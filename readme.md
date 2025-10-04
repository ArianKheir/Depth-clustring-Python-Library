# **Depth Clustering Python Library**

This Python library is built on top of the [depth_clustering C++ library](https://github.com/PRBonn/depth_clustering) and provides bindings to the underlying C++ code for easy usage in Python. The library performs clustering on 3D point cloud data, typically generated from LiDAR scans. This project allows for seamless integration with Python for processing point clouds and generating clustered point cloud data along with visual outputs.

## **Features**
- Python bindings to the depth_clustering C++ library using pybind11.
- Functions for converting and processing 3D point clouds in the form of NumPy arrays.
- Full pipeline implementation for reading, processing, and clustering point clouds from LiDAR data (e.g., 360° LiDAR images).
- Customizable parameters for clustering and ground removal.
- Outputs include clustered label images and depth images.
  
## **Table of Contents**
1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [Usage](#usage)
   - [Point Cloud Clustering Example](#point-cloud-clustering-example)
   - [Running the Full Pipeline](#running-the-full-pipeline)
4. [Functions Available in `pointcloud_func.py`](#functions-available-in-pointcloud_func.py)
5. [Pipeline Execution (`run_pipeline.py`)](#pipeline-execution-run_pipeline.py)
6. [Example Output](#example-output)
7. [Related Publications](#related-publications)
8. [License](#license)

## **Installation**
To use this library, you need to have the `depth_clustering` C++ library correctly set up, along with the pybind11 bindings.

### Steps to Install:
1. **Clone the depth_clustering Repository:**
```bash
   https://github.com/ArianKheir/Depth-clustring-Python-Library
   cd Depth-clustring-Python-Library
```
2. **Set up the Python Bindings:**
    - Make sure you have pybind11 installed. You can install it using pip:
```bash
    pip install pybind11
```
3. **Install the Python Package:**

 - You can install the Python package using:
 ```bash
    rm -rf build dist *.egg-info
    pip install --force-reinstall --no-build-isolation .
 ```

## Dependencies

 -   Python 3.x
 -   NumPy
 -   pybind11
 -   Open3D (for visualization)
 -   OpenCV (for image handling)
 
 **To install dependencies, run:**
 ```bash
    pip install numpy pybind11 open3d opencv-python
    pip install -r requirements.txt
```
## **Usage**

### **Point Cloud Clustering Example**
The main function of the `pointcloud_func.py` script is to take a 3D point cloud (in the form of an `n x 3` NumPy array), perform ground removal, and generate clusters from the data. Here's how you can use it:
```python
import numpy as np
import depth_clustering as dc
from pointcloud_func import pointcloud_to_cluster_image, load_calibration_data, extract_pointcloud_from_png

# Example: Processing a LiDAR scan (point cloud)
cfg_path = 'path_to_config_file.cfg'
png_path = 'path_to_png_file.png'

# Load calibration data (pitch angles)
pitch_angles = load_calibration_data(cfg_path)

# Extract point cloud from PNG image
points_nx3 = extract_pointcloud_from_png(png_path, pitch_angles)

# Run the clustering pipeline
label_image, depth_image, clusters = pointcloud_to_cluster_image(
points_nx3=points_nx3,
pitch_angles_rad=pitch_angles,
cfg_path=cfg_path
)

# Output: Label image with clustered labels and depth image
print("Label Image Shape:", label_image.shape)
print("Number of Clusters:", len(clusters))
```
### **Running the Full Pipeline**
If you have a series of LiDAR scan images (e.g., 360° LiDAR scans) in PNG format, you can process them in bulk using the `run_pipeline.py` script:
```bash
python run_pipeline.py --data_path /path/to/lidar/scans --cfg_path /path/to/config_file.cfg
```
This script will automatically load the scan files, process them through the full pipeline (including ground removal and clustering), and display the results.

---

## **Functions Available in `pointcloud_func.py`**

### **1. `load_calibration_data(cfg_path)`**
This function loads the calibration data from a `.cfg` file. The file contains sensor parameters, including pitch angles (in degrees), which are converted to radians.

**Arguments:**
- `cfg_path` (str): The path to the configuration file.

**Returns:**
- `pitch_angles_rad` (np.ndarray): Array of pitch angles in radians.

### **2. `extract_pointcloud_from_png(png_path, pitch_angles_rad)`**
This function reads a 16-bit PNG depth image and generates a 3D point cloud by converting each pixel to 3D coordinates using spherical to Cartesian transformations.

**Arguments:**
- `png_path` (str): The path to the PNG file containing the depth data.
- `pitch_angles_rad` (np.ndarray): The pitch angles in radians, loaded from the calibration file.

**Returns:**
- `points_nx3` (np.ndarray): The generated 3D point cloud, where each row corresponds to an [x, y, z] coordinate.

### **3. `pointcloud_to_cluster_image()`**
This function is the core of the clustering pipeline. It takes the point cloud and processes it through several stages, including projection, ground removal, and clustering.

**Arguments:**
- `points_nx3` (np.ndarray): The 3D point cloud.
- `pitch_angles_rad` (np.ndarray): Pitch angles in radians.
- `cfg_path` (str): Path to the configuration file.
- `angle_tol_deg` (float): The angular tolerance for clustering.
- `yaw_start_deg` (float, optional): Starting yaw angle for the point cloud.
- `yaw_end_deg` (float, optional): Ending yaw angle for the point cloud.
- `img_width` (int, optional): Image width for the projection.
- `offset_xyz` (tuple, optional): Offset for the sensor.

**Returns:**
- `label_image` (np.ndarray): The clustered label image.
- `depth_image` (np.ndarray): The depth image after projection.
- `clusters` (dict): A dictionary containing the cluster information.

---

## **Pipeline Execution (`run_pipeline.py`)**

This script provides an example of how to run the entire processing pipeline, from loading the LiDAR scan data to running the clustering algorithm and visualizing the results.

Key steps involved:
1. **Loading Calibration Data**: The configuration file is loaded to extract sensor parameters.
2. **Reading PNG Files**: It automatically finds all the `.png` files (LiDAR scans) in the given directory.
3. **Ground Removal**: The `DepthGroundRemover` module removes ground points from the point cloud.
4. **Clustering**: The `ImageBasedClusterer` clusters the remaining points into distinct objects.
5. **Visualization**: Results are visualized using Open3D, showing each cluster in a different color.

The script also supports debugging by outputting intermediate images (e.g., depth images, label images).

---

## **Example Output**
An example of the output from the clustering pipeline is shown below. The label image shows different clusters with unique colors.

![Clustered Point Cloud]('Tests & Python implementation'/cluster_labels_colored2.png)

In the output image:
- Each color represents a different cluster.
- The clusters are generated based on the clustering algorithm applied to the 3D point cloud data.

---

## **Related Publications**

Please cite related papers if you use this code:

**1.** I. Bogoslavskyi and C. Stachniss.  
*Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for Online Operation*.  
In *Proceedings of the International Conference on Intelligent Robots and Systems (IROS)*, 2016.  
[Link to Paper](http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16iros.pdf)

**2.** I. Bogoslavskyi and C. Stachniss.  
*Efficient Online Segmentation for Sparse 3D Laser Scans*.  
In *PFG - Journal of Photogrammetry, Remote Sensing and Geoinformation Science*, 2017.  
[Link to Paper](https://link.springer.com/article/10.1007%2Fs41064-016-0003-y)

---

## **License**
This project is licensed under the MIT License 

---

