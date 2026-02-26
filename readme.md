# Depth Clustering Python Library

A Python library built on top of the [depth_clustering C++ library](https://github.com/PRBonn/depth_clustering), providing pybind11 bindings for seamless integration with Python. The library performs fast range image-based segmentation and clustering on 3D point cloud data from LiDAR scans.

## Features

- Python bindings to the depth_clustering C++ library via pybind11
- Fast conversion of 3D point clouds from NumPy arrays
- Full pipeline: projection → ground removal → clustering
- Customizable parameters for clustering and ground removal
- Outputs: clustered label images, depth images, and cluster dictionaries

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Point Cloud Clustering Example](#point-cloud-clustering-example)
   - [Running the Full Pipeline](#running-the-full-pipeline)
3. [API Reference](#api-reference)
4. [Example Output](#example-output)
5. [Related Publications](#related-publications)
6. [License](#license)

---

## Installation

### 1. System Dependencies

Install the required system-level libraries via conda before building:

```bash
conda install -c conda-forge pcl eigen boost opencv
```

### 2. Clone the Repository

```bash
git clone https://github.com/ArianKheir/Depth-clustring-Python-Library.git
cd Depth-clustring-Python-Library/Depth_clustering\ pylib
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build and Install the Package

```bash
rm -rf build dist *.egg-info
pip install --no-build-isolation .
```

> **Note:** `--no-build-isolation` is required because the build depends on system-level conda packages (PCL, OpenCV, Eigen) that must be visible to the compiler.

---

## Dependencies

### Python (via `requirements.txt`)

| Package | Version |
|---|---|
| numpy | ≥ 1.21.0 |
| pybind11 | ≥ 2.6.0 |
| opencv-python | ≥ 4.5.0 |
| open3d | ≥ 0.15.0 |
| scipy | ≥ 1.7.0 |

### System (via conda)

| Package | Purpose |
|---|---|
| PCL ≥ 1.12 | Point cloud processing |
| Eigen ≥ 3.4 | Linear algebra |
| Boost ≥ 1.74 | C++ utilities |
| OpenCV ≥ 4.5 | Image handling |

---

## Usage

### Point Cloud Clustering Example

```python
import numpy as np
from pointcloud_func import pointcloud_to_cluster_image, load_calibration_data, extract_pointcloud_from_png

cfg_path = "path/to/config_file.cfg"
png_path = "path/to/lidar_scan.png"

# Load sensor calibration (pitch angles)
pitch_angles = load_calibration_data(cfg_path)

# Convert depth PNG to 3D point cloud
points_nx3 = extract_pointcloud_from_png(png_path, pitch_angles)

# Run the full clustering pipeline
label_image, depth_image, clusters = pointcloud_to_cluster_image(
    points_nx3=points_nx3,
    pitch_angles_rad=pitch_angles,
    cfg_path=cfg_path
)

print("Label Image Shape:", label_image.shape)
print("Number of Clusters:", len(clusters))
```

### Running the Full Pipeline

To process a directory of LiDAR scans in bulk:

```bash
python run_pipeline.py --data_path /path/to/lidar/scans --cfg_path /path/to/config.cfg
```

This automatically loads all `.png` scan files, runs ground removal and clustering, and visualizes each cluster in a distinct color using Open3D.

---

## API Reference

### `load_calibration_data(cfg_path)`

Loads sensor calibration parameters from a `.cfg` file and returns pitch angles in radians.

| Argument | Type | Description |
|---|---|---|
| `cfg_path` | `str` | Path to the configuration file |

**Returns:** `pitch_angles_rad` — `np.ndarray` of pitch angles in radians.

---

### `extract_pointcloud_from_png(png_path, pitch_angles_rad)`

Reads a 16-bit PNG depth image and converts it to a 3D point cloud using spherical-to-Cartesian projection.

| Argument | Type | Description |
|---|---|---|
| `png_path` | `str` | Path to the 16-bit depth PNG file |
| `pitch_angles_rad` | `np.ndarray` | Pitch angles from calibration |

**Returns:** `points_nx3` — `np.ndarray` of shape `(N, 3)` with `[x, y, z]` coordinates.

---

### `pointcloud_to_cluster_image(...)`

Core clustering pipeline: projects the point cloud, removes ground, and clusters remaining points.

| Argument | Type | Default | Description |
|---|---|---|---|
| `points_nx3` | `np.ndarray` | — | Input point cloud `(N, 3)` |
| `pitch_angles_rad` | `np.ndarray` | — | Sensor pitch angles |
| `cfg_path` | `str` | — | Path to config file |
| `angle_tol_deg` | `float` | — | Angular tolerance for clustering |
| `yaw_start_deg` | `float` | optional | Starting yaw angle |
| `yaw_end_deg` | `float` | optional | Ending yaw angle |
| `img_width` | `int` | optional | Projection image width |
| `offset_xyz` | `tuple` | optional | Sensor position offset |

**Returns:** `(label_image, depth_image, clusters)` — label `np.ndarray`, depth `np.ndarray`, and cluster `dict`.

---

## Example Output

The label image below shows the result of the clustering pipeline on a 360° LiDAR scan. Each color represents a distinct cluster.

![Clustered Point Cloud](./Tests%20%26%20Python%20implementation/cluster_labels_colored2.png)

---

## Related Publications

If you use this library, please cite the original depth_clustering work:

**[1]** I. Bogoslavskyi and C. Stachniss.
*Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for Online Operation.*
IROS 2016. [PDF](http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16iros.pdf)

**[2]** I. Bogoslavskyi and C. Stachniss.
*Efficient Online Segmentation for Sparse 3D Laser Scans.*
PFG Journal of Photogrammetry, Remote Sensing and Geoinformation Science, 2017.
[Link](https://link.springer.com/article/10.1007%2Fs41064-016-0003-y)

---

## License

This project is licensed under the MIT License.
