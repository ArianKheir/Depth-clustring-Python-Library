import numpy as np
import re
import depth_clustering as dc
import cv2
from PIL import Image

def load_calibration_data(cfg_path):
    """
    Parses the img.cfg file. This file is expected to be a single line of
    semicolon or space-separated values. The first few values are general
    parameters, and the rest are the vertical pitch angles for each laser ring.
    """
    try:
        # Open the configuration file for reading.
        with open(cfg_path, 'r') as f:
            # Read the entire file, remove leading/trailing whitespace.
            content = f.read().strip()

        # Split the content string by either semicolons or one or more whitespace characters.
        # Convert each resulting non-empty string to a float.
        all_values = [float(val) for val in re.split(r'[;\s]+', content) if val]

        # The file must contain at least 5 values (4 for projection params, 1+ for angles).
        if len(all_values) < 5:
            print(f"FATAL ERROR: Calibration file '{cfg_path}' does not contain enough values.")
            return None
        
        # The pitch angles are all values starting from the 5th element (index 4).
        pitch_angles_deg = all_values[4:]
        # Convert the pitch angles from degrees to radians, which is required for trigonometric functions.
        pitch_angles_rad = np.deg2rad(pitch_angles_deg)
        print(f"Successfully loaded {len(pitch_angles_rad)} pitch angles from '{cfg_path}'.")
        return pitch_angles_rad
    except FileNotFoundError:
        # Handle the case where the configuration file does not exist.
        print(f"FATAL ERROR: Calibration file not found at '{cfg_path}'.")
        return None
    except Exception as e:
        # Handle any other potential errors during file reading or parsing.
        print(f"FATAL ERROR: Failed to parse calibration file '{cfg_path}': {e}")
        return None
    
def extract_pointcloud_from_png(png_path: str, pitch_angles_rad: np.ndarray) -> np.ndarray:
    """
    Extract n×3 point cloud from your PNG depth image.
    This function converts a 2D depth image (where pixel intensity represents distance)
    into a 3D point cloud using spherical to Cartesian coordinate transformation.
    
    Returns:
    --------
    points_nx3 : np.ndarray of shape (n, 3) with [x, y, z] coordinates
    """

    # Load the 16-bit PNG image. The depth data is stored in a single 16-bit channel.
    img = Image.open(png_path)
    # Convert the image data to a NumPy array for numerical processing.
    distance_data = np.array(img, dtype=np.uint16)

    # --- Sensor Geometry and Constants ---
    IMG_HEIGHT, IMG_WIDTH = distance_data.shape
    # Define the horizontal field of view (yaw angles).
    YAW_START_DEG, YAW_END_DEG = 180, -180

    # Create an array of yaw angles, one for each column of the image.
    yaw_angles_rad = np.deg2rad(np.linspace(YAW_START_DEG, YAW_END_DEG, IMG_WIDTH))
    
    # Define the sensor's physical offset from the vehicle's coordinate origin.
    OFFSET_X, OFFSET_Y, OFFSET_Z = 0.79, 0.0, 1.73

    # This will store the final [x, y, z] points.
    points_list = []
    
    # Iterate over every pixel in the depth image.
    for r in range(IMG_HEIGHT):  # r corresponds to the row index (pitch angle).
        for c in range(IMG_WIDTH): # c corresponds to the column index (yaw angle).
            # Get the raw 16-bit pixel value.
            pixel_value = distance_data[r, c]
            # A pixel value of 0 indicates no return (no point).
            if pixel_value == 0:
                continue

            # --- Spherical to Cartesian Conversion ---
            # The pixel value is scaled to get the distance in meters.            
            distance_m = pixel_value / 500.0
            # Get the corresponding yaw and pitch for the current pixel column and row.
            yaw = yaw_angles_rad[c]
            pitch = pitch_angles_rad[r]

            # Pre-calculate cosine of pitch as it's used twice.
            cos_pitch = np.cos(pitch)
            # Calculate the 3D coordinates relative to the sensor.
            # Then add the sensor's offset to get the coordinates in the vehicle's frame.
            x = distance_m * np.cos(yaw) * cos_pitch + OFFSET_X
            y = distance_m * np.sin(yaw) * cos_pitch + OFFSET_Y
            z = distance_m * np.sin(pitch) + OFFSET_Z

            # Add the new 3D point to our list.
            points_list.append([x, y, z])

    # Convert the list of points to a NumPy array for efficient processing.   
    return np.array(points_list)

class ClusterCollector(dc.communication.PyAbstractClient_ClusterMap):
    """
    A Python class that acts as a client to receive the clustered clouds.
    It inherits from a C++ base class exposed via pybind11. The C++ library
    will call the `OnNewObjectReceived` method when it has results.
    """
    def __init__(self):
        # Initialize the C++ base class.
        super().__init__()
        # A dictionary to store the final clusters, mapping a label ID to a Cloud object.
        self.clusters = {}

    def OnNewObjectReceived(self, clusters, sender_id):
        """
        This method is the callback that is invoked by the C++ clusterer when
        new clusters are ready. It's the mechanism for passing data from C++ back to Python.
        """
        # Store the received dictionary of clusters.
        self.clusters = clusters

    def get_clusters(self):
        """A helper method to retrieve the stored clusters from Python code."""
        return self.clusters

    def clear(self):
        """Resets the stored clusters."""
        self.clusters = {}
        
class LabelImageCapture(dc.communication.PyAbstractClient_Mat):
    """
    A client class, similar to ClusterCollector, but designed to capture the
    intermediate label image from the clusterer for visualization and analysis.
    It inherits from the `PyAbstractClient_Mat` C++ base class.
    """
    def __init__(self):
        # Initialize the C++ base class.
        super().__init__()
        # This will hold the label image received from C++.
        self.label_image = None
    
    def OnNewObjectReceived(self, label_img, sender_id):
        """
        Callback from C++. The `label_img` is a C++ cv::Mat object, which
        pybind11 automatically converts to a NumPy-like array.
        """
        # We copy the data to ensure it's owned by Python.
        self.label_image = np.array(label_img, copy=True)
    
    def get_label_image(self):
        """Returns the stored label image to the Python caller."""
        return self.label_image if self.label_image is not None else np.array([])

def numpy_to_cloud(
    points_nx3: np.ndarray,
    pitch_angles_rad: np.ndarray,
    yaw_start_deg: float = 180.0,
    yaw_end_deg: float = -180.0,
    img_width: int = 870,
    offset_xyz: tuple = (0.79, 0.0, 1.73)
) -> dc.utils.Cloud:
    """
    Converts a standard NumPy (n, 3) point cloud array to a `depth_clustering.utils.Cloud` object.
    
    The C++ library requires each point to have a "ring index" associated with it. This function
    calculates the pitch angle for each input point and finds the closest matching ring
    from the calibration data to assign this index.
    """
    # Input validation.
    if points_nx3.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3), got {points_nx3.shape}")

    # Instantiate the C++ Cloud object that we will populate.
    cloud = dc.utils.Cloud()
    
    # --- Reverse the offset to get sensor-centric coordinates ---
    offset_x, offset_y, offset_z = offset_xyz
    # Subtracting the offset gives us the coordinates relative to the sensor's origin.
    points_relative = points_nx3 - np.array([offset_x, offset_y, offset_z])
    
    # --- Calculate the pitch angle for every point ---
    x, y, z = points_relative[:, 0], points_relative[:, 1], points_relative[:, 2]
    
    # The horizontal distance from the sensor's vertical axis.
    xy_dist = np.sqrt(x**2 + y**2)
    # The pitch angle is the arctangent of the vertical component (z) and horizontal distance.
    point_pitch = np.arctan2(z, xy_dist)
    
    # --- Assign each point to its nearest ring ---
    for i in range(len(points_nx3)):
        # For the current point's pitch, find the index of the closest pitch angle
        # in the calibration array. This index becomes the ring index.
        ring_idx = np.argmin(np.abs(pitch_angles_rad - point_pitch[i]))
        
        # Create a `RichPoint`, the C++ data structure for a single point.
        # It stores x, y, z coordinates and the calculated ring index.
        point = dc.utils.RichPoint(
            float(points_nx3[i, 0]),
            float(points_nx3[i, 1]),
            float(points_nx3[i, 2]),
            int(ring_idx)
        )
        # Add the newly created RichPoint to the Cloud object
        cloud.push_back(point)
    
    return cloud
    
def pointcloud_to_cluster_image(
    points_nx3: np.ndarray,
    pitch_angles_rad: np.ndarray,
    cfg_path: str,
    angle_tol_deg: float = 30.0,
    yaw_start_deg: float = 180.0,
    yaw_end_deg: float = -180.0,
    img_width: int = 870,
    offset_xyz: tuple = (0.79, 0.0, 1.73)
) -> tuple:
    """
    The main processing function. It takes a raw point cloud and orchestrates the
    C++ library's pipeline to produce a clustered label image and other outputs.
    
    Parameters are documented in the original code.
        
    Returns:
    --------
    tuple: (label_image, depth_image, clusters)
        - label_image: (H, W) uint16 array with a unique ID for each cluster.
        - depth_image: (H, W) float32 array with depth values (distance).
        - clusters: A dictionary mapping each cluster ID to its Cloud object.
    """
    
    # 1. Initialize projection parameters from the 'img.cfg' file.
    params = dc.projections.ProjectionParams.FromConfigFile(cfg_path)
    if params is None:
        raise ValueError(f"Failed to load config from {cfg_path}")
    
    # 2. Create the RingProjection object. This module is responsible for converting the
    #    3D point cloud into a 2D depth image based on the provided parameters.
    projection = dc.projections.RingProjection(params)
    
    # 3. Create the ground remover module. It will take the projected depth image and
    #    attempt to identify and remove ground points.
    angle_tol_rad = dc.utils.Radians.FromDegrees(angle_tol_deg)
    ground_remover = dc.ground_removal.DepthGroundRemover(params, angle_tol_rad)
    
    # 4. Create the main clusterer module. This takes the ground-removed image
    #    and performs the clustering algorithm.
    clusterer = dc.clusterers.ImageBasedClusterer(angle_tol_rad)
    # Set the difference type, which controls how the algorithm decides if two pixels belong to the same cluster.
    clusterer.SetDiffType(dc.image_labelers.DiffType.ANGLES)
    
    # 5. Instantiate our Python callback class to collect the final cluster objects.
    cluster_collector = ClusterCollector()
    
    # 6. Instantiate our Python callback class to capture the intermediate label image.
    label_saver = LabelImageCapture() 
    # Register the label saver with the clusterer.
    clusterer.SetLabelImageClient(label_saver)
    
    # 7. Wire the pipeline together. The output of one module becomes the input of the next.
    #    ground_remover -> clusterer -> cluster_collector
    ground_remover.AddClient(clusterer)
    clusterer.AddClient(cluster_collector)
    
    # 8. Convert the input NumPy point cloud into the C++ library's `Cloud` format.
    cloud = numpy_to_cloud(
        points_nx3, 
        pitch_angles_rad, 
        yaw_start_deg, 
        yaw_end_deg, 
        img_width,
        offset_xyz
    )
    
    # Check if the conversion was successful.
    if cloud is None or cloud.empty():
        raise ValueError("Failed to create valid cloud from input points")
    
    # 9. Set the projection object on the cloud and initialize it.
    #    This step actually performs the projection from 3D points to the 2D depth image.
    cloud.SetProjectionPtr(projection.Clone())
    projection_ptr = cloud.projection_ptr()
    if not projection_ptr:
        raise RuntimeError("Failed to get projection pointer")

    # This C++ method populates the internal depth image inside the projection object.
    projection_ptr.InitFromPoints(cloud.points())
    
    # 10. Get the resulting depth image back into Python as a NumPy array
    depth_image = np.array(projection_ptr.depth_image(), copy=True)
    
    # 11. Run the entire pipeline by feeding the initial cloud to the first module.
    try:
        ground_remover.OnNewObjectReceived(cloud, 0)
    except RuntimeError:
        # The underlying C++ library may sometimes throw an exception upon successful
        # completion when returning to Python. This is often harmless.
        pass
    
    # 12. Retrieve the final results from our Python callback/collector objects.
    label_image = label_saver.get_label_image()
    clusters = cluster_collector.get_clusters()
    
    return label_image, depth_image, clusters
    
# Example usage:
def example_usage():
    """
    Example of how to use the pointcloud_to_cluster_image function.
    """
    # Load calibration
    cfg_path = "/home/ariankheir/Downloads/scenario1/scenario1/img.cfg"
    pitch_angles = load_calibration_data(cfg_path)
    
    # Create or load your point cloud
    png_path = "/home/ariankheir/Downloads/scenario1/scenario1/scan00197.png"
    points_nx3 = extract_pointcloud_from_png(png_path, pitch_angles)

    print(f"Extracted {len(points_nx3)} points")
    print(f"Point cloud shape: {points_nx3.shape}") 
    
    # Run the main clustering pipeline.
    label_image, depth_image, clusters = pointcloud_to_cluster_image(
        points_nx3=points_nx3,
        pitch_angles_rad=pitch_angles,
        cfg_path=cfg_path,
        angle_tol_deg=8.0 # A smaller angle makes clustering stricter.
    )
    
    # --- Visualize and Save Results ---
    print(f"Label image shape: {label_image.shape}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Unique labels: {np.unique(label_image)}")
    
    # Save label image as colored visualization
    if label_image.max() > 0:
        # Normalize to 0-255
        normalized = (label_image * 255.0 / label_image.max()).astype(np.uint8)
        cv2.imwrite("cluster_labels.png", normalized)
        
        # Create color-coded version
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        cv2.imwrite("cluster_labels_colored.png", colored)
    
    # Save depth image
    if depth_image.max() > 0:
        depth_normalized = (depth_image / depth_image.max() * 255).astype(np.uint8)
        cv2.imwrite("depth_image.png", depth_normalized)
    
    return label_image, depth_image, clusters
    
# This is the main entry point when the script is executed.
if __name__ == "__main__":
    example_usage()
