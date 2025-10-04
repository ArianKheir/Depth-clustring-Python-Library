import numpy as np
import depth_clustering as dc
import open3d as o3d
import argparse
import pathlib
import time
from PIL import Image

# ==============================================================================
# 1. Custom Data Loader for Your PNG-based Dataset
# ==============================================================================
import re
import cv2

class LabelImageSaver(dc.communication.PyAbstractClient_Mat):
    """Saves the label image for debugging"""
    def __init__(self, output_dir="./debug_labels"):
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.frame_count = 0
    
    def OnNewObjectReceived(self, label_img, sender_id):
        """Called when the clusterer generates a label image"""
        # Convert cv::Mat to numpy array
        label_array = np.array(label_img, copy=False)
        
        max_label = label_array.max()
        unique_labels = np.unique(label_array)
        
        print(f"DEBUG LABELS: Frame {self.frame_count}, Max label: {max_label}, Unique labels: {len(unique_labels)}")
        print(f"DEBUG LABELS: Label range: {label_array.min()} to {label_array.max()}")
        print(f"DEBUG LABELS: Non-zero pixels: {np.count_nonzero(label_array)}")
        
        # Save normalized image
        if max_label > 0:
            normalized = (label_array * 255.0 / max_label).astype(np.uint8)
        else:
            normalized = label_array.astype(np.uint8)
        
        output_path = self.output_dir / f"labels_{self.frame_count:05d}.png"
        cv2.imwrite(str(output_path), normalized)
        print(f"DEBUG LABELS: Saved to {output_path}")
        
        self.frame_count += 1

def load_calibration_data(cfg_path):
    """
    Parses the img.cfg file. This file is expected to be a single line of
    semicolon or space-separated values.
    """
    try:
        with open(cfg_path, 'r') as f:
            content = f.read().strip()

        all_values = [float(val) for val in re.split(r'[;\s]+', content) if val]

        if len(all_values) < 5:
            print(f"FATAL ERROR: Calibration file '{cfg_path}' does not contain enough values.")
            return None

        pitch_angles_deg = all_values[4:]
        pitch_angles_rad = np.deg2rad(pitch_angles_deg)
        print(f"Successfully loaded {len(pitch_angles_rad)} pitch angles from '{cfg_path}'.")
        return pitch_angles_rad
    except FileNotFoundError:
        print(f"FATAL ERROR: Calibration file not found at '{cfg_path}'.")
        return None
    except Exception as e:
        print(f"FATAL ERROR: Failed to parse calibration file '{cfg_path}': {e}")
        return None

def read_png_to_cloud(png_path, pitch_angles_rad):
    """
    Reads a 16-bit PNG depth image, converts it to a 3D point cloud using
    the provided calibration, and returns a depth_clustering.utils.Cloud object.
    """
    try:
        img = Image.open(png_path)
        if img.mode not in ['I;16', 'I']:
            print(f"Warning: Image mode is {img.mode}, not a 16-bit format. File: '{png_path}'")

        distance_data = np.array(img, dtype=np.uint16)

    except FileNotFoundError:
        print(f"Error: PNG file not found at {png_path}")
        return None
    except Exception as e:
        print(f"Error reading or converting PNG '{png_path}': {e}")
        return None

    IMG_HEIGHT, IMG_WIDTH = distance_data.shape
    YAW_START_DEG, YAW_END_DEG = 180, -180

    if IMG_HEIGHT != len(pitch_angles_rad):
        print(f"FATAL ERROR: Image height ({IMG_HEIGHT}) does not match number of pitch angles ({len(pitch_angles_rad)}).")
        return None

    yaw_angles_rad = np.deg2rad(np.linspace(YAW_START_DEG, YAW_END_DEG, IMG_WIDTH))

    OFFSET_X, OFFSET_Y, OFFSET_Z = 0.79, 0.0, 1.73

    cloud = dc.utils.Cloud()

    for r in range(IMG_HEIGHT):
        for c in range(IMG_WIDTH):
            pixel_value = distance_data[r, c]
            if pixel_value == 0:
                continue
            distance_m = pixel_value / 500.0

            yaw = yaw_angles_rad[c]
            pitch = pitch_angles_rad[r]

            cos_pitch = np.cos(pitch)
            x = distance_m * np.cos(yaw) * cos_pitch
            y = distance_m * np.sin(yaw) * cos_pitch
            z = distance_m * np.sin(pitch)

            x_final = x + OFFSET_X
            y_final = y + OFFSET_Y
            z_final = z + OFFSET_Z

            point = dc.utils.RichPoint(x_final, y_final, z_final, r)
            cloud.push_back(point)

    return cloud

# ==============================================================================
# 2. Collector and Visualization Helpers
# ==============================================================================

class ClusterCollector(dc.communication.PyAbstractClient_ClusterMap):
    """
    A Python class that acts as a client to receive the clustered clouds.
    """
    def __init__(self):
        print("DEBUG: ClusterCollector.__init__() - START")
        super().__init__()
        self.clusters = {}
        print("DEBUG: ClusterCollector.__init__() - END")

    def OnNewObjectReceived(self, clusters, sender_id):
        """
        This method is called by the C++ clusterer when new clusters are ready.
        """
        print(f"DEBUG: ClusterCollector.OnNewObjectReceived - Received {len(clusters)} clusters from sender ID {sender_id}")
        self.clusters = clusters
        print("DEBUG: ClusterCollector.OnNewObjectReceived - END")

    def get_clusters(self):
        return self.clusters

    def clear(self):
        self.clusters = {}

def cloud_to_open3d(dc_cloud, color=None):
    """Converts a dc.utils.Cloud to an o3d.geometry.PointCloud."""
    points = np.array([[p.x(), p.y(), p.z()] for p in dc_cloud.points()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

# ==============================================================================
# 3. Main Pipeline Execution
# ==============================================================================

def run_pipeline(data_path, cfg_path, angle_tol_deg):
    """
    Sets up and runs the full ground removal and clustering pipeline.
    """
    print("DEBUG: run_pipeline - START")
    
    # --- Load Calibration Data ---
    print("DEBUG: Loading calibration data...")
    pitch_angles = load_calibration_data(cfg_path)
    if pitch_angles is None:
        return

    # --- Find all PNG files in the specified directory ---
    print("DEBUG: Finding PNG files...")
    png_files = sorted(list(pathlib.Path(data_path).glob("scan*.png")))
    if not png_files:
        print(f"Error: No 'scan*.png' files found in directory '{data_path}'.")
        return
    print(f"DEBUG: Found {len(png_files)} PNG files")

    # --- Setup Pipeline Components ---
    print("DEBUG: Initializing ProjectionParams...")
    print("Initializing C++ ProjectionParams from config file...")
    params = dc.projections.ProjectionParams.FromConfigFile(cfg_path)
    print("C++ ProjectionParams initialized successfully.")
    if params is None:
        print("FATAL ERROR: Failed to create ProjectionParams from config file.")
        return
    
    print("DEBUG: Creating RingProjection...")
    projection = dc.projections.RingProjection(params)
    print("DEBUG: RingProjection created successfully")

    # --- Setup Ground Remover ---
    print("DEBUG: Creating DepthGroundRemover...")
    print(f"Initializing DepthGroundRemover with angle tolerance {angle_tol_deg} degrees...")
    angle_tol_rad = dc.utils.Radians.FromDegrees(angle_tol_deg)
    depth_ground_remover = dc.ground_removal.DepthGroundRemover(params, angle_tol_rad)
    print("DepthGroundRemover initialized successfully.")

    # Clusterer (second stage)
    print("DEBUG: Creating ImageBasedClusterer...")
    angle_tol_rad = dc.utils.Radians.FromDegrees(angle_tol_deg)
    print("Initializing ImageBasedClusterer...")
    clusterer = dc.clusterers.ImageBasedClusterer(
        angle_tol_rad)
    print("ImageBasedClusterer initialized successfully.")
    
    print("DEBUG: Setting clusterer diff type...")
    clusterer.SetDiffType(dc.image_labelers.DiffType.ANGLES)
    print("DEBUG: Diff type set successfully")

    # ADD THIS: Wire label image saver for debugging
    print("DEBUG: Creating LabelImageSaver...")
    label_saver = LabelImageSaver()
    clusterer.SetLabelImageClient(label_saver)
    print("DEBUG: LabelImageSaver attached to clusterer")
    # Python Cluster Collector (final stage)
    print("DEBUG: Creating ClusterCollector...")
    cluster_collector = ClusterCollector()
    print("DEBUG: ClusterCollector created successfully")

    # --- Wire the Pipeline Together ---
    print("DEBUG: Wiring pipeline - adding clusterer as client to ground_remover...")
    try:
        depth_ground_remover.AddClient(clusterer)
        print("DEBUG: Successfully added clusterer to ground_remover")
    except Exception as e:
        print(f"DEBUG: ERROR adding clusterer to ground_remover: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("DEBUG: Wiring pipeline - adding cluster_collector as client to clusterer...")
    try:
        clusterer.AddClient(cluster_collector)
        print("DEBUG: Successfully added cluster_collector to clusterer")
    except Exception as e:
        print(f"DEBUG: ERROR adding cluster_collector to clusterer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Pipeline wired: GroundRemover -> Clusterer -> Python Collector")

    # --- Setup Open3D Visualization ---
    print("DEBUG: Creating Open3D visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Clustering Pipeline")
    print("DEBUG: Open3D visualizer created")

    # Main processing loop for each frame
    print("DEBUG: Starting frame processing loop...")
    for i, file_path in enumerate(png_files):
        frame_start_time = time.time()
        print(f"\n--- Processing Frame {i+1}/{len(png_files)}: {file_path.name} ---")

        # Clear previous frame's data
        print("DEBUG: Clearing previous frame data...")
        cluster_collector.clear()
        vis.clear_geometries()

        # 1. Load data using our custom function
        print("DEBUG: Loading cloud from PNG...")
        cloud = read_png_to_cloud(str(file_path), pitch_angles)
        if cloud is None or cloud.empty():
            print("Skipping empty or invalid cloud.")
            continue
        print(f"Loaded cloud with {cloud.size()} points.")

        # 2. Prepare the cloud for the C++ pipeline
        # Set projection and project points to depth image
        print("DEBUG: Setting projection on cloud...")
        cloud.SetProjectionPtr(projection.Clone())

        print("DEBUG: Projecting points to depth image...")
        projection_ptr = cloud.projection_ptr()
        if projection_ptr:
            projection_ptr.InitFromPoints(cloud.points())
            print("DEBUG: Points projected successfully")
        else:
            print("ERROR: Failed to get projection pointer")
            continue

        # 3. Trigger the pipeline by sending the cloud to the first component
        print("DEBUG: Triggering pipeline with OnNewObjectReceived...")
        try:
            depth_ground_remover.OnNewObjectReceived(cloud, 0)
            print("DEBUG: OnNewObjectReceived completed successfully")
        except RuntimeError as e:
            print(f"DEBUG: RuntimeError in OnNewObjectReceived: {e}")
            # The processing actually completed (you can see the timing logs)
            # The error is just in the return value conversion
            # So we can continue and try to get results anyway
            pass
        except Exception as e:
            print(f"DEBUG: ERROR in OnNewObjectReceived: {e}")
            import traceback
            traceback.print_exc()
            # Don't continue for other exceptions
            continue
        projection_ptr = cloud.projection_ptr()
        if projection_ptr:
            depth_img = projection_ptr.depth_image()
            depth_array = np.array(depth_img, copy=False)
            
            print(f"DEBUG DEPTH: Shape: {depth_array.shape}")
            print(f"DEBUG DEPTH: Non-zero pixels: {np.count_nonzero(depth_array)}")
            print(f"DEBUG DEPTH: Min: {depth_array.min():.3f}, Max: {depth_array.max():.3f}")
            print(f"DEBUG DEPTH: Mean (non-zero): {depth_array[depth_array > 0].mean():.3f}")
            
            # Save depth image for first frame
            if i == 0:
                normalized_depth = np.zeros_like(depth_array, dtype=np.uint8)
                mask = depth_array > 0
                if mask.any():
                    normalized_depth[mask] = (depth_array[mask] / depth_array[mask].max() * 255).astype(np.uint8)
                cv2.imwrite("debug_depth_after_ground_removal.png", normalized_depth)
                print("DEBUG DEPTH: Saved debug_depth_after_ground_removal.png")
        else:
            print("DEBUG DEPTH: ERROR - No projection found on cloud!")
        # 4. Retrieve results from our Python collector
        print("DEBUG: Retrieving clusters...")
        clusters = cluster_collector.get_clusters()

        # 5. Visualize the results
        if not clusters:
            print("No clusters found in this frame.")
        else:
            print(f"Visualizing {len(clusters)} found clusters.")
            for label, cluster_cloud in clusters.items():
                color = np.random.rand(3)
                o3d_pcd = cloud_to_open3d(cluster_cloud, color=color)
                vis.add_geometry(o3d_pcd)
                bbox = o3d_pcd.get_axis_aligned_bounding_box()
                bbox.color = color
                vis.add_geometry(bbox)

        # Update the visualizer
        print("DEBUG: Updating visualizer...")
        vis.poll_events()
        vis.update_renderer()

        # Control the frame rate to ~10 FPS
        frame_time = time.time() - frame_start_time
        sleep_time = max(0, 0.1 - frame_time)
        time.sleep(sleep_time)

    print("DEBUG: Destroying visualizer window...")
    vis.destroy_window()
    print("\n--- Pipeline finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LiDAR ground removal and clustering on a PNG-based dataset and visualize with Open3D."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the directory containing the 'scan*.png' files."
    )
    parser.add_argument(
        "--cfg_path",
        required=True,
        help="Path to the 'img.cfg' calibration file containing pitch angles."
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=8.0,
        help="Angle tolerance in degrees for the image-based clusterer."
    )
    args = parser.parse_args()

    run_pipeline(args.data_path, args.cfg_path, args.angle)

