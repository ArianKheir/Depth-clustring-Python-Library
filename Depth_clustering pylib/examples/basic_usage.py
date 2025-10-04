#!/usr/bin/env python3
"""
Basic usage example for the depth_clustering Python library.

This example demonstrates how to:
1. Create and manipulate point clouds
2. Set up projection parameters
3. Perform clustering
4. Remove ground points
"""

import numpy as np
import depth_clustering as dc

def main():
    print("Depth Clustering Python Library - Basic Usage Example")
    print("=" * 50)
    
    # 1. Create a point cloud
    print("\n1. Creating a point cloud...")
    cloud = dc.Cloud()
    
    # Add some sample points (simulating a LiDAR scan)
    for i in range(100):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-2, 5)  # Some points below ground, some above
        ring = i % 16  # Simulate 16-beam LiDAR
        cloud.push_back(dc.RichPoint(x, y, z, ring))
    
    print(f"Created cloud with {cloud.size()} points")
    
    # 2. Set up projection parameters for VLP-16
    print("\n2. Setting up projection parameters...")
    params = dc.ProjectionParams.VLP_16()
    print(f"Projection parameters: {params.rows()} rows x {params.cols()} cols")
    
    # 3. Initialize projection
    print("\n3. Initializing projection...")
    cloud.InitProjection(params)
    print("Projection initialized successfully")
    
    # 4. Create a ground remover
    print("\n4. Creating ground remover...")
    ground_angle = dc.Radians.FromDegrees(5.0)  # 5 degrees threshold
    ground_remover = dc.DepthGroundRemover(params, ground_angle)
    print("Ground remover created")
    
    # 5. Create a clusterer
    print("\n5. Creating Euclidean clusterer...")
    clusterer = dc.EuclideanClusterer(
        cluster_tollerance=0.2,
        min_cluster_size=10,
        max_cluster_size=1000
    )
    print("Clusterer created")
    
    # 6. Create an image-based clusterer
    print("\n6. Creating image-based clusterer...")
    angle_tolerance = dc.Radians.FromDegrees(8.0)
    image_clusterer = dc.ImageBasedClusterer(
        angle_tollerance=angle_tolerance,
        min_cluster_size=10,
        max_cluster_size=1000
    )
    print("Image-based clusterer created")
    
    # 7. Demonstrate pose operations
    print("\n7. Demonstrating pose operations...")
    pose = dc.Pose(1.0, 2.0, 0.5)  # x, y, theta
    print(f"Original pose: x={pose.x():.2f}, y={pose.y():.2f}, theta={pose.theta():.2f}")
    
    # Transform the cloud
    cloud.TransformInPlace(pose)
    print("Cloud transformed")
    
    # 8. Demonstrate radians operations
    print("\n8. Demonstrating radians operations...")
    angle1 = dc.Radians.FromDegrees(45.0)
    angle2 = dc.Radians.FromDegrees(30.0)
    angle_sum = angle1 + angle2
    print(f"45° + 30° = {angle_sum.ToDegrees():.2f}°")
    
    # 9. Create a bounding box
    print("\n9. Creating bounding box...")
    bbox = dc.Bbox(cloud)
    print(f"Bounding box center: ({bbox.center().x():.2f}, {bbox.center().y():.2f}, {bbox.center().z():.2f})")
    print(f"Bounding box scale: ({bbox.scale().x():.2f}, {bbox.scale().y():.2f}, {bbox.scale().z():.2f})")
    print(f"Bounding box volume: {bbox.volume():.2f}")
    
    # 10. Demonstrate folder reader
    print("\n10. Demonstrating folder reader...")
    # Note: This would require actual files in the specified directory
    # folder_reader = dc.FolderReader("/path/to/pointcloud/files", ".pcd", dc.Order.SORTED)
    # print("Folder reader created (commented out for demo)")
    
    print("\n" + "=" * 50)
    print("Basic usage example completed successfully!")
    print("The depth_clustering library is ready to use.")

if __name__ == "__main__":
    main()
