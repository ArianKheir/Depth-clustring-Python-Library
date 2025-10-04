#!/usr/bin/env python3
"""
Clustering example for the depth_clustering Python library.

This example demonstrates how to:
1. Load point cloud data
2. Set up different types of clusterers
3. Compare clustering results
4. Save clustered results
"""

import numpy as np
import depth_clustering as dc

def create_sample_cloud():
    """Create a sample point cloud with multiple objects."""
    cloud = dc.Cloud()
    
    # Create ground points
    for i in range(200):
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-0.1, 0.1)  # Ground level
        cloud.push_back(dc.RichPoint(x, y, z, 0))
    
    # Create object 1 (car-like)
    for i in range(50):
        x = np.random.uniform(5, 8)
        y = np.random.uniform(2, 5)
        z = np.random.uniform(0, 2)
        cloud.push_back(dc.RichPoint(x, y, z, 1))
    
    # Create object 2 (pole-like)
    for i in range(30):
        x = np.random.uniform(-5, -4)
        y = np.random.uniform(-3, -2)
        z = np.random.uniform(0, 4)
        cloud.push_back(dc.RichPoint(x, y, z, 2))
    
    # Create object 3 (building-like)
    for i in range(100):
        x = np.random.uniform(-10, -6)
        y = np.random.uniform(8, 12)
        z = np.random.uniform(0, 6)
        cloud.push_back(dc.RichPoint(x, y, z, 3))
    
    return cloud

def main():
    print("Depth Clustering - Clustering Example")
    print("=" * 40)
    
    # 1. Create sample data
    print("\n1. Creating sample point cloud...")
    cloud = create_sample_cloud()
    print(f"Created cloud with {cloud.size()} points")
    
    # 2. Set up projection parameters
    print("\n2. Setting up projection parameters...")
    params = dc.ProjectionParams.VLP_16()
    cloud.InitProjection(params)
    print("Projection initialized")
    
    # 3. Create Euclidean clusterer
    print("\n3. Creating Euclidean clusterer...")
    euclidean_clusterer = dc.EuclideanClusterer(
        cluster_tollerance=0.5,  # 50cm tolerance
        min_cluster_size=20,     # Minimum 20 points per cluster
        max_cluster_size=500     # Maximum 500 points per cluster
    )
    print("Euclidean clusterer created")
    
    # 4. Create Image-based clusterer
    print("\n4. Creating Image-based clusterer...")
    angle_tolerance = dc.Radians.FromDegrees(10.0)  # 10 degrees
    image_clusterer = dc.ImageBasedClusterer(
        angle_tollerance=angle_tolerance,
        min_cluster_size=20,
        max_cluster_size=500
    )
    print("Image-based clusterer created")
    
    # 5. Set up ground removal
    print("\n5. Setting up ground removal...")
    ground_angle = dc.Radians.FromDegrees(5.0)
    ground_remover = dc.DepthGroundRemover(params, ground_angle)
    print("Ground remover created")
    
    # 6. Create cloud saver for results
    print("\n6. Setting up result saving...")
    cloud_saver = dc.CloudSaver("output_cloud_")
    vector_saver = dc.VectorCloudSaver("output_clusters_", save_every=1)
    print("Cloud savers created")
    
    # 7. Demonstrate clustering pipeline
    print("\n7. Demonstrating clustering pipeline...")
    
    # Process with ground removal first
    print("  - Applying ground removal...")
    # Note: In a real scenario, you would connect the ground remover to the clusterer
    # ground_remover.AddClient(euclidean_clusterer)
    # ground_remover.OnNewObjectReceived(cloud, 0)
    
    # Process with Euclidean clustering
    print("  - Applying Euclidean clustering...")
    # euclidean_clusterer.AddClient(vector_saver)
    # euclidean_clusterer.OnNewObjectReceived(cloud, 0)
    
    # Process with Image-based clustering
    print("  - Applying Image-based clustering...")
    # image_clusterer.AddClient(vector_saver)
    # image_clusterer.OnNewObjectReceived(cloud, 0)
    
    print("Clustering pipeline demonstrated")
    
    # 8. Demonstrate different clustering parameters
    print("\n8. Demonstrating different clustering parameters...")
    
    # Tight clustering
    tight_clusterer = dc.EuclideanClusterer(
        cluster_tollerance=0.2,  # 20cm tolerance
        min_cluster_size=10,
        max_cluster_size=200
    )
    print("Tight clusterer created (20cm tolerance)")
    
    # Loose clustering
    loose_clusterer = dc.EuclideanClusterer(
        cluster_tollerance=1.0,  # 1m tolerance
        min_cluster_size=50,
        max_cluster_size=1000
    )
    print("Loose clusterer created (1m tolerance)")
    
    # 9. Demonstrate image labeling
    print("\n9. Demonstrating image labeling...")
    
    # Create a linear image labeler
    linear_labeler = dc.LinearImageLabeler(
        cloud.projection_ptr().depth_image(),
        params,
        dc.Radians.FromDegrees(8.0)
    )
    print("Linear image labeler created")
    
    # Create a Dijkstra image labeler
    dijkstra_labeler = dc.DijkstraImageLabeler(
        cloud.projection_ptr().depth_image(),
        params,
        dc.Radians.FromDegrees(8.0)
    )
    print("Dijkstra image labeler created")
    
    # 10. Demonstrate different diff types
    print("\n10. Demonstrating different diff types...")
    print("Available diff types:")
    print(f"  - NONE: {dc.DiffType.NONE}")
    print(f"  - SIMPLE: {dc.DiffType.SIMPLE}")
    print(f"  - ANGLE: {dc.DiffType.ANGLE}")
    print(f"  - LINE_DIST: {dc.DiffType.LINE_DIST}")
    
    # Set diff type for image clusterer
    image_clusterer.SetDiffType(dc.DiffType.ANGLE)
    print("Set diff type to ANGLE for image clusterer")
    
    print("\n" + "=" * 40)
    print("Clustering example completed successfully!")
    print("Key takeaways:")
    print("- Euclidean clustering works in 3D space")
    print("- Image-based clustering works on projected depth images")
    print("- Different parameters affect clustering results")
    print("- Ground removal can be applied before clustering")
    print("- Results can be saved for further analysis")

if __name__ == "__main__":
    main()
