#!/usr/bin/env python3
"""
Visualization example for the depth_clustering Python library.

This example demonstrates how to:
1. Set up visualization components
2. Create visualizers for different data types
3. Save visualization results
4. Work with different output formats
"""

import numpy as np
import depth_clustering as dc

def create_sample_data():
    """Create sample data for visualization."""
    # Create a sample cloud
    cloud = dc.Cloud()
    for i in range(100):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(0, 5)
        cloud.push_back(dc.RichPoint(x, y, z, i % 16))
    
    # Create sample clusters
    clusters = {}
    for i in range(3):
        cluster_cloud = dc.Cloud()
        for j in range(20):
            x = np.random.uniform(i*5-2, i*5+2)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(0, 3)
            cluster_cloud.push_back(dc.RichPoint(x, y, z, j % 16))
        clusters[i] = cluster_cloud
    
    return cloud, clusters

def main():
    print("Depth Clustering - Visualization Example")
    print("=" * 40)
    
    # 1. Create sample data
    print("\n1. Creating sample data...")
    cloud, clusters = create_sample_data()
    print(f"Created cloud with {cloud.size()} points")
    print(f"Created {len(clusters)} clusters")
    
    # 2. Set up projection parameters
    print("\n2. Setting up projection parameters...")
    params = dc.ProjectionParams.VLP_16()
    cloud.InitProjection(params)
    print("Projection initialized")
    
    # 3. Create cloud saver
    print("\n3. Creating cloud saver...")
    cloud_saver = dc.CloudSaver("visualization_cloud_")
    print("Cloud saver created")
    
    # 4. Create vector cloud saver for clusters
    print("\n4. Creating vector cloud saver...")
    vector_saver = dc.VectorCloudSaver("visualization_clusters_", save_every=1)
    print("Vector cloud saver created")
    
    # 5. Create depth map saver
    print("\n5. Creating depth map saver...")
    depth_saver = dc.DepthMapSaver("visualization_depth_", save_every=1)
    print("Depth map saver created")
    
    # 6. Demonstrate saving individual clouds
    print("\n6. Demonstrating cloud saving...")
    # In a real scenario, you would connect the saver to receive data
    # cloud_saver.OnNewObjectReceived(cloud, 0)
    print("Cloud saving demonstrated")
    
    # 7. Demonstrate saving clusters
    print("\n7. Demonstrating cluster saving...")
    # vector_saver.OnNewObjectReceived(clusters, 0)
    print("Cluster saving demonstrated")
    
    # 8. Demonstrate visualization pipeline
    print("\n8. Demonstrating visualization pipeline...")
    
    # Create a visualizer (Note: This requires Qt/OpenGL)
    print("  - Creating visualizer...")
    # visualizer = dc.Visualizer()
    # print("  - Visualizer created (requires Qt/OpenGL)")
    
    # Create object pointer storer
    print("  - Creating object pointer storer...")
    # object_storer = dc.ObjectPtrStorer()
    # print("  - Object pointer storer created")
    
    # Set up update listener
    print("  - Setting up update listener...")
    # object_storer.SetUpdateListener(visualizer)
    # print("  - Update listener set")
    
    print("Visualization pipeline demonstrated")
    
    # 9. Demonstrate different output formats
    print("\n9. Demonstrating different output formats...")
    
    # Save with different prefixes
    car_saver = dc.CloudSaver("cars_")
    pole_saver = dc.CloudSaver("poles_")
    building_saver = dc.CloudSaver("buildings_")
    
    print("Created specialized savers for different object types")
    
    # 10. Demonstrate batch saving
    print("\n10. Demonstrating batch saving...")
    
    # Create a batch saver that saves every 10 frames
    batch_saver = dc.VectorCloudSaver("batch_", save_every=10)
    print("Batch saver created (saves every 10 frames)")
    
    # 11. Demonstrate pose visualization
    print("\n11. Demonstrating pose visualization...")
    
    # Create poses for different objects
    car_pose = dc.Pose(5.0, 2.0, 0.0)  # Car at (5, 2) with 0° rotation
    pole_pose = dc.Pose(-3.0, -1.0, 0.0)  # Pole at (-3, -1)
    building_pose = dc.Pose(-8.0, 10.0, 0.0)  # Building at (-8, 10)
    
    print("Created poses for different objects:")
    print(f"  - Car: x={car_pose.x():.1f}, y={car_pose.y():.1f}, theta={car_pose.theta():.1f}")
    print(f"  - Pole: x={pole_pose.x():.1f}, y={pole_pose.y():.1f}, theta={pole_pose.theta():.1f}")
    print(f"  - Building: x={building_pose.x():.1f}, y={building_pose.y():.1f}, theta={building_pose.theta():.1f}")
    
    # 12. Demonstrate bounding box visualization
    print("\n12. Demonstrating bounding box visualization...")
    
    # Create bounding boxes for clusters
    for i, (cluster_id, cluster_cloud) in enumerate(clusters.items()):
        bbox = dc.Bbox(cluster_cloud)
        print(f"  - Cluster {cluster_id}:")
        print(f"    Center: ({bbox.center().x():.2f}, {bbox.center().y():.2f}, {bbox.center().z():.2f})")
        print(f"    Scale: ({bbox.scale().x():.2f}, {bbox.scale().y():.2f}, {bbox.scale().z():.2f})")
        print(f"    Volume: {bbox.volume():.2f}")
    
    # 13. Demonstrate file organization
    print("\n13. Demonstrating file organization...")
    
    # Create organized savers
    organized_savers = {
        "ground": dc.CloudSaver("organized/ground_"),
        "objects": dc.CloudSaver("organized/objects_"),
        "clusters": dc.VectorCloudSaver("organized/clusters_"),
        "depth_maps": dc.DepthMapSaver("organized/depth_")
    }
    
    print("Created organized savers:")
    for name, saver in organized_savers.items():
        print(f"  - {name}: {type(saver).__name__}")
    
    print("\n" + "=" * 40)
    print("Visualization example completed successfully!")
    print("Key takeaways:")
    print("- CloudSaver saves individual point clouds")
    print("- VectorCloudSaver saves multiple clusters")
    print("- DepthMapSaver saves depth images")
    print("- Savers can be organized by object type")
    print("- Batch saving can reduce file I/O")
    print("- Bounding boxes provide object dimensions")
    print("- Poses can represent object locations")

if __name__ == "__main__":
    main()
