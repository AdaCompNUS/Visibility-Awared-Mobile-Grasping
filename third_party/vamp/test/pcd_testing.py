#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import open3d as o3d
import vamp
from vamp import pybullet_interface as vpb
import time
import os
import glob


def load_pointcloud(pcd_path):
    """Load a point cloud from file using Open3D."""
    print(f"Loading point cloud from {pcd_path}...")

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        print(f"Loaded {len(points)} points from {os.path.basename(pcd_path)}")
        return points
    except Exception as e:
        print(f"Error loading point cloud {pcd_path}: {e}")
        return None


def load_all_pointclouds(directory):
    """
    Load all point cloud files from a directory.

    Args:
        directory: Directory containing point cloud files

    Returns:
        Combined point cloud as a numpy array
    """
    print(f"Searching for point cloud files in {directory}...")

    # Find all point cloud files (PLY, PCD, etc.)
    ply_files = glob.glob(os.path.join(directory, "*.ply"))
    pcd_files = glob.glob(os.path.join(directory, "*.pcd"))

    all_files = ply_files + pcd_files
    print(f"Found {len(all_files)} point cloud files")

    if not all_files:
        print(f"No point cloud files found in {directory}")
        return None

    # Load and combine all point clouds
    combined_points = []
    total_points = 0

    for file_path in all_files:
        pc = load_pointcloud(file_path)
        if pc is not None and len(pc) > 0:
            combined_points.append(pc)
            total_points += len(pc)

    if not combined_points:
        print("No valid point clouds loaded")
        return None

    # Combine all point clouds into a single array
    combined_pc = np.vstack(combined_points)
    print(
        f"Combined {len(combined_points)} point clouds with total {len(combined_pc)} points"
    )

    return combined_pc


def filter_points_by_radius(points, max_radius):
    """
    Filter out points beyond a specified radius from the origin.

    Args:
        points: Nx3 numpy array of points
        max_radius: Maximum allowed distance from origin

    Returns:
        Filtered points as a numpy array
    """
    # Calculate distance of each point from origin (0,0,0)
    distances = np.linalg.norm(points, axis=1)

    # Create a mask for points within the radius
    mask = distances <= max_radius

    # Apply the mask to get filtered points
    filtered_points = points[mask]

    print(
        f"Filtered points: kept {len(filtered_points)} of {len(points)} points (radius <= {max_radius})"
    )

    return filtered_points


def transform_points_fast(points, transform_matrix):
    """
    Transform points using fast NumPy vectorized operations.

    Args:
        points: Nx3 numpy array of points
        transform_matrix: 4x4 transformation matrix

    Returns:
        Nx3 numpy array of transformed points
    """
    # Ensure input is numpy array
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # Extract rotation matrix and translation vector from transform matrix
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]

    # Apply transformation to all points at once
    # R * points + t
    transformed_points = np.dot(points, rotation.T) + translation

    return transformed_points


def sample_points_for_visualization(points, max_points=100000):
    """
    Sample a subset of points for visualization to avoid PyBullet limits.

    Args:
        points: Nx3 numpy array of points
        max_points: Maximum number of points to return

    Returns:
        Subset of points as a list
    """
    if len(points) <= max_points:
        return points.tolist()

    # Calculate sampling ratio to get approximately max_points
    max_points / len(points)

    # Use random sampling
    indices = np.random.choice(len(points), size=max_points, replace=False)
    sampled_points = points[indices]

    print(
        f"Sampled {len(sampled_points)} points for visualization (from {len(points)} total)"
    )
    return sampled_points.tolist()


def main():
    """Test script for visualizing point cloud transformation using VAMP's built-in functions."""
    # Directory containing point cloud files
    pc_directory = "mp_collision_models"

    # Load all point cloud data from directory
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print("Failed to load point clouds. Exiting.")
        return

    # Define transformation matrix from map to base_link
    transform_matrix = np.array(
        [
            [0.1470, -0.9890, 0, 0.4500],
            [0.9890, 0.1470, 0, 2.7723],
            [0, 0, 1, 0.0000],
            [0, 0, 0, 1.0000],
        ]
    )

    # Measure transformation time
    start_time = time.time()

    # Transform points using fast vectorized method
    transformed_pc = transform_points_fast(original_pc, transform_matrix)

    transformation_time = time.time() - start_time
    print(
        f"Transformed {len(original_pc)} points in {1000 * transformation_time:.2f} ms"
    )

    # Filter points by radius (e.g., keep points within 5 meters from origin)
    max_radius = 3.5  # Maximum radius in meters - adjust as needed
    filtered_pc = filter_points_by_radius(transformed_pc, max_radius)

    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    # Use VAMP's built-in configuration
    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
    )

    # Create environment
    env = vamp.Environment()

    # Add the filtered and transformed point cloud to the environment
    r_min, r_max = 0.03, 0.08  # Min/max sphere radius for Fetch robot
    point_radius = 0.03  # Default point radius

    # Convert to list for VAMP compatibility - use filtered points for collision
    filtered_pc_list = filtered_pc.tolist()

    print(f"Adding {len(filtered_pc_list)} filtered points to environment...")
    add_start_time = time.time()
    build_time = env.add_pointcloud(filtered_pc_list, r_min, r_max, point_radius)
    add_time = time.time() - add_start_time

    print(f"CAPT construction time: {build_time * 1e-6:.3f}ms")
    print(f"Point cloud add time: {add_time:.3f}s")

    # Define start and goal configurations
    start_joints = [0.3, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]
    goal_joints = [0.37, 0.80, -0.40, -1.5, 1.5, 1.0, -0.0, 2.169129759130249]

    # Initialize the sampler
    sampler = vamp_module.halton()
    sampler.skip(0)

    # Try to plan path
    print("Planning motion...")
    result = planner_func(start_joints, goal_joints, env, plan_settings, sampler)

    if result.solved:
        print("Solved problem!")
        simplify = vamp_module.simplify(result.path, env, simp_settings, sampler)

        # Print statistics
        stats = vamp.results_to_dict(result, simplify)
        print(
            f"""
Planning Time: {stats['planning_time'].microseconds:8d}μs
Simplify Time: {stats['simplification_time'].microseconds:8d}μs
   Total Time: {stats['total_time'].microseconds:8d}μs

Planning Iters: {stats['planning_iterations']}
n Graph States: {stats['planning_graph_size']}

Path Length:
   Initial: {stats['initial_path_cost']:5.3f}
Simplified: {stats['simplified_path_cost']:5.3f}"""
        )

        plan = simplify.path
        plan.interpolate(vamp_module.resolution())
    else:
        print("Failed to solve problem! Displaying start and goals.")
        print(
            f"""
Planning Time: {int(result.nanoseconds / 1000):8d}μs
Planning Iters: {result.iterations}
n Graph States: {result.size}
"""
        )

        plan = vamp_module.Path()
        plan.append(vamp_module.Configuration(start_joints))
        plan.append(vamp_module.Configuration(goal_joints))

    # Create simulator using VAMP's configuration
    robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
    sim = vpb.PyBulletSimulator(
        str(robot_dir / "fetch_spherized.urdf"),
        vamp.ROBOT_JOINTS["fetch"],
        True,
    )

    # Sample a subset of points for visualization to avoid PyBullet limits
    # NOTE: We still use filtered points for collision checking, just sampling for viz
    viz_points = sample_points_for_visualization(filtered_pc, max_points=75000)

    # Show the environment - use the sampled points for visualization
    print(f"Visualizing {len(viz_points)} sampled points in PyBullet")
    sim.draw_pointcloud(viz_points)

    # Animate path
    sim.animate(plan)

    print("Visualization running. Press Ctrl+C to exit...")
    while True:
        pass


if __name__ == "__main__":
    main()
