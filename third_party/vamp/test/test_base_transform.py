#!/usr/bin/env python3
import numpy as np
import vamp
from vamp import pybullet_interface as vpb
from pathlib import Path
import open3d as o3d
import os
import glob
import time
import pybullet as pb


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

    # Use random sampling
    indices = np.random.choice(len(points), size=max_points, replace=False)
    sampled_points = points[indices]

    print(
        f"Sampled {len(sampled_points)} points for visualization (from {len(points)} total)"
    )
    return sampled_points.tolist()


def main():
    """Test script for verifying point cloud collision checking using VAMP's built-in functions."""
    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    # Use VAMP's built-in configuration
    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
    )

    # Set the base parameters for the Fetch robot
    # These were previously hardcoded in C++ as:
    # theta = -1.423f, x = -2.80515f, y = 0.03805f
    #     [INFO] [1754372455.620573]: Set base parameters: theta=-3.130230, x=-1.524094, y=1.700755
    # [INFO] [1754372455.625890]: Planning with VAMP (8-DOF):
    # [INFO] [1754372455.630961]: Start config values: [ 0.33914411 -0.98195327 -0.97468657  0.74365499  1.9173444   2.94919102
    #  -0.86906404 -0.81868091]
    # [INFO] [1754372455.634787]: Goal config values: [ 0.35       -1.48701633 -0.64027876  1.07882972  1.56798012  2.32140891
    #  -0.99906892  3.01588812]

    base_theta = -3.130230
    base_x = -1.524094
    base_y = 1.700755

    # Print the current base parameters
    print(
        f"Default base parameters: theta={vamp_module.get_base_theta():.6f}, x={vamp_module.get_base_x():.6f}, y={vamp_module.get_base_y():.6f}"
    )

    # Set the base parameters
    vamp_module.set_base_params(base_theta, base_x, base_y)
    print(
        f"Set base parameters: theta={base_theta:.6f}, x={base_x:.6f}, y={base_y:.6f}"
    )

    # Create environment
    env = vamp.Environment()

    # Load point clouds from directory
    pc_directory = "mp_collision_models"
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print("Failed to load point clouds. Exiting.")
        return

    # Filter points by radius
    max_radius = 1000  # Maximum radius in meters
    filtered_pc = filter_points_by_radius(original_pc, max_radius)

    # Convert to list for VAMP compatibility
    filtered_pc_list = filtered_pc.tolist()

    # Add the filtered point cloud to the environment
    r_min, r_max = 0.03, 0.08  # Min/max sphere radius for Fetch robot
    point_radius = 0.03  # Default point radius

    print(f"Adding {len(filtered_pc_list)} filtered points to environment...")
    add_start_time = time.time()
    build_time = env.add_pointcloud(filtered_pc_list, r_min, r_max, point_radius)
    add_time = time.time() - add_start_time

    print(f"CAPT construction time: {build_time * 1e-6:.3f}ms")
    print(f"Point cloud add time: {add_time:.3f}s")

    # Define start and goal configurations - using the same ones from paste-2.txt
    start_joints = [
        0.33914411,
        -0.98195327,
        -0.97468657,
        0.74365499,
        1.9173444,
        2.94919102,
        -0.86906404,
        -0.81868091,
    ]
    goal_joints = [
        0.35,
        -1.48701633,
        -0.64027876,
        1.07882972,
        1.56798012,
        2.32140891,
        -0.99906892,
        3.01588812,
    ]

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

    # Set the base position and orientation in PyBullet
    # Convert from VAMP's coordinate system to PyBullet's coordinate system
    # In PyBullet, we need to set the position and orientation of the robot base
    print(f"Setting robot base position to: x={base_x}, y={base_y}, theta={base_theta}")

    # Get the robot ID (usually 0 for the first loaded robot)
    robot_id = sim.skel_id

    # Convert theta to quaternion (rotation around Z axis)
    # PyBullet uses quaternions for orientation
    quat = pb.getQuaternionFromEuler([0, 0, base_theta])

    # Set the base position and orientation
    sim.client.resetBasePositionAndOrientation(
        robot_id,
        [base_x, base_y, 0],  # Position (x, y, z)
        quat,  # Orientation as quaternion
    )

    # Sample a subset of points for visualization to avoid PyBullet limits
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
