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
import matplotlib.pyplot as plt
import fire


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


def visualize_astar_path(
    base_path, obstacles=None, start=None, goal=None, title="A* Path Planning Result"
):
    """
    Visualize the A* path planning result.

    Args:
        base_path: List of base configurations (x, y, theta)
        obstacles: List of obstacle points (x, y)
        start: Start configuration (x, y, theta)
        goal: Goal configuration (x, y, theta)
        title: Plot title
    """
    plt.figure(figsize=(10, 10))

    # Extract x, y coordinates from the path
    path_x = [config[0] for config in base_path]
    path_y = [config[1] for config in base_path]

    # Plot the path
    plt.plot(path_x, path_y, "b-", linewidth=2, label="A* Path")

    # Plot obstacles if provided
    if obstacles is not None:
        # Downsample obstacles for plotting if there are too many
        if len(obstacles) > 10000:
            indices = np.random.choice(len(obstacles), size=10000, replace=False)
            obstacles = obstacles[indices]

        obstacle_x = [p[0] for p in obstacles]
        obstacle_y = [p[1] for p in obstacles]
        plt.scatter(obstacle_x, obstacle_y, c="gray", s=1, alpha=0.3, label="Obstacles")

    # Plot start and goal positions with orientation arrows
    if start is not None:
        plt.scatter(start[0], start[1], c="g", s=100, marker="o", label="Start")
        # Add orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(start[2])
        dy = arrow_length * np.sin(start[2])
        plt.arrow(
            start[0], start[1], dx, dy, head_width=0.1, head_length=0.1, fc="g", ec="g"
        )

    if goal is not None:
        plt.scatter(goal[0], goal[1], c="r", s=100, marker="o", label="Goal")
        # Add orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(goal[2])
        dy = arrow_length * np.sin(goal[2])
        plt.arrow(
            goal[0], goal[1], dx, dy, head_width=0.1, head_length=0.1, fc="r", ec="r"
        )

    # Set plot properties
    plt.grid(True)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    # Save the plot
    plt.savefig("astar_path_result.png")
    print("Saved visualization to 'astar_path_result.png'")

    # Show the plot
    plt.show()


def main(
    pc_directory: str = "mp_collision_models",
    visualize: bool = True,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_theta: float = 0.0,
    # goal_x: float = -4.296,
    # goal_y: float = -2.211,
    # goal_theta: float = -2.22,
    goal_x: float = -3.70515,
    goal_y: float = -2.3,
    goal_theta: float = -1.423,
    point_radius: float = 0.03,
    max_radius: float = 20.0,
    **kwargs,
):
    """Test the A* algorithm for base path planning with point cloud environments."""
    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    # start_base = [-0.522, -1.571, -1.486]  # x, y, theta
    # goal_base = [2.06, -2.71, 0.227]  # x, y, theta

    start_x, start_y, start_theta = -0.522, -1.571, -1.486
    goal_x, goal_y, goal_theta = 2.06, -2.71, 0.227

    # Use VAMP's built-in configuration
    (vamp_module, _, _, _) = vamp.configure_robot_and_planner_with_kwargs(
        robot, planner
    )

    # Print the current base parameters
    print(
        f"Default base parameters: theta={vamp_module.get_base_theta():.6f}, x={vamp_module.get_base_x():.6f}, y={vamp_module.get_base_y():.6f}"
    )

    # Create environment
    env = vamp.Environment()

    # Load point clouds from directory
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print(f"Failed to load point clouds from {pc_directory}. Exiting.")
        return

    # Filter points by radius
    filtered_pc = filter_points_by_radius(original_pc, max_radius)

    # Convert to list for VAMP compatibility
    filtered_pc_list = filtered_pc.tolist()

    # Add the filtered point cloud to the environment
    r_min, r_max = vamp.ROBOT_RADII_RANGES[robot]

    print(f"Adding {len(filtered_pc_list)} filtered points to environment...")
    add_start_time = time.time()
    build_time = env.add_pointcloud(filtered_pc_list, r_min, r_max, point_radius)
    add_time = time.time() - add_start_time

    print(f"CAPT construction time: {build_time * 1e-6:.3f}ms")
    print(f"Point cloud add time: {add_time:.3f}s")

    # Define start and goal base configurations for A* testing
    start_base = [start_x, start_y, start_theta]  # x, y, theta
    goal_base = [goal_x, goal_y, goal_theta]  # x, y, theta

    vamp_module.set_base_params(1.423, -2.80515, 0.03805)
    base_theta = vamp_module.get_base_theta()
    print("Base theta: ", base_theta)
    if vamp_module.validate([0.3, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0], env):
        print("Start configuration valid")
    else:
        print("Start configuration not valid")

    # Create BaseConfiguration objects
    start_base_config = vamp_module.MobileBaseConfiguration(start_base)
    goal_base_config = vamp_module.MobileBaseConfiguration(goal_base)

    # Create HybridAStarConfig
    hybrid_config = vamp_module.HybridAStarConfig()

    # Print hybrid A* configuration
    print("Hybrid A* Configuration:")
    print(f"  Cell Size: {hybrid_config.cell_size}")
    print(f"  Heading Resolution: {hybrid_config.heading_resolution}")

    # Create a vector to store the resulting path
    base_path = []

    # Call the A* planner
    print(f"Planning base path from {start_base} to {goal_base}...")
    plan_start_time = time.time()

    # Access the HybridAStar planner from the MultilayerRRTC class
    success = vamp_module.HybridAStar.plan(
        start_base_config, goal_base_config, env, hybrid_config, base_path
    )

    plan_time = time.time() - plan_start_time

    if success:
        print(f"A* planning succeeded in {plan_time:.3f}s!")
        print(f"Path length: {len(base_path)} waypoints")

        # Extract base configurations from the path
        base_configs = []
        for config in base_path:
            base_configs.append(config.config)

        # Print the first few and last few waypoints
        print("Path waypoints (first 5):")
        for i, config in enumerate(base_configs[:5]):
            print(f"  {i}: x={config[0]:.3f}, y={config[1]:.3f}, theta={config[2]:.3f}")

        if len(base_configs) > 10:
            print("...")

        print("Path waypoints (last 5):")
        for i, config in enumerate(base_configs[-5:]):
            print(
                f"  {len(base_configs)-5+i}: x={config[0]:.3f}, y={config[1]:.3f}, theta={config[2]:.3f}"
            )

        # Visualize the path
        if visualize:
            visualize_astar_path(
                base_configs,
                obstacles=filtered_pc[:, :2],  # Use only x, y coordinates
                start=start_base,
                goal=goal_base,
                title=f"A* Path Planning Result ({len(base_configs)} waypoints)",
            )

            # Create simulator using VAMP's configuration
            robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
            sim = vpb.PyBulletSimulator(
                str(robot_dir / "fetch_spherized.urdf"),
                vamp.ROBOT_JOINTS["fetch"],
                True,
            )

            # Sample a subset of points for visualization to avoid PyBullet limits
            viz_points = sample_points_for_visualization(filtered_pc, max_points=75000)

            # Show the environment - use the sampled points for visualization
            print(f"Visualizing {len(viz_points)} sampled points in PyBullet")
            sim.draw_pointcloud(viz_points)

            # Visualize the path in PyBullet
            print("Visualizing path in PyBullet...")

            # Default arm configuration for visualization
            default_arm_config = [0.3, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]

            # Animate the robot along the path
            for i, base_config in enumerate(base_configs):
                print(f"Waypoint {i}/{len(base_configs)}")

                # Set the base position and orientation
                quat = pb.getQuaternionFromEuler([0, 0, base_config[2]])
                sim.client.resetBasePositionAndOrientation(
                    sim.skel_id,
                    [base_config[0], base_config[1], 0],  # Position (x, y, z)
                    quat,  # Orientation as quaternion
                )

                # Set the arm configuration
                sim.set_joint_positions(default_arm_config)

                # Add a small delay for visualization
                time.sleep(0.1)

            print("A* path visualization complete!")
    else:
        print(f"A* planning failed after {plan_time:.3f}s!")

        if visualize:
            # Create simulator for visualization of the environment
            robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
            sim = vpb.PyBulletSimulator(
                str(robot_dir / "fetch_spherized.urdf"),
                vamp.ROBOT_JOINTS["fetch"],
                True,
            )

            # Sample a subset of points for visualization to avoid PyBullet limits
            viz_points = sample_points_for_visualization(filtered_pc, max_points=75000)

            # Show the environment - use the sampled points for visualization
            print(f"Visualizing {len(viz_points)} sampled points in PyBullet")
            sim.draw_pointcloud(viz_points)

            # Visualize start and goal positions
            visualize_astar_path(
                [],  # Empty path
                obstacles=filtered_pc[:, :2],  # Use only x, y coordinates
                start=start_base,
                goal=goal_base,
                title="A* Path Planning Failed",
            )


if __name__ == "__main__":
    fire.Fire(main)
