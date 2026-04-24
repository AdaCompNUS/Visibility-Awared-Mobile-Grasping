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
import fire
import pandas as pd


def load_pointcloud(pcd_path):
    """Load a point cloud from file using Open3D."""
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        return points
    except Exception as e:
        return None


def load_all_pointclouds(directory):
    """Load all point cloud files from a directory."""
    # Find all point cloud files
    ply_files = glob.glob(os.path.join(directory, "*.ply"))
    pcd_files = glob.glob(os.path.join(directory, "*.pcd"))
    all_files = ply_files + pcd_files

    if not all_files:
        return None

    # Load and combine all point clouds
    combined_points = []
    for file_path in all_files:
        pc = load_pointcloud(file_path)
        if pc is not None and len(pc) > 0:
            combined_points.append(pc)

    if not combined_points:
        return None

    # Combine all point clouds into a single array
    combined_pc = np.vstack(combined_points)
    return combined_pc


def filter_points_by_radius(points, max_radius):
    """Filter out points beyond a specified radius from the origin."""
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= max_radius
    filtered_points = points[mask]
    return filtered_points


def sample_points_for_visualization(points, max_points=100000):
    """Sample a subset of points for visualization."""
    if len(points) <= max_points:
        return points.tolist()
    indices = np.random.choice(len(points), size=max_points, replace=False)
    return points[indices].tolist()


def visualize_static_path(sim, arm_path, base_configs, robot_urdf, num_instances=32):
    """
    Visualize the robot path by drawing multiple robot instances along the path.

    Args:
        sim: PyBullet simulator instance
        arm_path: List of arm configurations
        base_configs: List of base configurations [x, y, theta]
        robot_urdf: Path to the robot URDF file
        num_instances: Number of robot instances to display along the path
    """
    # No longer drawing path lines on the ground
    path_markers = []

    # If we have a valid path, create robot instances along it
    if len(arm_path) > 0 and len(base_configs) > 0:
        # Calculate which waypoints to show
        total_waypoints = len(arm_path)

        # Always include start and end points
        indices = [0]  # Start point

        if num_instances > 2 and total_waypoints > 2:
            # Calculate intermediate waypoints
            step = (total_waypoints - 1) / (num_instances - 1)
            for i in range(1, num_instances - 1):
                idx = min(int(i * step), total_waypoints - 1)
                if idx not in indices:
                    indices.append(idx)

        # Add end point
        if total_waypoints > 1 and (total_waypoints - 1) not in indices:
            indices.append(total_waypoints - 1)

        # Track created robot IDs
        robot_ids = []

        # Create a robot at each selected waypoint
        for idx in indices:
            # Load a new robot model for this waypoint
            robot_id = sim.client.loadURDF(
                robot_urdf,
                [base_configs[idx][0], base_configs[idx][1], 0],
                pb.getQuaternionFromEuler([0, 0, base_configs[idx][2]]),
                useFixedBase=False,
            )
            robot_ids.append(robot_id)

            # Get the arm configuration
            arm_config = arm_path[idx]
            if isinstance(arm_config, list):
                arm_config_list = arm_config
            elif isinstance(arm_config, np.ndarray):
                arm_config_list = arm_config.tolist()
            else:
                arm_config_list = arm_config.to_list()

            # Get joint names and indices mapping for this robot
            joint_name_to_idx = {}
            for j in range(sim.client.getNumJoints(robot_id)):
                joint_info = sim.client.getJointInfo(robot_id, j)
                joint_name = joint_info[1].decode("utf-8")
                joint_name_to_idx[joint_name] = j

            # Set joint positions for this robot using joint names
            for j, joint_name in enumerate(vamp.ROBOT_JOINTS["fetch"]):
                if joint_name in joint_name_to_idx:
                    joint_idx = joint_name_to_idx[joint_name]
                    sim.client.resetJointState(robot_id, joint_idx, arm_config_list[j])

            # Only apply transparency to intermediate waypoints
            # Start and end keep their original mesh colors
            if idx != 0 and idx != total_waypoints - 1:  # Intermediate only
                alpha = 0.25
                # Very light blue for intermediate robots
                color = [0.75, 0.85, 1.0, alpha]

                # Apply color to all links of the robot
                for link_idx in range(
                    sim.client.getNumJoints(robot_id) + 1
                ):  # +1 for base link
                    sim.client.changeVisualShape(robot_id, link_idx - 1, rgbaColor=color)

        return robot_ids, path_markers

    return [], path_markers


def main(
    pc_directory: str = "mp_collision_models",
    point_radius: float = 0.03,
    max_radius: float = 20.0,
    num_instances: int = 10,
    **kwargs,
):
    """Test whole body motion planning with point cloud environments."""
    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    # Use VAMP's built-in configuration
    (vamp_module, _, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
    )

    # Define start and goal configurations (from test_whole_body.py)
    start_base = [0, 0, -1.57]  # x, y, theta
    goal_base = [-2.689, 0.752, 1.288]  # x, y, theta

    start_joints = [0.296, 1.325, 1.398, -0.229, 1.717, -0.001, 1.66, 0.008]
    goal_joints = [0.344, -0.999, -0.325, 1.476, 1.679, -0.624, 1.165, -1.237]

    # Create environment
    env = vamp.Environment()

    # Load point clouds from directory
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print(f"Failed to load point clouds from {pc_directory}. Exiting.")
        return

    # Filter points by radius
    filtered_pc = filter_points_by_radius(original_pc, max_radius)
    filtered_pc_list = filtered_pc.tolist()

    # Add the filtered point cloud to the environment
    r_min, r_max = vamp.ROBOT_RADII_RANGES[robot]
    env.add_pointcloud(filtered_pc_list, r_min, r_max, point_radius)

    # Initialize the sampler
    sampler = vamp_module.halton()
    sampler.skip(0)

    # Plan path
    print("Planning whole body motion...")

    # Use multilayer_rrtc for planning
    result = vamp_module.multilayer_rrtc(
        start_joints,  # arm start config
        goal_joints,  # arm goal config
        start_base,  # base start config
        goal_base,  # base goal config
        env,
        plan_settings,
        sampler,
    )

    if result.is_successful():
        print("Path planning successful!")

        # Get the arm path from the multilayer planning result
        arm_path = result.arm_result.path

        # Get the base path from the result
        base_path = result.base_path
        
        # Print the actual goal base position from the planner
        if len(base_path) > 0:
            actual_goal_base = base_path[-1].config
            print(f"Actual goal base position from planner: [{actual_goal_base[0]:.3f}, {actual_goal_base[1]:.3f}, {actual_goal_base[2]:.3f}]")
            print(f"Requested goal base position: [{goal_base[0]:.3f}, {goal_base[1]:.3f}, {goal_base[2]:.3f}]")

        # Convert arm path to a list of lists for whole_body_simplify
        arm_path_list = []
        for config in arm_path:
            arm_path_list.append(config.to_list())

        # Simplify the path - pass base_path directly, not converted to lists
        whole_body_result = vamp_module.whole_body_simplify(
            arm_path_list, base_path, env, simp_settings, sampler
        )

        # Interpolate both arm and base paths together using density
        density = 0.02
        whole_body_result.interpolate(density)

        # Get the interpolated paths
        arm_path = whole_body_result.arm_result.path
        base_path = whole_body_result.base_path

        # Extract base configurations as lists
        base_configs = []
        for config in base_path:
            base_configs.append(config.config)
    else:
        print("Failed to solve planning problem!")
        # Create empty paths
        arm_path = vamp_module.Path()
        base_configs = []

    # Create simulator using VAMP's configuration
    robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
    robot_urdf = str(robot_dir / "fetch_spherized.urdf")

    # Create the main simulator instance for visualization (white background, no shadows)
    sim = vpb.PyBulletSimulator(
        robot_urdf,
        vamp.ROBOT_JOINTS["fetch"],
        True,  # Enable GUI
    )

    # Ensure shadows are disabled (no sky/light/shadow look)
    try:
        sim.client.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
    except Exception:
        pass

    # Remove the default robot from the scene
    sim.client.removeBody(sim.skel_id)

    # Sample a subset of points for visualization
    viz_points = sample_points_for_visualization(filtered_pc, max_points=70000)

    # Show the environment - use the sampled points for visualization
    sim.draw_pointcloud(viz_points)

    # Create static path visualization with multiple robot instances
    if result.is_successful():
        print(f"Visualizing path with {num_instances} robot instances...")

        # Create multiple robot instances along the path
        robot_ids, path_markers = visualize_static_path(
            sim, arm_path, base_configs, robot_urdf, num_instances
        )

        print(f"Created {len(robot_ids)} robot instances. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Visualization stopped by user.")
    else:
        # Just show start and goal configurations
        print("Displaying start and goal configurations (no path found)...")

        # Load a robot model for the start configuration
        start_robot_id = sim.client.loadURDF(
            robot_urdf,
            [start_base[0], start_base[1], 0],
            pb.getQuaternionFromEuler([0, 0, start_base[2]]),
            useFixedBase=False,
        )

        # Get joint names and indices mapping for start robot
        start_joint_name_to_idx = {}
        for j in range(sim.client.getNumJoints(start_robot_id)):
            joint_info = sim.client.getJointInfo(start_robot_id, j)
            joint_name = joint_info[1].decode("utf-8")
            start_joint_name_to_idx[joint_name] = j

        # Set start joint positions
        for j, joint_name in enumerate(vamp.ROBOT_JOINTS["fetch"]):
            if joint_name in start_joint_name_to_idx:
                joint_idx = start_joint_name_to_idx[joint_name]
                sim.client.resetJointState(start_robot_id, joint_idx, start_joints[j])

        # Load a separate robot model for the goal configuration
        goal_robot_id = sim.client.loadURDF(
            robot_urdf,
            [goal_base[0], goal_base[1], 0],
            pb.getQuaternionFromEuler([0, 0, goal_base[2]]),
            useFixedBase=False,
        )

        # Get joint names and indices mapping for goal robot
        joint_name_to_idx = {}
        for j in range(sim.client.getNumJoints(goal_robot_id)):
            joint_info = sim.client.getJointInfo(goal_robot_id, j)
            joint_name = joint_info[1].decode("utf-8")
            joint_name_to_idx[joint_name] = j

        # Set goal joint configuration using joint names
        for j, joint_name in enumerate(vamp.ROBOT_JOINTS["fetch"]):
            if joint_name in joint_name_to_idx:
                joint_idx = joint_name_to_idx[joint_name]
                sim.client.resetJointState(goal_robot_id, joint_idx, goal_joints[j])

        # Keep original mesh colors for both start and goal

        print("Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Visualization stopped by user.")


if __name__ == "__main__":
    fire.Fire(main)
