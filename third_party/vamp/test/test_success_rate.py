#!/usr/bin/env python3
import numpy as np
import vamp
from vamp import pybullet_interface as vpb
from pathlib import Path
import time
import pybullet as pb
import fire
import pandas as pd
import random
import os
import glob


def load_pointcloud(pcd_path):
    """Load a point cloud from file using Open3D."""
    print(f"Loading point cloud from {pcd_path}...")

    try:
        import open3d as o3d

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


def sample_random_base_position():
    """
    Sample a random base position within the specified range.

    Returns:
        A list [x, y, theta] for the base position and orientation
    """
    x = random.uniform(-4.0, 4.0)
    y = random.uniform(-2.0, 2.0)
    theta = random.uniform(-np.pi, np.pi)

    return [x, y, theta]


def sample_random_joint_configuration():
    """
    Sample a random joint configuration for Fetch's 8 joints in motion planning.

    Fetch joint limits (approximate):
    - Joint 0 (torso_lift): [0.0, 0.4]
    - Joint 1 (shoulder_pan): [-1.6056, 1.6056]
    - Joint 2 (shoulder_lift): [-1.221, 1.518]
    - Joint 3 (upperarm_roll): [-3.14, 3.14]
    - Joint 4 (elbow_flex): [-2.251, 2.251]
    - Joint 5 (forearm_roll): [-3.14, 3.14]
    - Joint 6 (wrist_flex): [-2.16, 2.16]
    - Joint 7 (wrist_roll): [-3.14, 3.14]

    Returns:
        A list of 8 joint values for motion planning
    """
    joint_limits = [
        [0.0, 0.4],  # torso_lift
        [-1.6056, 1.6056],  # shoulder_pan
        [-1.221, 1.518],  # shoulder_lift
        [-3.14, 3.14],  # upperarm_roll
        [-2.251, 2.251],  # elbow_flex
        [-3.14, 3.14],  # forearm_roll
        [-2.16, 2.16],  # wrist_flex
        [-3.14, 3.14],  # wrist_roll
    ]

    joints = []
    for limits in joint_limits:
        joints.append(random.uniform(limits[0], limits[1]))

    return joints


def main(
    pc_directory: str = "mp_collision_models",
    point_radius: float = 0.03,
    max_radius: float = 20.0,
    num_samples: int = 100,
    **kwargs,
):
    """
    Test the success rate of motion planning by sampling random start and goal configurations.

    Args:
        pc_directory: Directory containing point cloud files
        point_radius: Radius of points in the point cloud
        max_radius: Maximum radius from origin to consider points
        num_samples: Number of planning attempts to make
    """
    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    # Use VAMP's built-in configuration
    (vamp_module, _, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
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

    # Create simulator for visualization
    robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
    sim = vpb.PyBulletSimulator(
        str(robot_dir / "fetch_spherized.urdf"),
        vamp.ROBOT_JOINTS["fetch"],
        True,
    )

    # Sample a subset of points for visualization
    viz_points = filtered_pc[
        np.random.choice(len(filtered_pc), min(75000, len(filtered_pc)), replace=False)
    ].tolist()
    sim.draw_pointcloud(viz_points)

    # Initialize statistics
    stats = {
        "total_attempts": 0,
        "valid_configs": 0,
        "planning_successes": 0,
        "planning_failures": 0,
        "planning_times": [],
    }

    try:
        # Run multiple tests
        for test_num in range(num_samples):
            print(f"\n=== Test {test_num + 1}/{num_samples} ===")

            # Keep sampling until we get valid start and goal configurations
            valid_config = False
            max_sampling_attempts = 50
            sampling_attempts = 0

            while not valid_config and sampling_attempts < max_sampling_attempts:
                sampling_attempts += 1

                # Sample random start and goal configurations
                start_base = sample_random_base_position()
                goal_base = sample_random_base_position()
                start_joints = sample_random_joint_configuration()
                goal_joints = sample_random_joint_configuration()

                # Set base parameters
                vamp_module.set_base_params(start_base[2], start_base[0], start_base[1])
                vamp_module.set_base_params(goal_base[2], goal_base[0], goal_base[1])

                # Consider all configurations valid
                valid_config = True
                stats["valid_configs"] += 1
                print(
                    f"Using sampled start and goal configurations (attempt {sampling_attempts})"
                )
                print(f"Start: base={start_base}, joints={start_joints}")
                print(f"Goal: base={goal_base}, joints={goal_joints}")

            if not valid_config:
                print(
                    f"Failed to find valid configurations after {max_sampling_attempts} attempts. Skipping this test."
                )
                continue

            stats["total_attempts"] += 1

            # Initialize the sampler
            sampler = vamp_module.halton()
            sampler.skip(0)

            # Plan path
            print("Planning whole body motion...")

            plan_start_time = time.time()
            result = vamp_module.multilayer_rrtc(
                start_joints,
                goal_joints,
                start_base,
                goal_base,
                env,
                plan_settings,
                sampler,
            )
            plan_time = time.time() - plan_start_time
            stats["planning_times"].append(plan_time)

            planning_successful = result.is_successful()

            if planning_successful:
                stats["planning_successes"] += 1
                print(f"Planning SUCCESSFUL in {plan_time:.2f} seconds")

                # Get the arm path from the multilayer planning result
                arm_path = result.arm_result.path

                # Get the base path from the result
                base_path = result.base_path
                base_configs = []
                for config in base_path:
                    base_configs.append(config.config)

                # Convert base_configs to the expected format for whole_body_simplify
                base_path_list = []
                for config in base_configs:
                    base_path_list.append(list(config))

                # Convert arm path to a list of lists for whole_body_simplify
                arm_path_list = []
                for config in arm_path:
                    arm_path_list.append(config.to_list())

                # Simplify the paths
                whole_body_result = vamp_module.whole_body_simplify(
                    arm_path_list, base_path_list, env, simp_settings, sampler
                )

                # Interpolate both arm and base paths together
                interpolation_resolution = 64
                whole_body_result.interpolate(interpolation_resolution)

                # Get the interpolated paths
                arm_path = whole_body_result.arm_result.path
                base_path = whole_body_result.base_path

                # Extract base configurations as lists
                base_configs = []
                for config in base_path:
                    base_configs.append(config.config)

                # Visualize the path
                print("Visualizing whole-body motion...")

                # Show start configuration
                quat_start = pb.getQuaternionFromEuler([0, 0, start_base[2]])
                sim.client.resetBasePositionAndOrientation(
                    sim.skel_id,
                    [start_base[0], start_base[1], 0],
                    quat_start,
                )
                sim.set_joint_positions(start_joints)
                print("Showing start configuration... Press Enter to animate path...")
                input()

                # Animate the path
                arm_path_len = len(arm_path)
                for i in range(arm_path_len):
                    # Get arm and base configurations for this step
                    arm_config = (
                        arm_path[i].to_list()
                        if hasattr(arm_path[i], "to_list")
                        else arm_path[i]
                    )
                    base_config = base_configs[i]

                    # Set the base position and orientation
                    quat = pb.getQuaternionFromEuler([0, 0, base_config[2]])
                    sim.client.resetBasePositionAndOrientation(
                        sim.skel_id,
                        [base_config[0], base_config[1], 0],
                        quat,
                    )

                    # Set the arm configuration
                    sim.set_joint_positions(arm_config)

                    # Add a small delay for visualization
                    time.sleep(0.02)

                # Goal configuration is implicitly shown at the end of the animation
                print("Path animation complete. Goal configuration shown.")

            else:
                stats["planning_failures"] += 1
                print(f"Planning FAILED after {plan_time:.2f} seconds")

                # Just show start and goal configurations
                print("Displaying start configuration...")
                quat_start = pb.getQuaternionFromEuler([0, 0, start_base[2]])
                sim.client.resetBasePositionAndOrientation(
                    sim.skel_id,
                    [start_base[0], start_base[1], 0],
                    quat_start,
                )
                sim.set_joint_positions(start_joints)
                print("Press Enter to show goal configuration...")
                input()

                print("Displaying goal configuration...")
                quat_goal = pb.getQuaternionFromEuler([0, 0, goal_base[2]])
                sim.client.resetBasePositionAndOrientation(
                    sim.skel_id,
                    [goal_base[0], goal_base[1], 0],
                    quat_goal,
                )
                sim.set_joint_positions(goal_joints)
                print("Goal configuration shown.")

            # Calculate and show current success rate
            current_success_rate = (
                stats["planning_successes"] / stats["total_attempts"]
            ) * 100
            print(
                f"Current success rate: {current_success_rate:.2f}% ({stats['planning_successes']}/{stats['total_attempts']})"
            )

            # Wait for user input before starting the next sample
            print("\nPress Enter to continue to the next sample...")
            input()

        # Calculate final statistics
        # Ensure total_attempts is not zero before calculating rates
        if stats["total_attempts"] > 0:
            success_rate = (stats["planning_successes"] / stats["total_attempts"]) * 100
            avg_planning_time = (
                sum(stats["planning_times"]) / len(stats["planning_times"])
                if stats["planning_times"]
                else 0
            )
        else:
            success_rate = 0.0
            avg_planning_time = 0.0

        print("\n=== Final Statistics ===")
        print(f"Total valid configuration pairs: {stats['valid_configs']}")
        print(f"Total planning attempts: {stats['total_attempts']}")
        print(f"Planning successes: {stats['planning_successes']}")
        print(f"Planning failures: {stats['planning_failures']}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Average planning time: {avg_planning_time:.3f} seconds")

        # Export statistics to CSV
        results_df = pd.DataFrame(
            {
                "Valid Configuration Pairs": [stats["valid_configs"]],
                "Total Planning Attempts": [stats["total_attempts"]],
                "Planning Successes": [stats["planning_successes"]],
                "Planning Failures": [stats["planning_failures"]],
                "Success Rate (%)": [success_rate],
                "Average Planning Time (s)": [avg_planning_time],
            }
        )

        results_df.to_csv("planning_success_rates.csv", index=False)
        print("Results saved to planning_success_rates.csv")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

        # Calculate statistics for completed tests
        if stats["total_attempts"] > 0:
            success_rate = (stats["planning_successes"] / stats["total_attempts"]) * 100
            avg_planning_time = (
                sum(stats["planning_times"]) / len(stats["planning_times"])
                if stats["planning_times"]
                else 0
            )

            print("\n=== Partial Statistics ===")
            print(f"Total valid configuration pairs: {stats['valid_configs']}")
            print(f"Total planning attempts: {stats['total_attempts']}")
            print(f"Planning successes: {stats['planning_successes']}")
            print(f"Planning failures: {stats['planning_failures']}")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Average planning time: {avg_planning_time:.3f} seconds")

    print("Testing complete.")


if __name__ == "__main__":
    fire.Fire(main)
