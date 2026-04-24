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
from scipy.spatial.transform import Rotation as R


def transform_pose_to_world(base_pos, base_yaw, ee_pos, ee_quat):
    """
    Transform end effector pose from robot base frame to world frame.

    Args:
        base_pos: [x, y, z] position of the robot base in world frame
        base_yaw: Yaw angle of the robot base in world frame
        ee_pos: [x, y, z] position of the end effector in robot base frame
        ee_quat: [x, y, z, w] quaternion of the end effector in robot base frame

    Returns:
        world_pos: [x, y, z] position of the end effector in world frame
        world_quat: [x, y, z, w] quaternion of the end effector in world frame
    """
    # Create base rotation matrix from yaw angle
    base_rot = R.from_euler("z", base_yaw)
    base_rot_matrix = base_rot.as_matrix()

    # Transform end effector position to world frame
    ee_pos_rotated = base_rot_matrix @ np.array(ee_pos)
    world_pos = np.array(
        [
            base_pos[0] + ee_pos_rotated[0],
            base_pos[1] + ee_pos_rotated[1],
            ee_pos_rotated[2],  # Assuming base_pos[2] = 0 (z is up)
        ]
    )

    # Transform end effector orientation to world frame
    ee_rot = R.from_quat([ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]])
    base_rot_quat = base_rot.as_quat()  # scipy uses xyzw format
    base_rot = R.from_quat(base_rot_quat)
    world_rot = base_rot * ee_rot
    world_quat = world_rot.as_quat()  # xyzw format

    return world_pos, world_quat


def print_ee_poses(vamp_module, start_joints, goal_joints, start_base, goal_base):
    """
    Print the end effector poses for start and goal configurations in both robot and world frames.

    Args:
        vamp_module: The VAMP robot module
        start_joints: Start arm joint configuration
        goal_joints: Goal arm joint configuration
        start_base: Start base pose [x, y, yaw]
        goal_base: Goal base pose [x, y, yaw]
    """
    print("\n===== End Effector Pose Analysis =====")

    # Get end effector pose for start configuration
    print("\nStart configuration:")
    print(f"Arm joints: {start_joints}")
    print(f"Base pose: {start_base}")

    # Call the eefk function to get the end effector pose in robot frame
    start_ee_pos, start_ee_quat = vamp_module.eefk(start_joints)

    print("\nStart end effector pose (robot frame):")
    print(f"Position: {start_ee_pos}")
    print(f"Orientation (xyzw quaternion): {start_ee_quat}")

    # Transform to world frame
    start_world_pos, start_world_quat = transform_pose_to_world(
        [start_base[0], start_base[1], 0], start_base[2], start_ee_pos, start_ee_quat
    )

    print("\nStart end effector pose (world frame):")
    print(f"Position: {start_world_pos}")
    print(f"Orientation (xyzw quaternion): {start_world_quat}")

    # Get end effector pose for goal configuration
    print("\nGoal configuration:")
    print(f"Arm joints: {goal_joints}")
    print(f"Base pose: {goal_base}")

    # Call the eefk function to get the end effector pose in robot frame
    goal_ee_pos, goal_ee_quat = vamp_module.eefk(goal_joints)

    print("\nGoal end effector pose (robot frame):")
    print(f"Position: {goal_ee_pos}")
    print(f"Orientation (xyzw quaternion): {goal_ee_quat}")

    # Transform to world frame
    goal_world_pos, goal_world_quat = transform_pose_to_world(
        [goal_base[0], goal_base[1], 0], goal_base[2], goal_ee_pos, goal_ee_quat
    )

    print("\nGoal end effector pose (world frame):")
    print(f"Position: {goal_world_pos}")
    print(f"Orientation (xyzw quaternion): {goal_world_quat}")

    # Print the difference in positions
    pos_diff = goal_world_pos - start_world_pos
    pos_dist = np.linalg.norm(pos_diff)

    print("\nDifference between start and goal (world frame):")
    print(f"Position difference: {pos_diff}")
    print(f"Euclidean distance: {pos_dist:.3f} meters")

    # Convert quaternions to Euler angles for easier interpretation
    start_euler = R.from_quat(start_world_quat).as_euler("xyz", degrees=True)
    goal_euler = R.from_quat(goal_world_quat).as_euler("xyz", degrees=True)

    print("\nEuler angles (xyz, degrees):")
    print(f"Start: {start_euler}")
    print(f"Goal: {goal_euler}")
    print(f"Difference: {goal_euler - start_euler}")

    print("\n========================================")


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


def main(
    pc_directory: str = "mp_collision_models",
    point_radius: float = 0.03,
    max_radius: float = 20.0,
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

# [INFO] [1759806515.208940, 82.879000]: Start base config: [-1.039, 1.61, 3.034]
# [INFO] [1759806515.210304, 82.880000]: Goal base config: [-2.689, 0.762, 1.559]
# [INFO] [1759806515.205988, 82.876000]: Start arm config: [0.296, 1.325, 1.398, -0.229, 1.717, -0.001, 1.66, 0.008]
# [INFO] [1759806515.207771, 82.878000]: Goal arm config: [0.344, -0.999, -0.325, 1.476, 1.679, -0.624, 1.165, -1.237]

    # Define start and goal base configurations (from test_astar.py)
    # start_base = [-1.653, 0.592, 2.85]  # x, y, theta
    start_base = [-2.689, 0.762, 1.108]  # x, y, theta
    # goal_base = [-1.679, 1.322, 1.724]  # x, y, theta
    goal_base = [-1.039, 1.61, 3.034]  # x, y, theta

    # Define start and goal arm configurations (from test_base_transform.py)
    start_joints = [0.296, 1.325, 1.398, -0.229, 1.717, -0.001, 1.66, 0.008]  # x, y, theta
    # goal_joints = [0.287, 1.325, 1.398, -0.226, 1.718, -0.001, 1.66, 0.008]  # x, y, theta
    goal_joints = [0.344, -0.999, -0.325, 1.476, 1.679, -0.624, 1.165, -1.237]

    # Print end effector poses for start and goal configurations
    print_ee_poses(vamp_module, start_joints, goal_joints, start_base, goal_base)

    # Create environment
    env = vamp.Environment()

    if vamp_module.validate(start_joints, env):
        print("Start configuration valid")
    else:
        print("Start configuration not valid")

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

    # Initialize the sampler
    sampler = vamp_module.halton()
    sampler.skip(0)

    # Try to plan path
    print("Planning whole body motion...")
    print(f"Start config: base={start_base}, joints={start_joints}")
    print(f"Goal config: base={goal_base}, joints={goal_joints}")

    # Use multilayer_rrtc for planning
    result = vamp_module.multilayer_rrtc(
        start_joints,  # arm start config array
        goal_joints,  # arm goal config array
        start_base,  # base start config array
        goal_base,  # base goal config array
        env,
        plan_settings,
        sampler,
    )

    if result.is_successful():
        print("Solved problem!")

        # Get the arm path from the multilayer planning result
        arm_path = result.arm_result.path

        # Get the base path from the result
        base_path = result.base_path

        # Convert arm path to a list of lists for whole_body_simplify
        arm_path_list = []
        for config in arm_path:
            arm_path_list.append(config.to_list())

        print(
            f"Using whole_body_simplify with {len(arm_path_list)} arm configurations and {len(base_path)} base configurations"
        )

        # Check for collisions along the initial planned path
        print("\n--- Collision Check: Initial Planned Path ---")
        base_configs_list = []
        for config in base_path:
            base_configs_list.append(config.config)

        print(
            f"Checking collisions for {len(arm_path_list)} arm configs and {len(base_configs_list)} base configs..."
        )
        initial_collision_found = vamp_module.check_whole_body_collisions(
            env, arm_path_list, base_configs_list
        )

        if initial_collision_found:
            print("COLLISION DETECTED in initial planned path!")
            print(
                "Check the output above for 'VAMP_DEBUG' messages to see collision details."
            )
        else:
            print("NO collisions detected in initial planned path.")
        print("--- End Initial Path Collision Check ---\n")

        # Use whole_body_simplify instead of simplify
        whole_body_result = vamp_module.whole_body_simplify(
            arm_path_list, base_path, env, simp_settings, sampler
        )

        # Check for collisions along the simplified path
        print("\n--- Collision Check: Simplified Path ---")
        simplified_arm_path_list = []
        for config in whole_body_result.arm_result.path:
            if isinstance(config, list):
                simplified_arm_path_list.append(config)
            elif isinstance(config, np.ndarray):
                simplified_arm_path_list.append(config.tolist())
            else:
                simplified_arm_path_list.append(config.to_list())

        simplified_base_configs_list = []
        for config in whole_body_result.base_path:
            simplified_base_configs_list.append(config.config)

        print(
            f"Checking collisions for {len(simplified_arm_path_list)} simplified arm configs and {len(simplified_base_configs_list)} simplified base configs..."
        )
        simplified_collision_found = vamp_module.check_whole_body_collisions(
            env, simplified_arm_path_list, simplified_base_configs_list
        )

        if simplified_collision_found:
            print("COLLISION DETECTED in simplified path!")
            print(
                "Check the output above for 'VAMP_DEBUG' messages to see collision details."
            )
        else:
            print("NO collisions detected in simplified path.")
        print("--- End Simplified Path Collision Check ---\n")

        # Print statistics
        # Create a stats dictionary with metrics from both planning and simplification
        stats = vamp.results_to_dict(result.arm_result, whole_body_result.arm_result)

        # Add base planning metrics to stats dictionary
        stats["base_planning_time"] = pd.Timedelta(
            nanoseconds=result.base_result.nanoseconds
        )
        stats["base_planning_iterations"] = result.base_result.iterations

        # Update total planning time to use the overall result nanoseconds (sum of base + arm planning)
        stats["multilayer_rrtc_planning_time"] = pd.Timedelta(
            nanoseconds=result.nanoseconds
        )

        print(
            f"""
RRTC Planning Time: {stats['multilayer_rrtc_planning_time'].microseconds:10d}μs
Base Planning Time: {stats['base_planning_time'].microseconds:10d}μs
Total Planning Time: {stats['planning_time'].microseconds:10d}μs

Planning Iters: {stats['planning_iterations']}
Base Planning Iters: {stats['base_planning_iterations']}
n Graph States: {stats['planning_graph_size']}

Path Length:
   Initial: {stats['initial_path_cost']:5.3f}
Simplified: {stats['simplified_path_cost']:5.3f}"""
        )

        # Verify that arm and base paths have the same length after simplification
        if not whole_body_result.validate_paths():
            print("WARNING: Simplified arm and base paths have different lengths!")
            print(
                f"Arm path: {len(whole_body_result.arm_result.path)}, Base path: {len(whole_body_result.base_path)}"
            )
        else:
            print(
                f"Verified: Arm and base paths both have {len(whole_body_result.arm_result.path)} waypoints"
            )

        # Interpolate both arm and base paths together
        print("Interpolating whole-body path...")
        # interpolation_resolution = 64
        density = 0.02
        # interpolation_resolution = vamp_module.resolution()
        whole_body_result.interpolate(density)
        print(f"Interpolated with density {density}")

        # Get the interpolated paths
        arm_path = whole_body_result.arm_result.path
        base_path = whole_body_result.base_path

        # Extract base configurations as lists
        base_configs = []
        for config in base_path:
            base_configs.append(config.config)

        print(f"Base path length: {len(base_configs)} waypoints")
        print(f"Arm path length: {len(arm_path)} waypoints")

        # Check for collisions along the interpolated path
        print("\n--- Collision Check: Interpolated Path ---")
        interpolated_arm_path_list = []
        for config in arm_path:
            if isinstance(config, list):
                interpolated_arm_path_list.append(config)
            elif isinstance(config, np.ndarray):
                interpolated_arm_path_list.append(config.tolist())
            else:
                interpolated_arm_path_list.append(config.to_list())

        print(
            f"Checking collisions for {len(interpolated_arm_path_list)} interpolated arm configs and {len(base_configs)} interpolated base configs..."
        )
        interpolated_collision_found = vamp_module.check_whole_body_collisions(
            env, interpolated_arm_path_list, base_configs
        )

        if interpolated_collision_found:
            print("COLLISION DETECTED in interpolated path!")
            print(
                "Check the output above for 'VAMP_DEBUG' messages to see collision details."
            )
        else:
            print("NO collisions detected in interpolated path.")
        print("--- End Interpolated Path Collision Check ---\n")
    else:
        print("Failed to solve problem! Displaying start and goals.")

        # Perform a diagnostic check on the first segment of the base path
        # to determine if the collision is from self-collision or environment.
        print("\n--- Diagnostic Check for Layer 0 -> 1 ---")
        if len(result.base_path) >= 2:
            base_config_0 = result.base_path[0].config
            base_config_1 = result.base_path[1].config

            # We use the start arm configuration for the start of the segment.
            # For the end, since the planner failed to find a valid arm config,
            # we'll use the original goal configuration for this test. This helps
            # isolate whether the base motion itself is the problem.
            arm_config_0 = start_joints
            arm_config_1 = goal_joints

            print("Checking for collision between the first two base waypoints...")
            print(f"Base 0: {list(base_config_0)}")
            print(f"Base 1: {list(base_config_1)}")

            # `check_whole_body_collisions` returns true if a collision is found.
            is_collision_found = vamp_module.check_whole_body_collisions(
                env,
                [arm_config_0, arm_config_1],
                [list(base_config_0), list(base_config_1)],
            )

            if is_collision_found:
                print("DIAGNOSTIC: Collision DETECTED between layer 0 and 1.")
                print(
                    "Check the output above for 'VAMP_DEBUG' messages to see if it was self-collision or environment."
                )
            else:
                print("DIAGNOSTIC: NO collision detected between layer 0 and 1.")
        else:
            print("DIAGNOSTIC: Not enough base path waypoints to perform check (< 2).")
        print("--- End of Diagnostic Check ---\n")

        # Create stats dictionary for failure case
        stats = {
            "planning_time": pd.Timedelta(nanoseconds=result.arm_result.nanoseconds),
            "base_planning_time": pd.Timedelta(
                nanoseconds=result.base_result.nanoseconds
            ),
            "total_planning_time": pd.Timedelta(nanoseconds=result.nanoseconds),
            "planning_iterations": result.arm_result.iterations,
            "base_planning_iterations": result.base_result.iterations,
            "planning_graph_size": (
                sum(result.arm_result.size) if result.arm_result.size else 0
            ),
        }

        print(
            f"""
Planning Time: {stats['planning_time'].microseconds:8d}μs
Base Planning Time: {stats['base_planning_time'].microseconds:8d}μs
Total Planning Time: {stats['total_planning_time'].microseconds:8d}μs
Planning Iters: {stats['planning_iterations']}
Base Planning Iters: {stats['base_planning_iterations']}
n Graph States: {stats['planning_graph_size']}
"""
        )

        # Create empty paths
        arm_path = vamp_module.Path()
        base_configs = []

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

    # Animate the whole-body motion (base + arm)
    if result.is_successful():
        print("Visualizing whole-body motion (base + arm)...")

        # After whole-body interpolation, both paths should have the same length
        arm_path_len = len(arm_path)
        base_path_len = len(base_configs)

        if arm_path_len != base_path_len:
            print(
                f"ERROR: Path lengths still don't match after interpolation! Arm: {arm_path_len}, Base: {base_path_len}"
            )
            # We should never reach here if whole-body interpolation worked correctly
            return

        print(f"Starting animation with {arm_path_len} synchronized waypoints...")

        try:
            for i in range(arm_path_len):
                # Get arm configuration for this step
                arm_config = arm_path[i]
                if isinstance(arm_config, list):
                    arm_config_list = arm_config
                elif isinstance(arm_config, np.ndarray):
                    arm_config_list = arm_config.tolist()
                else:
                    arm_config_list = arm_config.to_list()

                # Get base configuration for this step
                base_config = base_configs[i]

                # Set the base position and orientation
                quat = pb.getQuaternionFromEuler([0, 0, base_config[2]])
                sim.client.resetBasePositionAndOrientation(
                    sim.skel_id,
                    [base_config[0], base_config[1], 0],  # Position (x, y, z)
                    quat,  # Orientation as quaternion
                )

                # Set the arm configuration
                sim.set_joint_positions(arm_config_list)

                # Add a small delay for visualization
                time.sleep(0.05)

            print("Whole-body motion visualization complete!")

            # Keep the visualization window open
            print("Press Ctrl+C to exit...")
            while True:
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Visualization stopped by user.")
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        # Just show start and goal configurations
        print("Displaying start and goal configurations...")

        # Show start configuration
        quat_start = pb.getQuaternionFromEuler([0, 0, start_base[2]])
        sim.client.resetBasePositionAndOrientation(
            sim.skel_id,
            [start_base[0], start_base[1], 0],  # Position (x, y, z)
            quat_start,  # Orientation as quaternion
        )
        sim.set_joint_positions(start_joints)

        print("Showing start configuration. Press Enter to continue...")
        input()

        # Show goal configuration
        quat_goal = pb.getQuaternionFromEuler([0, 0, goal_base[2]])
        sim.client.resetBasePositionAndOrientation(
            sim.skel_id,
            [goal_base[0], goal_base[1], 0],  # Position (x, y, z)
            quat_goal,  # Orientation as quaternion
        )
        sim.set_joint_positions(goal_joints)

        print("Showing goal configuration. Press Enter to exit...")
        input()


if __name__ == "__main__":
    fire.Fire(main)
