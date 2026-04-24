#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time
import numpy as np
import glob
import os
import open3d as o3d
from trac_ik_python.trac_ik import IK
from geometry_msgs.msg import Pose
import vamp  # Import VAMP for collision checking
from vamp import pybullet_interface as vpb
from pathlib import Path
import math
import fire
import pandas as pd
from scipy.spatial.transform import Rotation as R


# --- Helper Functions for Point Cloud Handling ---
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


# --- Advanced Seed Initialization Functions ---
def load_costmap(costmap_path):
    """
    Load a previously generated costmap from file.

    Args:
        costmap_path: Path to the costmap .npz file

    Returns:
        costmap: 2D numpy array representing the costmap
        metadata: Dictionary with costmap metadata
    """
    print(f"Loading costmap from {costmap_path}")

    try:
        data = np.load(costmap_path)

        # Extract costmap and metadata
        costmap = data["costmap"]

        # Create metadata dictionary
        metadata = {
            "resolution": float(data["resolution"]),
            "origin_x": float(data["origin_x"]),
            "origin_y": float(data["origin_y"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "max_distance": float(data["max_distance"]),
        }

        # Check if valid_area_mask exists in the data
        if "valid_area_mask" in data:
            metadata["valid_area_mask"] = data["valid_area_mask"]

        print(
            f"Loaded costmap with dimensions {metadata['width']}x{metadata['height']}"
        )
        return costmap, metadata

    except Exception as e:
        print(f"Error loading costmap: {e}")
        return None, None


def world_to_grid(world_x, world_y, metadata):
    """
    Convert world coordinates to grid coordinates.

    Args:
        world_x, world_y: World coordinates
        metadata: Costmap metadata

    Returns:
        grid_x, grid_y: Grid coordinates
    """
    grid_x = int((world_x - metadata["origin_x"]) / metadata["resolution"])
    grid_y = int((world_y - metadata["origin_y"]) / metadata["resolution"])
    return grid_x, grid_y


def grid_to_world(grid_x, grid_y, metadata):
    """
    Convert grid coordinates to world coordinates.

    Args:
        grid_x, grid_y: Grid coordinates
        metadata: Costmap metadata

    Returns:
        world_x, world_y: World coordinates
    """
    world_x = metadata["origin_x"] + grid_x * metadata["resolution"]
    world_y = metadata["origin_y"] + grid_y * metadata["resolution"]
    return world_x, world_y


def find_valid_base_positions(
    costmap, metadata, ee_pose, manipulation_radius, cost_threshold=0.3
):
    """
    Find valid base positions for the robot by intersecting the costmap with a circle
    representing the manipulation range around the end effector.

    Args:
        costmap: 2D costmap array (values 0-1, NaN outside boundary)
        metadata: Costmap metadata dictionary
        ee_pose: End effector pose
        manipulation_radius: Radius of the manipulation range in meters
        cost_threshold: Minimum cost value to consider a cell valid (0-1)

    Returns:
        valid_positions: List of valid base positions in world coordinates with costs and orientation scores
        circle_mask: 2D boolean array showing the circular region
        valid_mask: 2D boolean array showing the valid base positions
    """
    # Convert end effector position to grid coordinates
    ee_grid_x, ee_grid_y = world_to_grid(
        ee_pose.position.x, ee_pose.position.y, metadata
    )

    # Create 2D grid coordinates
    y_grid, x_grid = np.mgrid[0 : metadata["height"], 0 : metadata["width"]]

    # Calculate distances from each grid cell to end effector position
    distances = (
        np.sqrt((x_grid - ee_grid_x) ** 2 + (y_grid - ee_grid_y) ** 2)
        * metadata["resolution"]
    )

    # Create circle mask (cells within manipulation range)
    circle_mask = distances <= manipulation_radius

    # Create valid cells mask (cells with cost >= threshold and not NaN)
    valid_cells = (costmap >= cost_threshold) & ~np.isnan(costmap)

    # Combine circle mask and valid cells mask to get final valid positions
    valid_mask = circle_mask & valid_cells

    # Convert valid grid positions to world coordinates
    valid_positions = []
    for y in range(metadata["height"]):
        for x in range(metadata["width"]):
            if valid_mask[y, x]:
                world_x, world_y = grid_to_world(x, y, metadata)

                # Calculate orientation towards the end effector
                dx = ee_pose.position.x - world_x
                dy = ee_pose.position.y - world_y
                theta = math.atan2(dy, dx)

                # Get the cost value (lower is better for sampling)
                cost_value = costmap[y, x]

                valid_positions.append((world_x, world_y, cost_value, theta))

    return valid_positions, circle_mask, valid_mask


def deterministic_sample_base_positions(valid_positions, num_samples=5):
    """
    Sample base positions from the valid positions in a deterministic manner.
    Always selects positions with the lowest cost values.

    Args:
        valid_positions: List of valid base positions (world_x, world_y, cost, theta)
        num_samples: Number of positions to sample

    Returns:
        sampled_positions: Sampled base positions with orientation (world_x, world_y, theta)
    """
    if not valid_positions:
        print("No valid base positions to sample from")
        return []

    # If fewer valid positions than requested samples, return all positions
    if len(valid_positions) <= num_samples:
        return [(x, y, theta) for x, y, _, theta in valid_positions]

    # Sort positions by cost (ascending order)
    sorted_positions = sorted(valid_positions, key=lambda pos: pos[2])

    # Take the top num_samples positions with lowest costs
    top_positions = sorted_positions[:num_samples]

    # Return top positions with their orientation
    sampled_positions = [(x, y, theta) for x, y, _, theta in top_positions]

    return sampled_positions


# Function to generate deterministic seed using costmap for base position
def generate_advanced_seed(
    lower, upper, pose, costmap_path="costmap.npz", manipulation_radius=1.0
):
    """
    Generate a deterministic seed for IK using costmap for base position and fixed midpoint values for arm joints.
    Always selects the best (lowest cost) base position from the costmap.

    Args:
        lower: Lower joint limits
        upper: Upper joint limits
        pose: Target end effector pose
        costmap_path: Path to the costmap file
        manipulation_radius: Radius of manipulation for base sampling

    Returns:
        seed: Deterministic seed for IK
    """
    print("Generating deterministic seed for IK...")

    # Initialize the seed array with midpoints as fallback
    seed = [(l + u) / 2.0 for l, u in zip(lower, upper)]  # noqa: E741

    try:
        # Try to load the costmap for base position sampling
        costmap, metadata = load_costmap(costmap_path)

        if costmap is not None and metadata is not None:
            # Find valid base positions
            valid_positions, _, _ = find_valid_base_positions(
                costmap, metadata, pose, manipulation_radius, cost_threshold=0.3
            )

            print(f"Found {len(valid_positions)} valid base positions")

            if valid_positions:
                # Always select the position with the lowest cost
                sampled_positions = deterministic_sample_base_positions(
                    valid_positions, num_samples=1
                )

                if sampled_positions:
                    # Extract best base position and orientation
                    base_x, base_y, base_theta = sampled_positions[0]

                    # Update the first 3 elements of the seed (base position and orientation)
                    seed[0] = base_x
                    seed[1] = base_y
                    seed[2] = base_theta

                    print(
                        f"Using best base position: [{base_x:.4f}, {base_y:.4f}, {base_theta:.4f}]"
                    )
    except Exception as e:
        print(f"Error in base position sampling: {e}")
        print("Falling back to midpoint values for base position")

        # Use midpoint values for base position as a deterministic fallback (first 3 DOFs)
        for i in range(3):
            seed[i] = (lower[i] + upper[i]) / 2.0

    # Use midpoint values for arm joints (remaining DOFs)
    for i in [5, 6, 7, 8, 9, 10]:
        seed[i] = (lower[i] + upper[i]) / 2.0

    print(f"Deterministic seed generated: {[round(val, 3) for val in seed]}")
    return seed


# --- Functions from the motion planning code ---
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


def main(
    pc_directory: str = "mp_collision_models",
    point_radius: float = 0.03,
    max_radius: float = 20.0,
    **kwargs,
):
    """Integrated whole body IK and motion planning with point cloud environments."""

    # --- PART 1: Set up environment and load point clouds ---
    print("\n=== SETTING UP ENVIRONMENT AND LOADING POINT CLOUDS ===\n")

    # Set up VAMP for robot collision checking
    robot = "fetch"  # Use fetch robot model
    planner = "rrtc"
    vamp_module, _, plan_settings, simp_settings = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
    )

    # Create VAMP environment
    env = vamp.Environment()

    # Load point clouds from directory
    print("Loading point clouds from directory...")
    original_pc = load_all_pointclouds(pc_directory)
    viz_points = []

    if original_pc is not None:
        # Filter points by radius
        filtered_pc = filter_points_by_radius(original_pc, max_radius)

        # Sample points for visualization
        viz_points = sample_points_for_visualization(filtered_pc, max_points=75000)

        # Add point cloud to VAMP environment for collision checking
        r_min, r_max = vamp.ROBOT_RADII_RANGES[robot]

        print(f"Adding {len(viz_points)} filtered points to environment...")
        add_start_time = time.time()
        build_time = env.add_pointcloud(viz_points, r_min, r_max, point_radius)
        add_time = time.time() - add_start_time

        print(f"CAPT construction time: {build_time * 1e-6:.3f}ms")
        print(f"Point cloud add time: {add_time:.3f}s")
    else:
        print("Warning: No point cloud data loaded.")
        viz_points = []

    # --- PART 2: Run Whole Body IK to find goal configuration ---
    print("\n=== RUNNING WHOLE BODY IK TO FIND GOAL CONFIGURATION ===\n")

    # Set up IK solver with fetch_ext.urdf
    print("Solving inverse kinematics using fetch_ext.urdf...")
    BASE_LINK = "world_link"
    EE_LINK = "gripper_link"

    # Open the fetch_ext URDF file for IK solution
    with open("resources/fetch_ext/fetch ext.urdf", "r") as f:
        urdf_str = f.read()

    ik_solver = IK(BASE_LINK, EE_LINK, urdf_string=urdf_str, timeout=0.5, epsilon=1e-6)

    max_samples = 100  # Predefined maximum number of attempts

    # Get joint limits
    lower, upper = ik_solver.get_joint_limits()

    # Set desired pose - use the test case from the first file
    # Test cases
    test_cases = [
        [-3.66, -3.23, 1.27, 0.427, -0.726, -0.476, 0.254],
        [-3.66, -3.23, 1.00, 0.0, 1.0, 0.0, 0.0],
        [-2.75, -0.79, 1.26, 0.42737215, -0.7255706, -0.47561587, 0.25434207],
        [-2.75, -0.79, 1.26, 1.0, 0.0, 0.0, 0.0],
        [-2.75, -0.79, 1.26, 0.0, 1.0, 0.0, 0.0],
    ]

    test_index = 0

    # Set desired pose
    pose = Pose()
    pose.position.x = test_cases[test_index][0]
    pose.position.y = test_cases[test_index][1]
    pose.position.z = test_cases[test_index][2]
    pose.orientation.x = test_cases[test_index][3]
    pose.orientation.y = test_cases[test_index][4]
    pose.orientation.z = test_cases[test_index][5]
    pose.orientation.w = test_cases[test_index][6]
    # pose = Pose()
    # pose.position.x = -3.66
    # pose.position.y = -3.23
    # pose.position.z = 1.26
    # pose.orientation.x = 0.427
    # pose.orientation.y = -0.726
    # pose.orientation.z = -0.476
    # pose.orientation.w = 0.254

    # Initialize variables for the resampling loop
    solution = None
    is_valid = False
    sample_count = 0

    all_time = time.time()

    while not is_valid and sample_count < max_samples:
        sample_count += 1
        print(f"\nAttempt {sample_count}/{max_samples}")

        # Use advanced seed generation instead of midpoint approach
        costmap_path = "costmap.npz"  # Update this to the actual path of your costmap
        seed = generate_advanced_seed(
            lower, upper, pose, costmap_path, manipulation_radius=1.0
        )
        print(
            f"Initial configuration (advanced seed): {[round(val, 3) for val in seed]}"
        )

        # Start timing IK solution
        start_time = time.time()

        # Solve IK
        solution = ik_solver.get_ik(
            seed,
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        # End timing
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"IK solving time: {solve_time:.4f} seconds")

        if not solution:
            print("[x] IK failed. Trying again...")
            continue

        print("[✓] IK solution found!")

        # --- Perform Collision Checking with VAMP ---
        print("Performing collision checking on IK solution...")

        # VAMP only takes 8 DOF (torso + 7 arm joints), not the full solution
        vamp_solution = list(solution)
        base_config = vamp_solution[:3]

        if len(vamp_solution) > 8:
            print(
                f"Extracting 8 DOF from {len(vamp_solution)}-DOF IK solution for VAMP validation"
            )
            # Extract the relevant joints for VAMP (skipping base)
            vamp_solution = vamp_solution[3:]

        print(f"Validating arm configuration: {[round(v, 3) for v in vamp_solution]}")
        vamp_module.set_base_params(base_config[2], base_config[0], base_config[1])

        # Check if the solution is valid (collision-free)
        is_valid = vamp_module.validate(vamp_solution, env)

        if is_valid:
            print("[✓] IK solution is valid (collision-free)")
        else:
            print("[x] IK solution is not valid (has collisions). Trying again...")

    # Check if we found a valid solution
    if is_valid:
        print(f"\n[✓] Found valid solution after {sample_count} attempts")
        all_time_end = time.time()
        print(
            f"Searching valid whole ik totally use {all_time_end - all_time:.4f} seconds"
        )
    else:
        print(
            f"\n[x] Failed to find valid solution after {max_samples} attempts. Exiting."
        )
        return

    # Extract the base position and arm configuration
    if len(solution) >= 3:
        base_position = [solution[0], solution[1]]  # x, y
        base_orientation = solution[2]  # theta (yaw)
        print(
            f"\nExtracted base position from first 3 values: [{base_position[0]:.4f}, {base_position[1]:.4f}]"
        )
        print(f"Extracted base orientation (yaw): {base_orientation:.4f}")

        # Create the goal_base configuration for motion planning
        goal_base = [base_position[0], base_position[1], base_orientation]
    else:
        print(
            "\nWarning: Solution has fewer than 3 values, cannot extract base position"
        )
        return

    # Extract the arm configuration
    if len(solution) >= 11:  # If we have the expected 11 values (3 base + 8 arm)
        arm_config = solution[3:11]  # Extract 8 arm joint values
        print(
            f"\nExtracted arm configuration from joints 3-10: {[round(v, 3) for v in arm_config]}"
        )

        # Set the goal_joints for motion planning
        goal_joints = arm_config
    else:
        print("\nWarning: Solution doesn't have enough values for arm configuration")
        return

    # --- PART 3: Motion Planning to the IK Solution ---
    print("\n=== RUNNING MOTION PLANNING TO THE IK SOLUTION ===\n")

    # Define start configuration (you can change these based on your needs)
    start_base = [0.0, 0.0, 0.0]  # x, y, theta
    start_joints = [0.3, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]  # Default arm configuration

    # Print end effector poses for start and goal configurations
    print_ee_poses(vamp_module, start_joints, goal_joints, start_base, goal_base)

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

        print(
            f"Using whole_body_simplify with {len(arm_path_list)} arm configurations and {len(base_path_list)} base configurations"
        )

        # Use whole_body_simplify
        whole_body_result = vamp_module.whole_body_simplify(
            arm_path_list, base_path_list, env, simp_settings, sampler
        )

        # Print statistics
        stats = vamp.results_to_dict(result.arm_result, whole_body_result.arm_result)

        # Add base planning metrics to stats dictionary
        stats["base_planning_time"] = pd.Timedelta(
            nanoseconds=result.base_result.nanoseconds
        )
        stats["base_planning_iterations"] = result.base_result.iterations

        # Update total planning time to use the overall result nanoseconds
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
        interpolation_resolution = 64
        whole_body_result.interpolate(interpolation_resolution)
        print(f"Interpolated with resolution {interpolation_resolution}")

        # Get the interpolated paths
        arm_path = whole_body_result.arm_result.path
        base_path = whole_body_result.base_path

        # Extract base configurations as lists
        base_configs = []
        for config in base_path:
            base_configs.append(config.config)

        print(f"Base path length: {len(base_configs)} waypoints")
        print(f"Arm path length: {len(arm_path)} waypoints")
    else:
        print("Failed to solve problem! Displaying start and goal.")

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

    # --- PART 4: Visualization ---
    print("\n=== VISUALIZING RESULTS ===\n")

    # Create simulator using VAMP's configuration
    robot_dir = Path("resources") / "fetch"

    # Use the fetch_spherized.urdf path
    spherized_urdf_path = str(robot_dir / "fetch_spherized.urdf")
    print(f"Loading robot from: {spherized_urdf_path}")

    # Ensure the URDF file exists
    if not os.path.exists(spherized_urdf_path):
        print(
            f"Warning: {spherized_urdf_path} not found. Please ensure the file exists."
        )
        # Fallback to regular fetch URDF if spherized not found
        alternative_path = str(
            Path(pybullet_data.getDataPath()) / "fetch" / "fetch.urdf"
        )
        if os.path.exists(alternative_path):
            print(f"Using alternative URDF: {alternative_path}")
            spherized_urdf_path = alternative_path
        else:
            print("No suitable URDF found. Exiting.")
            return

    sim = vpb.PyBulletSimulator(
        spherized_urdf_path,
        vamp.ROBOT_JOINTS["fetch"],
        True,
    )

    # Visualize point cloud
    print(f"Visualizing {len(viz_points)} sampled points in PyBullet")
    sim.draw_pointcloud(viz_points)

    # Add target marker at goal position
    p.addUserDebugText(
        "Target Position",
        [pose.position.x, pose.position.y, pose.position.z],
        [1, 1, 1],
        1.0,
    )

    # Add coordinate axes at target position
    axis_length = 0.1
    p.addUserDebugLine(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.position.x + axis_length, pose.position.y, pose.position.z],
        [1, 0, 0],
        3,  # X-axis (red)
    )
    p.addUserDebugLine(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.position.x, pose.position.y + axis_length, pose.position.z],
        [0, 1, 0],
        3,  # Y-axis (green)
    )
    p.addUserDebugLine(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.position.x, pose.position.y, pose.position.z + axis_length],
        [0, 0, 1],
        3,  # Z-axis (blue)
    )

    # Animate the whole-body motion (base + arm) if planning was successful
    if result.is_successful():
        print("Visualizing whole-body motion (base + arm)...")

        # After whole-body interpolation, both paths should have the same length
        arm_path_len = len(arm_path)
        base_path_len = len(base_configs)

        if arm_path_len != base_path_len:
            print(
                f"ERROR: Path lengths don't match after interpolation! Arm: {arm_path_len}, Base: {base_path_len}"
            )
            # Display just the start and goal configurations

            # Show start configuration
            quat_start = p.getQuaternionFromEuler([0, 0, start_base[2]])
            sim.client.resetBasePositionAndOrientation(
                sim.skel_id,
                [start_base[0], start_base[1], 0],
                quat_start,
            )
            sim.set_joint_positions(start_joints)

            print("Showing start configuration. Press Enter to continue...")
            input()

            # Show goal configuration
            quat_goal = p.getQuaternionFromEuler([0, 0, goal_base[2]])
            sim.client.resetBasePositionAndOrientation(
                sim.skel_id,
                [goal_base[0], goal_base[1], 0],
                quat_goal,
            )
            sim.set_joint_positions(goal_joints)

            print("Showing goal configuration. Press Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Visualization stopped by user.")

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
                quat = p.getQuaternionFromEuler([0, 0, base_config[2]])
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
            print("Showing final configuration (goal). Press Ctrl+C to exit...")

            # Keep the visualization window open
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Visualization stopped by user.")
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        # Just show start and goal configurations
        print("Displaying start and goal configurations...")

        # Show start configuration
        quat_start = p.getQuaternionFromEuler([0, 0, start_base[2]])
        sim.client.resetBasePositionAndOrientation(
            sim.skel_id,
            [start_base[0], start_base[1], 0],  # Position (x, y, z)
            quat_start,  # Orientation as quaternion
        )
        sim.set_joint_positions(start_joints)

        print("Showing start configuration. Press Enter to continue...")
        input()

        # Show goal configuration
        quat_goal = p.getQuaternionFromEuler([0, 0, goal_base[2]])
        sim.client.resetBasePositionAndOrientation(
            sim.skel_id,
            [goal_base[0], goal_base[1], 0],  # Position (x, y, z)
            quat_goal,  # Orientation as quaternion
        )
        sim.set_joint_positions(goal_joints)

        print("Showing goal configuration. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Visualization stopped by user.")


if __name__ == "__main__":
    fire.Fire(main)
