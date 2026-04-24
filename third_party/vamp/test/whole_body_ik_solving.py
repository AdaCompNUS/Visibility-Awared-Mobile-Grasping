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


# --- Helper Functions for Point Cloud Handling (Matching paste-2.txt exactly) ---
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


# --- Advanced Seed Initialization Functions (from paste-2.txt) ---


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
    # for tmp_1, tmp_2 in zip(lower, upper):
    #     print(f"Lower bound: {tmp_1} Upper boun: {tmp_2}")

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
        # seed[i] = (lower[i] + upper[i]) / 2.0
        seed[i] = np.random.uniform(lower[i], upper[i])

    print(f"Deterministic seed generated: {[round(val, 3) for val in seed]}")
    return seed


# --- Set up environment parameters ---
pc_directory = "mp_collision_models"  # Default directory path, can be changed
max_radius = 20.0  # Maximum distance from origin for filtering
point_radius = 0.03  # Point radius parameter for collision checking

# --- Configure VAMP for Collision Checking ---
# Set up VAMP for robot collision checking
robot = "fetch"  # Use fetch robot model
vamp_module, _, plan_settings, simp_settings = (
    vamp.configure_robot_and_planner_with_kwargs(robot, "rrtc")
)

# Create VAMP environment
env = vamp.Environment()

# --- Load Point Clouds ---
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

# --- Run Trac-IK with fetch_ext.urdf for IK solving ---
print("Solving inverse kinematics using fetch_ext.urdf...")
BASE_LINK = "world_link"
EE_LINK = "gripper_link"

# Open the fetch_ext URDF file for IK solution
with open("resources/fetch_ext/fetch ext.urdf", "r") as f:
    urdf_str = f.read()

ik_solver = IK(BASE_LINK, EE_LINK, urdf_string=urdf_str, timeout=0.5, epsilon=1e-6)

# Add this at the beginning of your script or before this section
max_samples = 10  # Predefined maximum number of attempts

# Get joint limits
lower, upper = ik_solver.get_joint_limits()

# Test cases
test_cases = [
    [-3.66, -3.23, 1.27, 0.427, -0.726, -0.476, 0.254],
    [-3.66, -3.23, 1.00, 0.0, 1.0, 0.0, 0.0],
    [-2.75, -0.79, 1.26, -0.381, 0.899, 0.140, -0.164],
    [-2.75, -0.79, 1.26, 0.42737215, -0.7255706, -0.47561587, 0.25434207],
    [-2.75, -0.79, 1.26, 1.0, 0.0, 0.0, 0.0],
    [-2.75, -0.79, 1.26, 0.0, 1.0, 0.0, 0.0],
]

# position [-3.301, 1.396, 0.874], orientation [-0.381, 0.899, 0.140, -0.164]

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
    print(f"Initial configuration (advanced seed): {[round(val, 3) for val in seed]}")

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
        # Extract the first 8 joints for VAMP
        vamp_solution = vamp_solution[3:]

    print(f"Validating arm configuration: {[round(v, 3) for v in vamp_solution]}")
    vamp_module.set_base_params(base_config[2], base_config[0], base_config[1])

    # Check if the solution is valid (collision-free)
    is_valid = vamp_module.validate(vamp_solution, env)

    if is_valid:
        print("[✓] IK solution is valid (collision-free)")
    else:
        print("[x] IK solution is not valid (has collisions). Trying again...")

# After the loop, check if we found a valid solution
if is_valid:
    print(f"\n[✓] Found valid solution after {sample_count} attempts")
    all_time_end = time.time()
    print(f"Searching valid whole ik totally use {all_time_end - all_time:.4f} seconds")
    # You can add code here to use the solution
else:
    print(f"\n[x] Failed to find valid solution after {max_samples} attempts. Exiting.")
    exit()

# --- Extract the base position from the first 3 joints in the solution ---
print("\nIK solver joint names:")
ik_joint_names = ik_solver.joint_names
for i, name in enumerate(ik_joint_names):
    print(f"  {i}: {name} = {solution[i]:.4f}")

# Use the first 3 values for base position
if len(solution) >= 3:
    base_position = [solution[0], solution[1], 0]  # x, y and keep z at 0
    base_orientation = [0, 0, solution[2]]  # Use the third value for theta (yaw)
    print(
        f"\nExtracted base position from first 3 values: [{base_position[0]:.4f}, {base_position[1]:.4f}, {base_position[2]:.4f}]"
    )
    print(f"Extracted base orientation (yaw): {base_orientation[2]:.4f}")
else:
    base_position = [0, 0, 0]
    base_orientation = [0, 0, 0]
    print(
        "\nWarning: Solution has fewer than 3 values, using default base position and orientation"
    )

# The arm configuration starts from the 4th value (index 3) in the solution
if len(solution) >= 11:  # If we have the expected 11 values (3 base + 8 arm)
    arm_config = solution[3:11]  # Extract 8 arm joint values
    print(
        f"\nExtracted arm configuration from joints 3-10: {[round(v, 3) for v in arm_config]}"
    )
elif len(solution) > 3:
    # Use whatever arm joints are available after the base
    arm_config = solution[3:]
    print(f"\nExtracted partial arm configuration: {[round(v, 3) for v in arm_config]}")
else:
    # Fallback if we don't have enough values
    arm_config = vamp_solution
    print(f"\nUsing default VAMP solution for arm: {[round(v, 3) for v in arm_config]}")

# --- Initialize VAMP PyBullet Simulator for Visualization ---
print("\nInitializing VAMP PyBullet simulator for visualization...")

# Create simulator using VAMP's configuration
robot_dir = Path("resources") / "fetch"
if not robot_dir.exists():
    print(f"Warning: Robot directory {robot_dir} not found. Creating it.")
    robot_dir.mkdir(parents=True, exist_ok=True)

# Use the fetch_spherized.urdf path from the reference code
spherized_urdf_path = str(robot_dir / "fetch_spherized.urdf")
print(f"Loading robot from: {spherized_urdf_path}")

# Ensure the URDF file exists
if not os.path.exists(spherized_urdf_path):
    print(f"Warning: {spherized_urdf_path} not found. Please ensure the file exists.")
    # Fallback to regular fetch URDF if spherized not found
    alternative_path = str(Path(pybullet_data.getDataPath()) / "fetch" / "fetch.urdf")
    if os.path.exists(alternative_path):
        print(f"Using alternative URDF: {alternative_path}")
        spherized_urdf_path = alternative_path
    else:
        print("No suitable URDF found. Exiting.")
        exit()

sim = vpb.PyBulletSimulator(
    spherized_urdf_path,
    vamp.ROBOT_JOINTS["fetch"],
    True,
)

# --- Visualize Point Cloud with VAMP Bullet's draw_points ---
print(f"Visualizing {len(viz_points)} sampled points with VAMP bullet simulator")
sim.draw_pointcloud(viz_points)

# --- Set Robot Base Position and Arm Configuration in VAMP Simulator ---
# Set the base position and orientation based on the first 3 values from the IK solution

print(
    f"Setting base position: [{base_position[0]:.4f}, {base_position[1]:.4f}, {base_position[2]:.4f}]"
)
print(f"Setting base orientation (yaw): {base_orientation[2]:.4f}")

# Convert orientation to quaternion
quat = p.getQuaternionFromEuler(base_orientation)

# Set the base position and orientation in the VAMP simulator
sim.client.resetBasePositionAndOrientation(
    sim.skel_id,
    base_position,  # Position from IK solution (first 2 values)
    quat,  # Orientation as quaternion (from 3rd value)
)

# Set the arm configuration
print(f"Setting arm configuration: {[round(v, 3) for v in arm_config]}")
sim.set_joint_positions(arm_config)

# --- Add Target Marker at Goal Position ---
# Add a simple marker at the target position
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

# --- Get Current End Effector Position using VAMP's Forward Kinematics ---
print("\nGetting current end effector position...")
ee_pos, ee_quat = vamp_module.eefk(arm_config)

# Transform from robot base to world frame
world_ee_pos = [
    base_position[0] + ee_pos[0],
    base_position[1] + ee_pos[1],
    base_position[2] + ee_pos[2],
]

print("\nCurrent end effector position:")
print(f"  Position in robot frame: {ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}")
print(
    f"  Position in world frame: {world_ee_pos[0]:.4f}, {world_ee_pos[1]:.4f}, {world_ee_pos[2]:.4f}"
)
print(f"  Target: {pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}")


# We can also get the EE position directly from the simulator for verification
ee_link_name = "gripper_link"  # This is typically the end effector link in Fetch
ee_link_index = -1

# Find the index of the end effector link
for i in range(p.getNumJoints(sim.skel_id)):
    info = p.getJointInfo(sim.skel_id, i)
    name = info[1].decode("utf-8")
    if name == ee_link_name:
        ee_link_index = i
        break

if ee_link_index != -1:
    # Get state of the end effector link in the simulator
    sim_ee_state = p.getLinkState(sim.skel_id, ee_link_index)
    sim_ee_pos = sim_ee_state[0]  # Position
    sim_ee_orn = sim_ee_state[1]  # Orientation

    print("\nSimulator end effector position:")
    print(f"  Position: {sim_ee_pos[0]:.4f}, {sim_ee_pos[1]:.4f}, {sim_ee_pos[2]:.4f}")

    # Calculate distance again using simulator position
    sim_dist = np.sqrt(
        sum(
            (sim_ee_pos[i] - p[i]) ** 2
            for i, p in enumerate([pose.position.x, pose.position.y, pose.position.z])
        )
    )
    print(f"  Distance to target (using simulator position): {sim_dist:.4f} meters")
else:
    print(f"Could not find end effector link '{ee_link_name}' in the simulator model")

# Add a line connecting current EE position to target
p.addUserDebugLine(
    world_ee_pos,
    [pose.position.x, pose.position.y, pose.position.z],
    [0, 1, 1],  # Cyan
    1,
)

# --- Run static visualization ---
print("\nRunning static visualization - close the PyBullet window to exit")
print("The robot position is now fixed and will not move.")

while p.isConnected():
    time.sleep(0.1)

p.disconnect()
