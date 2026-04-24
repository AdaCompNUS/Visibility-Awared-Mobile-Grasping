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
    """Load all point cloud files from a directory."""
    print(f"Searching for point cloud files in {directory}...")
    all_files = glob.glob(os.path.join(directory, "*.ply")) + glob.glob(
        os.path.join(directory, "*.pcd")
    )
    print(f"Found {len(all_files)} point cloud files")
    if not all_files:
        print(f"No point cloud files found in {directory}")
        return None

    combined_points = [
        pc
        for file_path in all_files
        if (pc := load_pointcloud(file_path)) is not None and len(pc) > 0
    ]
    if not combined_points:
        print("No valid point clouds loaded")
        return None

    combined_pc = np.vstack(combined_points)
    print(
        f"Combined {len(combined_points)} point clouds with total {len(combined_pc)} points"
    )
    return combined_pc


def filter_points_by_radius(points, max_radius):
    """Filter out points beyond a specified radius from the origin."""
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= max_radius
    filtered_points = points[mask]
    print(
        f"Filtered points: kept {len(filtered_points)} of {len(points)} points (radius <= {max_radius})"
    )
    return filtered_points


def sample_points_for_visualization(points, max_points=100000):
    """Sample a subset of points for visualization."""
    if len(points) <= max_points:
        return points.tolist()
    indices = np.random.choice(len(points), size=max_points, replace=False)
    sampled_points = points[indices]
    print(
        f"Sampled {len(sampled_points)} points for visualization (from {len(points)} total)"
    )
    return sampled_points.tolist()


def main():
    """Test script for verifying point cloud and attachment collision checking."""
    # Setup planner configuration
    robot = "fetch"
    planner = "rrtc"

    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(robot, planner)
    )

    # --- THIS IS THE CRUCIAL TRANSFORMATION DATA ---
    # The base parameters define the transformation from the robot's local frame to the world frame.
    base_theta = -1.423
    base_x = -2.80515
    base_y = 0.03805
    vamp_module.set_base_params(base_theta, base_x, base_y)
    print(
        f"Set base parameters for VAMP: theta={base_theta:.6f}, x={base_x:.6f}, y={base_y:.6f}"
    )

    # Create environment
    env = vamp.Environment()

    # Load point clouds
    pc_directory = "mp_collision_models"
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print("Failed to load point clouds. Exiting.")
        return

    # Filter and add point cloud to environment
    max_radius = 1000
    filtered_pc = filter_points_by_radius(original_pc, max_radius)
    env.add_pointcloud(filtered_pc.tolist(), 0.03, 0.08, 0.03)

    # --- Define and add the attachment to the VAMP planner ---
    attachment_radius = 0.08
    attachment_offset = 0.12  # 12cm in front of the gripper (local X-axis)

    print(
        f"\nCreating an attachment with radius {attachment_radius}m at an offset of {attachment_offset}m from the end-effector."
    )
    attachment = vamp.Attachment([attachment_offset, 0, 0], [0, 0, 0, 1])
    attachment.add_spheres([vamp.Sphere([0, 0, 0], attachment_radius)])
    env.attach(attachment)
    print("Attachment added to the planning environment (in robot's local frame).")

    # Define start and goal configurations
    start_joints = [0.3, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]
    goal_joints = [0.37, 0.80, -0.40, -1.5, 1.5, 1.0, -0.0, 2.169129759130249]

    # Initialize sampler and plan path
    sampler = vamp_module.halton()
    print("\nPlanning motion with attachment...")
    result = planner_func(start_joints, goal_joints, env, plan_settings, sampler)

    if result.solved:
        print("Solved problem!")
        simplify = vamp_module.simplify(result.path, env, simp_settings, sampler)
        plan = simplify.path
        plan.interpolate(vamp_module.resolution())
    else:
        print("Failed to solve problem! Displaying start and goal.")
        plan = vamp_module.Path()
        plan.append(vamp_module.Configuration(start_joints))
        plan.append(vamp_module.Configuration(goal_joints))

    # --- Setup PyBullet Visualization ---
    robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
    sim = vpb.PyBulletSimulator(
        str(robot_dir / "fetch_spherized.urdf"),
        vamp.ROBOT_JOINTS["fetch"],
        True,
    )

    # Set the robot's base position IN THE WORLD FRAME for visualization
    robot_id = sim.skel_id
    quat = pb.getQuaternionFromEuler([0, 0, base_theta])
    sim.client.resetBasePositionAndOrientation(robot_id, [base_x, base_y, 0], quat)

    # Draw the point cloud environment
    viz_points = sample_points_for_visualization(filtered_pc, max_points=75000)
    sim.draw_pointcloud(viz_points)

    # Add the visual representation of the attachment sphere
    attachment_sphere_id = sim.add_sphere(
        attachment_radius, [0, 0, 0], color=[0.9, 0.1, 0.1, 1.0]
    )

    # --- Correct Animation Callback with World Transformation ---
    def animation_callback(configuration):
        """
        Updates the visual attachment by transforming VAMP's local-frame data
        into the PyBullet world frame.
        """
        # 1. Get EEF pose in the robot's LOCAL FRAME from VAMP
        local_pos_ee, orientation_xyzw = vamp_module.eefk(configuration)

        # 2. Update the VAMP attachment object (this happens in the local frame)
        attachment.set_ee_pose(local_pos_ee, orientation_xyzw)

        # 3. Get the attachment's position, also in the LOCAL FRAME
        posed_sphere_local = attachment.posed_spheres[0]
        local_pos_attachment = np.array(posed_sphere_local.position)

        # 4. Transform the attachment's local position to the WORLD FRAME
        c, s = np.cos(base_theta), np.sin(base_theta)

        # Apply 2D rotation
        rotated_pos = np.array(
            [
                c * local_pos_attachment[0] - s * local_pos_attachment[1],
                s * local_pos_attachment[0] + c * local_pos_attachment[1],
                local_pos_attachment[2],
            ]
        )

        # Apply base translation
        world_pos_attachment = rotated_pos + np.array([base_x, base_y, 0])

        # 5. Update the PyBullet sphere with the final WORLD position
        sim.update_object_position(attachment_sphere_id, world_pos_attachment.tolist())

    # Animate the path using the callback
    print("Animating path with world-transformed attachment...")
    sim.animate(plan, animation_callback)

    print(
        "\nVisualization running. The red sphere should follow the gripper. Press Ctrl+C to exit."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == "__main__":
    main()
