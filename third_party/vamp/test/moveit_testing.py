#!/usr/bin/env python3
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle, Plane
from geometry_msgs.msg import Pose, Point
import numpy as np
import time
import os
import sys
import struct
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import glob
import open3d as o3d


def load_pointcloud(pcd_path):
    """Load a point cloud from file using Open3D."""
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        return points
    except Exception as e:
        return None


def load_all_pointclouds(directory):
    """
    Load all point cloud files from a directory.

    Args:
        directory: Directory containing point cloud files

    Returns:
        Combined point cloud as a numpy array
    """
    # Find all point cloud files (PLY, PCD, etc.)
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
    transformed_points = np.dot(points, rotation.T) + translation

    return transformed_points


def downsample_pointcloud(points, voxel_size=0.05):
    """
    Downsample a point cloud using voxel grid filtering.

    Args:
        points: Nx3 numpy array of points
        voxel_size: Size of voxel grid for downsampling

    Returns:
        Downsampled points as a numpy array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply voxel grid downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_points


def publish_pointcloud(points, pub, frame_id="map", color=(0, 255, 0)):
    """
    Publish point cloud for visualization in RViz with color.

    Args:
        points: Nx3 numpy array of points
        pub: ROS publisher for PointCloud2 messages
        frame_id: Frame ID for the point cloud
        color: RGB color tuple (0-255 for each value)
    """
    # Create header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Convert RGB color to uint32
    rgb = struct.unpack("I", struct.pack("BBBB", color[0], color[1], color[2], 255))[0]

    # Create a structured array with XYZ and RGB fields
    cloud_points = []
    for point in points:
        cloud_points.append((point[0], point[1], point[2], rgb))

    # Define the fields for the point cloud message
    fields = [
        pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField("rgb", 12, pc2.PointField.UINT32, 1),
    ]

    # Create and publish the point cloud message
    cloud_msg = pc2.create_cloud(header, fields, cloud_points)
    pub.publish(cloud_msg)


def add_collision_objects(scene, points, point_size=0.03):
    """
    Add collision objects to the planning scene from point cloud data.
    This uses a simplified approach with octree-based collision objects.

    Args:
        scene: MoveIt planning scene interface
        points: Nx3 numpy array of points
        point_size: Size of collision spheres
    """
    # Clear existing objects
    scene.remove_world_object()

    # Total points added to motion planner
    total_points = len(points)
    print(f"Adding {total_points} points to motion planner...")

    # Batch points into manageable chunks to avoid overwhelming the planning scene
    batch_size = 1000
    num_batches = (len(points) + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(points))
        batch_points = points[start_idx:end_idx]

        # Create collision object
        co = CollisionObject()
        co.header.stamp = rospy.Time.now()
        co.header.frame_id = "base_link"
        co.id = f"pointcloud_batch_{batch_idx}"
        co.operation = CollisionObject.ADD

        # Add spheres for each point in the batch
        for i, point in enumerate(batch_points):
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [point_size]  # Sphere radius

            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            pose.orientation.w = 1.0

            co.primitives.append(sphere)
            co.primitive_poses.append(pose)

        # Add collision object to the scene
        scene.add_object(co)

    add_time = time.time() - start_time
    add_time_us = add_time * 1000000  # Convert to microseconds
    print(f"TIME TO ADD POINTCLOUD: {add_time_us:.0f} μs for {total_points} points")


def main():
    """Test script for MoveIt motion planning performance with the Fetch robot."""
    # Initialize ROS node with minimal logging
    rospy.init_node("moveit_test", anonymous=True, log_level=rospy.FATAL)

    # Directory containing point cloud files
    pc_directory = "mp_collision_models"

    # Load all point cloud data from directory
    original_pc = load_all_pointclouds(pc_directory)
    if original_pc is None:
        print("Failed to load point clouds. Exiting.")
        return

    # Define transformation matrix from map to base_link - same as VAMP test
    transform_matrix = np.array(
        [
            [0.1470, -0.9890, 0, 0.4500],
            [0.9890, 0.1470, 0, 2.7723],
            [0, 0, 1, 0.0000],
            [0, 0, 0, 1.0000],
        ]
    )

    # Transform points using fast vectorized method
    transformed_pc = transform_points_fast(original_pc, transform_matrix)

    # Filter points by radius - using same value as VAMP test
    max_radius = 5.5  # Maximum radius in meters
    filtered_pc = filter_points_by_radius(transformed_pc, max_radius)

    # Skip downsampling - use all filtered points
    downsampled_pc = filtered_pc  # No downsampling, just use the filtered points

    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)

    # Initialize robot commander
    robot = moveit_commander.RobotCommander()

    # Initialize planning scene interface
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(0.5)  # Allow planning scene to initialize

    # Get the available groups from the robot
    available_groups = robot.get_group_names()

    # Initialize arm group
    arm_group = moveit_commander.MoveGroupCommander("arm")

    # Check if there's an arm_with_torso group or similar
    if "arm_with_torso" in available_groups:
        combined_group = moveit_commander.MoveGroupCommander("arm_with_torso")
        has_torso_planning = True
    elif "fetch" in available_groups:
        combined_group = moveit_commander.MoveGroupCommander("fetch")
        has_torso_planning = True
    else:
        combined_group = arm_group
        has_torso_planning = False

    # Display trajectory publisher for visualization in RViz
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path", DisplayTrajectory, queue_size=20
    )

    # Point cloud publisher for visualization in RViz
    pointcloud_publisher = rospy.Publisher(
        "/environment_pointcloud", PointCloud2, queue_size=1
    )

    # Publish point cloud for visualization in RViz
    max_viz_points = 75000
    if len(downsampled_pc) > max_viz_points:
        viz_indices = np.random.choice(
            len(downsampled_pc), max_viz_points, replace=False
        )
        viz_points = downsampled_pc[viz_indices]
    else:
        viz_points = downsampled_pc

    publish_pointcloud(viz_points, pointcloud_publisher)

    # Wait for RViz to visualize point cloud
    rospy.sleep(1.0)

    # Add point cloud as collision objects
    add_collision_objects(scene, downsampled_pc)

    # Define start and goal joint configurations
    arm_start = [1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]  # Arm joints
    arm_goal = [0.80, -0.40, -1.5, 1.5, 1.0, -0.0, 2.169129759130249]  # Arm joints

    # Set arm start configuration
    arm_group.set_joint_value_target(arm_start)
    arm_success = arm_group.go(wait=True)

    # Find the torso joint name if we have torso planning
    torso_joint_name = None
    if has_torso_planning:
        # Find the torso joint (usually named "torso_lift_joint")
        joint_names = robot.get_joint_names()
        for joint in joint_names:
            if "torso" in joint.lower() and "lift" in joint.lower():
                torso_joint_name = joint
                break

    # Define combined start and goal states
    if has_torso_planning and torso_joint_name:
        # Get the current joint values for the combined group
        current_joints = combined_group.get_current_joint_values()

        # Find the index of the torso joint in the combined group
        combined_joint_names = combined_group.get_active_joints()
        torso_index = -1
        for i, name in enumerate(combined_joint_names):
            if name == torso_joint_name:
                torso_index = i
                break

        if torso_index >= 0:
            # Create combined joint targets with torso height
            combined_joints = current_joints.copy()
            combined_joints[torso_index] = 0.3  # Start torso height
            combined_start = combined_joints

            combined_joints = current_joints.copy()
            combined_joints[torso_index] = 0.37  # Goal torso height
            # Set arm joints
            arm_indices = [
                i
                for i, name in enumerate(combined_joint_names)
                if name in arm_group.get_active_joints()
            ]
            for i, arm_idx in enumerate(arm_indices):
                if i < len(arm_goal):
                    combined_joints[arm_idx] = arm_goal[i]
            combined_goal = combined_joints
        else:
            combined_start = arm_start
            combined_goal = arm_goal
    else:
        combined_start = arm_start
        combined_goal = arm_goal

    # Set planning parameters similar to VAMP's
    combined_group.set_planning_time(5.0)  # 5 seconds planning time
    combined_group.set_num_planning_attempts(10)
    combined_group.set_max_velocity_scaling_factor(0.5)
    combined_group.set_max_acceleration_scaling_factor(0.5)

    # Set goal for planning
    combined_group.set_joint_value_target(combined_goal)

    # Plan and measure planning time
    print("Planning motion trajectory...")
    plan_start_time = time.time()
    success, plan, planning_time, error_code = combined_group.plan()
    plan_time = time.time() - plan_start_time

    # Display results
    if success:
        plan_time_us = plan_time * 1000000  # Convert to microseconds
        print(f"TIME TO FIND TRAJECTORY: {plan_time_us:.0f} μs")
        print(f"Trajectory has {len(plan.joint_trajectory.points)} points")

        # Display trajectory in RViz
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)

        # Execute the plan
        execute_start_time = time.time()
        combined_group.execute(plan, wait=True)
        execute_time = time.time() - execute_start_time
        execute_time_us = execute_time * 1000000  # Convert to microseconds
        print(f"TIME TO EXECUTE TRAJECTORY: {execute_time_us:.0f} μs")

    else:
        plan_time_us = plan_time * 1000000  # Convert to microseconds
        print(f"Failed to find a trajectory. Error code: {error_code}")
        print(f"TIME TO ATTEMPT PLANNING: {plan_time_us:.0f} μs")

    # Clean up
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
