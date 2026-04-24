#!/usr/bin/env python3
import time

import numpy as np

from grasp_anywhere.robot.ik import trac_solve_arm_only
from grasp_anywhere.robot.utils.transform_utils import extract_position_orientation
from grasp_anywhere.samplers.place_sampler import HeuristicPlaceSampler
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning


class PlacePlanner:
    """
    Handles the stage of placing an object.
    This class is a pure planner, taking a segmented placement area point cloud
    and finding a valid place pose for the robot.
    """

    def __init__(
        self,
        robot,
        max_ik_attempts=5,
        max_replan_attempts=5,
        enable_visualization=False,
    ):
        """
        Initializes the PlacePlanner.
        Args:
            robot: An instance of the Fetch class.
            max_ik_attempts (int): Max attempts for the IK solver.
            max_replan_attempts (int): Max attempts for replanning due to collisions or timeouts.
            enable_visualization (bool): Whether to show visualization during planning.
            logger (callable, optional): Logger function to use. Defaults to print.
        """
        self.robot = robot
        self.max_ik_attempts = max_ik_attempts
        self.max_replan_attempts = max_replan_attempts
        self.enable_visualization = enable_visualization
        log.info("PlacePlanner initialized.")

    def go_to_prepose(
        self,
        base_config,
        arm_config,
        collision_points,
        enable_replanning=True,
        enable_pcd_alignment=True,
    ):
        """
        Moves the robot to a pre-place pose that is elevated above the target place pose.
        This method only uses whole-body motion planning and control.

        Args:
            base_config: Base configuration
            arm_config: Arm configuration
            collision_points: Point cloud for collision avoidance
            enable_replanning: Whether to enable replanning
            enable_pcd_alignment: Whether to enable point cloud alignment

        Returns:
            bool: True if successful, False otherwise
        """
        log.info("Moving to pre-place pose...")

        # Always use whole-body motion planning for pre-pose (no fixed-base IK)
        log.info("Using whole-body motion planning for pre-place pose.")
        success = move_to_config_with_replanning(
            self.robot,
            goal_joints=arm_config,
            goal_base=base_config,
            enable_replanning=enable_replanning,
            enable_pcd_alignment=enable_pcd_alignment,
            max_replan_attempts=self.max_replan_attempts,
        )

        if not success:
            log.warning("WARNING: Failed to move to pre-place pose.")
            return False

        log.info("Successfully moved to pre-place pose.")
        return True

    def put_down(
        self,
        place_pose,
        collision_points,
    ):
        """
        Executes the in-place placement sequence: move down, open gripper, move up.
        This method only uses joint-space control and assumes the robot is already
        at the pre-place pose (6cm above the target).

        Args:
            place_pose: 4x4 matrix representing the target place pose
            collision_points: Point cloud for collision avoidance

        Returns:
            bool: True if successful, False otherwise
        """
        log.info("Executing in-place placement sequence...")

        # Get current joint values (should be at elevated pose)
        elevated_joints = self.robot.get_arm_joint_values()
        base_config = self.robot.get_base_params()
        torso_pos = self.robot.get_torso_position()

        place_position, place_orientation = extract_position_orientation(place_pose)
        final_place_joints = trac_solve_arm_only(
            elevated_joints,
            base_config,
            torso_pos,
            place_position,
            place_orientation,
        )
        if final_place_joints is None:
            log.warning("WARNING: Failed to find IK for final place pose.")
            return False

        # Use motion planning to move to final place pose
        log.info("Executing motion planning to final place pose.")
        result = self.robot.send_joint_values(
            [torso_pos] + final_place_joints, duration=3.0
        )
        time.sleep(0.5)
        if result is None:
            log.warning(
                "WARNING: Failed to execute motion planning to final place pose."
            )
            return False

        # Open gripper to release object
        log.info("Opening gripper to release object...")
        self.robot.control_gripper(1.0)
        time.sleep(0.5)

        # Detach object from end-effector since it's now placed
        log.info("Detaching object from end-effector...")
        self.robot.detach_objects_from_eef()

        # Move back up to elevated pose using motion planning
        log.info("Moving back up to elevated pose...")
        result = self.robot.send_joint_values(
            [torso_pos] + elevated_joints, duration=3.0
        )
        time.sleep(0.5)
        if result is None:
            log.warning("WARNING: Failed to execute motion planning to elevated pose.")
            return False

        log.info("Successfully completed in-place placement sequence.")
        return True

    def execute(
        self,
        place_pose,
        base_config,
        arm_config,
        collision_points,
        enable_replanning=True,
        enable_pcd_alignment=True,
    ):
        """
        Executes the complete place sequence: go to pre-pose, then execute place.
        This method combines both whole-body motion and in-place control.
        """
        log.info("Executing full place sequence...")

        # First, go to pre-pose
        if not self.go_to_prepose(
            base_config,
            arm_config,
            collision_points,
            enable_replanning,
            enable_pcd_alignment,
        ):
            return False

        # Then, execute the in-place placement
        if not self.put_down(place_pose, collision_points):
            return False

        return True

    def plan(self, placement_pcd_world, combined_points):
        """
        Executes the place planning pipeline for a given placement area point cloud.

        Args:
            placement_pcd_world (np.ndarray): The point cloud of the placement area in the world frame.
            combined_points (np.ndarray): The full collision point cloud of the environment.

        Returns:
            Tuple of (base_config, arm_config) or (None, None) if no pose is found.
        """
        if placement_pcd_world is None or len(placement_pcd_world) == 0:
            log.error(
                "ERROR: Cannot plan place pose, extracted placement point cloud is empty."
            )
            return None, None

        # Use the HeuristicPlaceSampler to find a valid place pose.
        sampler = HeuristicPlaceSampler(
            robot=self.robot,
            placement_pcd_world=placement_pcd_world,
        )

        base_config, arm_config = sampler.sample_and_validate(
            combined_points, debug=self.enable_visualization
        )
        return base_config, arm_config

    def run_inplace(self, placement_pcd_world, static_collision_points=None):
        """
        Executes in-place placement using only arm movement (no base movement).
        Calculates the center of the point cloud, generates a top-down pose, and executes placement.

        Args:
            placement_pcd_world (np.ndarray): The point cloud of the placement area in the world frame.
            static_collision_points (np.ndarray, optional): Static collision points to reset environment.

        Returns:
            bool: True if successful, False otherwise.
        """
        if placement_pcd_world is None or len(placement_pcd_world) == 0:
            log.error("ERROR: Cannot plan in-place placement, point cloud is empty.")
            return False, None

        # Calculate center of the placement area
        place_center = np.mean(placement_pcd_world, axis=0)
        log.info(f"Planning in-place placement at center: {place_center}")

        if place_center[2] > 1.0:  # More than 1m high
            log.warning(
                f"WARNING: Placement center height {place_center[2]:.3f}m seems high for table-top manipulation."
            )
        elif place_center[2] < 0.3:  # Less than 30cm high
            log.warning(
                f"WARNING: Placement center height {place_center[2]:.3f}m seems low, might be floor or invalid."
            )

        # --- 2. Calculate top-down placement pose ---
        # Use same height offset as point_sampler.py for consistency (25cm above target)
        gripper_height_offset = 0.2  # Same as point_sampler.py radius parameter

        # Position gripper above the surface (top-down approach)
        place_position = place_center + np.array([0.0, 0.0, gripper_height_offset])

        log.info(f"Using gripper height offset: {gripper_height_offset:.3f}m")

        # --- 3. Create top-down placement pose matrix (following point_sampler.py approach) ---
        # Use top-down approach vector (same as point_sampler.py)
        approach_vec = np.array([0.0, 0.0, -1.0])
        place_pose_matrix = self._create_topdown_pose_matrix(
            place_position, approach_vec
        )

        log.info("Generated top-down placement pose using point_sampler approach.")

        # Execute in-place placement using only fixed-base IK
        return (
            self.execute_inplace_placement(place_pose_matrix, static_collision_points),
            place_pose_matrix,
        )

    def execute_inplace_placement(
        self, place_pose_matrix, static_collision_points=None
    ):
        """
        Executes the in-place placement sequence using only fixed-base IK.

        Args:
            place_pose_matrix (np.ndarray): 4x4 pose matrix for placement
            static_collision_points (np.ndarray, optional): Static collision points to reset environment.

        Returns:
            bool: True if successful, False otherwise
        """
        log.info("Executing in-place placement with fixed-base IK only...")

        # Get current robot state
        base_config = self.robot.get_base_params()
        torso_pos = self.robot.get_torso_position()
        current_arm_joints = self.robot.get_arm_joint_values()

        # Extract position and orientation for IK
        position, orientation = extract_position_orientation(place_pose_matrix)

        # Try fixed-base IK for placement pose
        arm_joints = trac_solve_arm_only(
            current_arm_joints, base_config, torso_pos, position, orientation
        )

        if arm_joints is None:
            log.error(
                "ERROR: In-place placement failed. Could not solve IK with fixed base for placement pose."
            )
            return False

        # 1. Move to placement pose (with dynamic collision avoidance if available)
        log.info("Moving to placement pose...")
        result = self.robot.send_joint_values([torso_pos] + arm_joints, duration=4.0)
        time.sleep(0.2)
        if result is None:
            log.warning("WARNING: Failed to execute motion planning to placement pose.")
            return False

        # Reset collision environment to static pointclouds only after reaching placement pose
        if static_collision_points is not None:
            log.info("Resetting collision environment to static pointclouds only...")
            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(static_collision_points, point_radius=0.03)
            log.info(
                "Collision environment reset to static pointclouds for remaining placement sequence."
            )

        # 2. Move down for final placement (10cm down) using joint interpolation
        final_place_position = place_pose_matrix[:3, 3] - np.array([0.0, 0.0, 0.10])
        final_place_pose = place_pose_matrix.copy()
        final_place_pose[:3, 3] = final_place_position
        final_position, final_orientation = extract_position_orientation(
            final_place_pose
        )

        log.info(
            "Moving down 10cm for final placement using Cartesian interpolation..."
        )
        result = self.robot.send_cartesian_interpolated_motion(
            final_position, final_orientation, duration=3.0, num_waypoints=25
        )
        if result is None:
            log.warning("Failed to move down for placement.")
            return False

        # This block was previously indented under 'if final_arm_joints is not None:'
        # We need to maintain the logic flow. Since we return False on failure above, we can continue.
        time.sleep(0.2)

        # Open gripper to release object
        log.info("Opening gripper to release object...")
        self.robot.control_gripper(1.0)
        time.sleep(0.2)

        # Detach object from end-effector since it's now placed
        log.info("Detaching object from end-effector...")
        self.robot.detach_objects_from_eef()

        # Move back up to elevated pose using Cartesian interpolation
        log.info("Moving back up to elevated pose...")
        result = self.robot.send_cartesian_interpolated_motion(
            position, orientation, duration=3.0, num_waypoints=20
        )
        if result is None:
            log.warning("Warning: Failed to execute smooth retract motion.")
        time.sleep(0.2)

        log.info("In-place placement sequence completed successfully.")
        return True

    def _create_topdown_pose_matrix(self, position, approach_vector):
        """
        Creates a 4x4 pose matrix for the Fetch gripper using top-down approach.
        This follows the exact same method as point_sampler.py for consistency.

        Args:
            position: 3D position for the gripper
            approach_vector: Approach vector (should be [0, 0, -1] for top-down)

        Returns:
            4x4 pose matrix
        """
        x_axis = approach_vector
        z_axis_ref = np.array([0, 0, 1])
        y_axis = np.cross(z_axis_ref, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.cross(np.array([0, 1, 0]), x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        pose_matrix = np.eye(4)
        pose_matrix[:3, 0] = x_axis
        pose_matrix[:3, 1] = y_axis
        pose_matrix[:3, 2] = z_axis
        pose_matrix[:3, 3] = position
        return pose_matrix
