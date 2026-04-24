#!/usr/bin/env python3
import time

import numpy as np

from grasp_anywhere.robot.ik import trac_solve_arm_only
from grasp_anywhere.robot.utils.transform_utils import extract_position_orientation
from grasp_anywhere.samplers.point_sampler import PointSampler
from grasp_anywhere.utils.logger import log


class StaticPointPlanner:
    """
    Handles moving the robot to point at static table-top objects using only arm movement.
    This planner is designed for in-place confirmation of objects without moving the base
    or rotating the camera. It only uses fixed-base IK and fails if no solution is found.
    """

    def __init__(
        self,
        robot,
        manipulation_radius,
        enable_visualization=False,
    ):
        """
        Initializes the StaticPointPlanner.
        Args:
            robot: An instance of the Fetch class.
            manipulation_radius (float): The radius around the object for pre-pose sampling.
            enable_visualization (bool): Whether to show visualization during planning.
            logger (callable, optional): Logger function to use. Defaults to print.
        """
        self.robot = robot
        self.manipulation_radius = manipulation_radius
        self.enable_visualization = enable_visualization
        log.info("StaticPointPlanner initialized.")

    def execute_point_static(self, point_pose_matrix):
        """
        Moves the robot to the specified pointing pose using ONLY fixed-base IK.
        No whole-body motion planning fallback is used.

        Args:
            point_pose_matrix (np.ndarray): The 4x4 pose matrix for the pointing pose.

        Returns:
            bool: True if successful, False if fixed-base IK fails.
        """
        log.info("Closing gripper to point.")
        self.robot.control_gripper(0.0)
        time.sleep(1.0)

        log.info("Attempting static pointing with fixed-base IK only...")
        position, orientation = extract_position_orientation(point_pose_matrix)

        # Get current robot state
        base_config = self.robot.get_base_params()
        torso_pos = self.robot.get_torso_position()
        current_arm_joints = self.robot.get_arm_joint_values()

        # Try fixed-base IK only
        arm_joints = trac_solve_arm_only(
            current_arm_joints, base_config, torso_pos, position, orientation
        )

        if arm_joints:
            log.info("Found IK solution with fixed base. Executing static arm motion.")
            result = self.robot.send_joint_values(
                [torso_pos] + arm_joints, duration=5.0
            )
            if result is None:
                log.info("Failed to execute static arm motion.")
                return False
            return True
        else:
            log.info("Fixed-base IK failed for static pointing. No fallback used.")
            return False

    def run(self, object_pcd_world, camera_pose, combined_points):
        """
        Executes the static pointing planning pipeline for a given segmented view.

        Args:
            object_pcd_world (np.ndarray): The point cloud of the object in the world frame.
            camera_pose (np.ndarray): The 4x4 camera pose matrix in the world frame.
            combined_points (np.ndarray): The full collision point cloud of the environment.

        Returns:
            Tuple of (4x4 point pose matrix, arm_seed, object_center) or (None, None, None) if no pose is found.
        """
        if object_pcd_world is None or len(object_pcd_world) == 0:
            log.error(
                "ERROR: Cannot plan static pointing, object point cloud is empty."
            )
            return None, None, None

        if camera_pose is None:
            log.error("ERROR: Could not get camera pose for static pointing planning.")
            return None, None, None

        object_center_world = np.mean(object_pcd_world, axis=0)
        log.info(
            f"Planning static pointing for object at center: {object_center_world}"
        )

        # calculate the object height
        # object_height = np.max(object_pcd_world, axis=0)[2] - np.min(object_pcd_world, axis=0)[2]
        # sample_hand_radius = object_height + 0.1

        sampler = PointSampler(
            robot=self.robot,
            object_center_world=object_center_world,
            manipulation_radius=self.manipulation_radius,
            enable_visualization=self.enable_visualization,
            radius=0.15,
        )

        point_pose_matrix, arm_seed = sampler.sample_and_validate(
            camera_pose, combined_points, object_pcd_world
        )

        if point_pose_matrix is None:
            log.error(
                "ERROR: Failed to find a valid static pointing pose after sampling."
            )
            return None, None, object_center_world

        log.info("Successfully found a valid static pointing pose.")
        return point_pose_matrix, arm_seed, object_center_world
