#!/usr/bin/env python3
import time

import numpy as np

from grasp_anywhere.samplers.point_sampler import PointSampler
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning


class PointPlanner:
    """
    Handles moving the robot to point at an object.
    This class is a pure planner, taking a segmented object point cloud
    and finding a valid pointing pose for the robot.
    """

    def __init__(
        self,
        robot,
        manipulation_radius,
        max_replan_attempts=5,
        enable_visualization=False,
    ):
        """
        Initializes the PointPlanner.
        Args:
            robot: An instance of the Fetch class.
            manipulation_radius (float): The radius around the object for pre-pose sampling.
            max_replan_attempts (int): Max attempts for replanning due to collisions or timeouts.
            enable_visualization (bool): Whether to show visualization during planning.
            logger (callable, optional): Logger function to use. Defaults to print.
        """
        self.robot = robot
        self.manipulation_radius = manipulation_radius
        self.max_replan_attempts = max_replan_attempts
        self.enable_visualization = enable_visualization
        log.info("PointPlanner initialized.")

    def execute(
        self,
        base_config,
        arm_config,
        collision_points,
        enable_replanning=True,
        enable_pcd_alignment=True,
    ):
        """
        Moves the robot to the specified pointing pose.
        """
        log.info("Closing gripper to point.")
        self.robot.control_gripper(0.0)
        time.sleep(1.0)

        current_base_config = self.robot.get_base_params()
        if np.allclose(base_config, current_base_config, atol=0.03):
            log.info("Already at target base config. Executing arm motion.")
            result = self.robot.send_joint_values(arm_config, duration=5.0)
            time.sleep(0.5)
            if result is None:
                log.info("Failed to execute arm motion.")
                return False
            return True

        log.info("Fixed-base IK failed. Falling back to whole-body motion planning.")
        success = move_to_config_with_replanning(
            self.robot,
            arm_config,
            base_config,
            enable_replanning=enable_replanning,
            enable_pcd_alignment=enable_pcd_alignment,
            max_replan_attempts=self.max_replan_attempts,
        )
        return success

    def plan(self, object_pcd_world, camera_pose, combined_points):
        """
        Executes the pointing planning pipeline for a given segmented view.

        Args:
            object_pcd_world (np.ndarray): The point cloud of the object in the world frame.
            camera_pose (np.ndarray): The 4x4 camera pose matrix in the world frame.
            combined_points (np.ndarray): The full collision point cloud of the environment.

        Returns:
            Tuple of (4x4 point pose matrix, arm_seed, object_center) or (None, None, None) if no pose is found.
        """
        if object_pcd_world is None or len(object_pcd_world) == 0:
            print("ERROR: Cannot plan pointing, object point cloud is empty.")
            return None, None, None

        if camera_pose is None:
            print("ERROR: Could not get camera pose for pointing planning.")
            return None, None, None

        object_center_world = np.mean(object_pcd_world, axis=0)
        print(f"Planning pointing for object at center: {object_center_world}")

        sampler = PointSampler(
            robot=self.robot,
            object_center_world=object_center_world,
            manipulation_radius=self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )

        base_config, arm_config = sampler.sample_and_validate(
            camera_pose, combined_points, object_pcd_world
        )

        return base_config, arm_config, object_center_world
