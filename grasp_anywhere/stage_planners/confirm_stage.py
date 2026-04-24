#!/usr/bin/env python3
import time

import numpy as np

from grasp_anywhere.samplers.point_sampler import PointSampler
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning


class ConfirmPlanner:
    """
    Handles moving the robot to point at an object and ask a question.
    """

    def __init__(
        self,
        robot,
        manipulation_radius,
        max_ik_attempts=20,
        max_replan_attempts=5,
        enable_visualization=False,
        logger=None,
    ):
        """
        Initializes the ConfirmPlanner.
        Args:
            robot: An instance of the Fetch class.
            manipulation_radius (float): The radius around the object for pre-pose sampling.
            max_ik_attempts (int): Max attempts for the IK solver.
            max_replan_attempts (int): Max attempts for replanning due to collisions or timeouts.
            enable_visualization (bool): Whether to show visualization during planning.
            logger (callable, optional): Logger function to use. Defaults to print.
        """
        self.robot = robot
        self.manipulation_radius = manipulation_radius
        self.max_ik_attempts = max_ik_attempts
        self.max_replan_attempts = max_replan_attempts
        self.enable_visualization = enable_visualization

        log.info("ConfirmPlanner initialized.")

    def execute(
        self,
        base_config,
        arm_config,
        enable_replanning=True,
        enable_pcd_alignment=True,
    ):
        """
        Moves the robot to the specified pointing pose.
        """
        log.info("Closing gripper to point.")
        self.robot.control_gripper(0.0)
        time.sleep(1.0)

        log.info("Moving to pointing pose...")

        success, _ = move_to_config_with_replanning(
            self.robot,
            arm_config,
            base_config,
            enable_replanning=enable_replanning,
            enable_pcd_alignment=enable_pcd_alignment,
            max_replan_attempts=self.max_replan_attempts,
        )
        return success

    def plan(
        self,
        object_pcd,
        camera_pose,
        collision_points,
    ):
        """
        Plans the pointing motion.
        """
        # Pointing logic
        if object_pcd is None or len(object_pcd) == 0:
            return False, "Cannot plan pointing, object point cloud is empty.", None

        if camera_pose is None:
            return False, "Could not get camera pose for pointing planning.", None

        object_center_world = np.mean(object_pcd, axis=0)
        print(f"Planning pointing for object at center: {object_center_world}")

        sampler = PointSampler(
            robot=self.robot,
            object_center_world=object_center_world,
            manipulation_radius=self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )

        base_config, arm_config = sampler.sample_and_validate(
            camera_pose, collision_points, object_pcd
        )

        if base_config is None:
            return False, "Failed to find a valid pointing pose after sampling.", None

        print("Successfully found a valid pointing pose.")
        return (
            True,
            "Successfully found a valid pointing pose.",
            (base_config, arm_config),
        )
