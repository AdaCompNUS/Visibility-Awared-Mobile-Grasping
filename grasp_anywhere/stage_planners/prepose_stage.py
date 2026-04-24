#!/usr/bin/env python3
from typing import Any

import numpy as np

from grasp_anywhere.samplers.prepose_sampler import PreposeSampler
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning


class PreposePlanner:
    """
    Handles the first stage of grasping: moving to a pre-pose.
    This class is now a pure planner, taking a segmented object point cloud
    and finding a valid pre-pose for the robot.
    """

    def __init__(
        self,
        robot,
        manipulation_radius,
        max_ik_attempts=20,
        max_replan_attempts=5,
        enable_visualization=False,
    ):
        """
        Initializes the PreposePlanner.
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
        self.failed_due_to_reachability = False
        log.info("PreposePlanner initialized.")

    def execute_prepose(
        self,
        config_generator,
        enable_replanning=True,
        enable_pcd_alignment=True,
    ):
        """
        Moves the robot to the specified pre-pose using whole-body motion planning
        to a target configuration (arm + base).
        """
        log.info("Moving to pre-pose (config) with whole-body replanning...")

        for target_base_config, target_arm_config in config_generator:
            success, msg = move_to_config_with_replanning(
                self.robot,
                goal_joints=target_arm_config,
                goal_base=target_base_config,
                enable_replanning=enable_replanning,
                enable_pcd_alignment=enable_pcd_alignment,
                max_replan_attempts=self.max_replan_attempts,
            )
            if msg == "TARGET_IN_COLLISION":
                log.info("Target is in collision, trying next sample.")
                continue
            return success
        return False

    def plan(self, object_pcd_world, camera_pose, combined_points):
        """
        Plans the pre-pose for a given segmented view.

        Args:
            object_pcd_world (np.ndarray): The point cloud of the object in the world frame.
            camera_pose (np.ndarray): The 4x4 camera pose matrix in the world frame.
            combined_points (np.ndarray | Callable[[], np.ndarray]):
                Either a static environment point cloud or a zero-arg callable
                that returns the latest scene/environment point cloud when called.

        Returns:
            A tuple of (generator, object_center) where generator yields (base_config, arm_config)
            or (None, None) if no plan is found.
        """
        if object_pcd_world is None or len(object_pcd_world) == 0:
            print("ERROR: Cannot plan pre-pose, object point cloud is empty.")
            return None, None

        # Calculate the center of the object point cloud
        object_center_world = np.mean(object_pcd_world, axis=0)
        print(f"Planning pre-pose for object at center: {object_center_world}")

        # Use the PreposeSampler to find a valid pre-pose configuration.
        sampler = PreposeSampler(
            robot=self.robot,
            object_center_world=object_center_world,
            manipulation_radius=self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )

        def _get_scene_points(cp: Any):
            # Support both static arrays and provider callables
            try:
                if callable(cp):
                    return cp()
            except Exception:
                pass
            return cp

        def config_generator():
            self.failed_due_to_reachability = False
            for _ in range(self.max_replan_attempts):
                current_scene_points = _get_scene_points(combined_points)
                if current_scene_points is None or len(current_scene_points) == 0:
                    log.warning(
                        "Scene point cloud empty/unavailable; skipping this sampling iteration."
                    )
                    continue
                base_config, arm_config = sampler.sample_and_validate(
                    current_scene_points, object_pcd_world
                )

                if base_config is None or arm_config is None:
                    self.failed_due_to_reachability = sampler.failed_due_to_reachability
                    return  # Stop generation

                print("Successfully found a valid pre-pose configuration.")
                yield base_config, arm_config

        return config_generator(), object_center_world
