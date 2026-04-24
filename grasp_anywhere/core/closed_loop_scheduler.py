#!/usr/bin/env python3
"""
ClosedLoopScheduler: A minimal baseline scheduler with pure closed-loop planning.

This baseline has minimum heuristics:
- No observation stage
- Only prepose → grasp loop
- Relies on resampling of prepose module for recovery

Flow (repeated until success or max_attempts):
1. Pre-pose (sample and move to prepose)
2. Look at object
3. Grasp attempt
4. If failed, go back to step 1 (resample prepose)
"""
import time
from typing import Optional

import numpy as np
import rospy
import yaml

from grasp_anywhere.dataclass.reachability_map import (
    CapabilityMap,
    ReachabilityMap,
)
from grasp_anywhere.dataclass.torso_map import TorsoMap
from grasp_anywhere.grasping_client.grasp_generation import predict_grasps
from grasp_anywhere.stage_planners.grasp_stage import GraspPlanner
from grasp_anywhere.stage_planners.prepose_stage import PreposePlanner
from grasp_anywhere.utils.grasp_utils import select_diverse_grasps
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.perception_utils import get_pcd_from_mask
from grasp_anywhere.grasping_client.sam_client import SamClient, SamConfig
from grasp_anywhere.utils.seg_utils import mask_from_segmentation_id
from grasp_anywhere.utils.visualization_utils import visualize_grasps


class ClosedLoopScheduler:
    """
    Minimal closed-loop scheduler: Pre-pose -> Grasp loop only.
    No observation stage, relies on prepose resampling for recovery.
    """

    def __init__(self, robot, config_path="grasp_anywhere/configs/real_fetch.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.manipulation_radius = self.config["planning"]["manipulation_radius"]
        self.enable_replanning = self.config["planning"]["enable_replanning"]
        self.enable_icp_refinement = self.config["planning"]["enable_icp_refinement"]
        self.enable_pcd_alignment = self.config["planning"]["enable_pcd_alignment"]
        self.enable_visualization = self.config["debug"]["enable_visualization"]

        self.robot = robot
        self.prepose_planner = None
        self.grasp_planner = None

        self._initialize_environment_and_planners()

    def _initialize_environment_and_planners(self):
        """
        Initializes the planners and collision environment.
        """
        if not self.robot.is_sim and not rospy.core.is_initialized():
            rospy.init_node("grasp_interface_node", anonymous=True)

        self.robot.set_torso_height(0.3)
        log.info("Fetch robot initialized")

        # Load capability map (used by PreposePlanner for filtering)
        self.robot.capability_map = CapabilityMap.from_file(
            self.config["planning"]["capability_map_path"]
        )
        # Load reachability map (used by PreposePlanner for base filtering)
        self.robot.reachability_map = ReachabilityMap.from_file(
            self.config["planning"]["reachability_map_path"]
        )
        # Load torso map (used by PreposePlanner for torso height selection)
        self.robot.torso_map = TorsoMap.from_file(
            self.config["planning"]["torso_map_path"]
        )

        # Initialize planners - only prepose and grasp
        self.prepose_planner = PreposePlanner(
            self.robot,
            self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )
        self.grasp_planner = GraspPlanner(
            self.robot, enable_visualization=self.enable_visualization
        )
        log.info("Planners initialized (PreposePlanner + GraspPlanner only).")

        # Initialize Grasping Config
        from grasp_anywhere.grasping_client.config import GraspingConfig

        services = self.config.get("services", {})
        self.grasping_config = GraspingConfig(
            url=services.get("graspnet_url", "http://localhost:4003")
        )
        self.sam_client = SamClient(
            SamConfig(url=services.get("sam_url", "http://localhost:4001"))
        )

    def grasp_anywhere(
        self, object_pcd, max_attempts: int = 5, target_model_id: Optional[str] = None
    ):
        """
        Minimal closed-loop grasping: Pre-pose -> Grasp loop.

        No observation stage. On failure, resample prepose and retry.
        This represents pure closed-loop planning with minimum heuristics.
        """
        object_coord = np.asarray(object_pcd, dtype=np.float32).reshape(1, 3)
        log.info(
            f"Starting CLOSED-LOOP grasp_anywhere with target coordinate "
            f"{object_coord.flatten().tolist()}"
        )

        # Track failure reasons
        attempt_outcomes = []

        for attempt in range(max_attempts):
            log.info(f"--- Attempt {attempt + 1}/{max_attempts} ---")

            # --- Stage 1: Pre-Pose ---
            log.info("Stage 1: Pre-Pose")

            # Get sensor snapshot and update scene
            snapshot = self.robot.robot_env.get_sensor_snapshot()
            depth = snapshot["depth"]
            camera_intrinsic = snapshot["intrinsics"]
            joint_names, joint_positions = snapshot["joint_states"]
            joint_dict = dict(zip(joint_names, joint_positions))

            current_extrinsic = self.robot.compute_camera_pose_from_joints(joint_dict)

            # Update scene
            self.robot.scene.update(
                depth,
                camera_intrinsic,
                current_extrinsic,
                joint_dict,
                enable_icp_alignment=self.enable_pcd_alignment,
            )
            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(
                self.robot.scene.current_environment(),
                filter_robot=True,
                point_radius=0.03,
            )

            # Plan prepose
            config_generator, object_center = self.prepose_planner.plan(
                object_coord,
                current_extrinsic,
                self.robot.scene.current_environment,
            )

            prepose_success = False
            if config_generator is not None:
                prepose_success = self.prepose_planner.execute_prepose(
                    config_generator,
                    self.enable_replanning,
                    self.enable_pcd_alignment,
                )

            if not prepose_success:
                log.warning("Pre-pose failed. Will resample on next attempt.")
                if getattr(self.prepose_planner, "failed_due_to_reachability", False):
                    attempt_outcomes.append("OUT_OF_REACHABILITY")
                else:
                    attempt_outcomes.append("PLANNING_FAILURE")
                continue  # Retry with new prepose sample

            log.info("Pre-pose successful.")
            time.sleep(1.0)

            # --- Stage 2: Look At Object ---
            log.info("Stage 2: Look At Object")
            self.robot.point_head_at(
                np.asarray(object_center, dtype=np.float32).tolist()
            )
            time.sleep(1.5)

            # --- Stage 3: Grasp ---
            log.info("Stage 3: Grasp")

            # Get fresh sensor data
            snapshot = self.robot.robot_env.get_sensor_snapshot()
            rgb = snapshot["rgb"]
            depth = snapshot["depth"]
            camera_intrinsic = snapshot["intrinsics"]
            segmentation = snapshot.get("segmentation")
            joint_names, joint_positions = snapshot["joint_states"]
            joint_dict = dict(zip(joint_names, joint_positions))

            current_extrinsic = self.robot.compute_camera_pose_from_joints(joint_dict)

            # Update scene
            self.robot.scene.update(
                depth,
                camera_intrinsic,
                current_extrinsic,
                joint_dict,
                enable_icp_alignment=self.enable_pcd_alignment,
            )
            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(
                self.robot.scene.current_environment(),
                filter_robot=True,
                point_radius=0.03,
            )

            # Segment object
            segmap = self._get_segmap(
                rgb,
                depth,
                camera_intrinsic,
                current_extrinsic,
                object_coord,
                segmentation,
                target_model_id,
            )

            if segmap is None or np.count_nonzero(segmap) == 0:
                log.warning("Segmentation failed. Will retry with new prepose.")
                attempt_outcomes.append("PERCEPTION_FAILURE")
                continue

            # Prepare point clouds
            full_pcd_cam, segment_pcd_cam = self._prepare_pointclouds_for_grasping(
                depth, segmap, camera_intrinsic, current_extrinsic, joint_dict
            )

            # Predict grasps
            pred_grasps_cam, scores = predict_grasps(
                self.grasping_config,
                full_pc=full_pcd_cam,
                segment_pc=segment_pcd_cam,
                rgb=rgb,
                depth=depth,
                segmap=segmap,
                K=camera_intrinsic,
                visualize=self.enable_visualization,
            )

            if pred_grasps_cam is None or len(pred_grasps_cam) == 0:
                log.warning("No grasps detected. Will retry with new prepose.")
                attempt_outcomes.append("PERCEPTION_FAILURE")
                continue

            if self.enable_visualization:
                visualize_grasps(pred_grasps_cam, scores, rgb, depth, camera_intrinsic)

            grasp_poses_world = [current_extrinsic @ g for g in pred_grasps_cam]

            # Try diverse grasps
            diverse_indices = select_diverse_grasps(grasp_poses_world, 3)

            for idx in diverse_indices:
                pose = grasp_poses_world[idx]
                log.info(f"Attempting grasp (Rank {idx})")

                success, msg = self.grasp_planner.run(
                    pose,
                    current_extrinsic,
                    self.robot.scene.current_environment(),
                )

                if success:
                    log.info(f"Grasp successful on attempt {attempt + 1}!")
                    return True, "GRASP_SUCCESS"
                else:
                    log.warning(f"Grasp failed: {msg}")
                    if "Failed to find IK" in msg:
                        attempt_outcomes.append("IK_FAILED")
                    else:
                        attempt_outcomes.append("GRASP_EXECUTION_FAILED")

            # All grasps for this prepose failed, will resample prepose
            log.warning("All grasp attempts failed. Will resample prepose.")

        # All attempts exhausted
        log.error(f"All {max_attempts} attempts failed. History: {attempt_outcomes}")

        # Return most relevant failure reason
        if "GRASP_EXECUTION_FAILED" in attempt_outcomes:
            return False, "GRASP_EXECUTION_FAILED"
        if "IK_FAILED" in attempt_outcomes:
            return False, "IK_FAILED"
        if "PERCEPTION_FAILURE" in attempt_outcomes:
            return False, "PERCEPTION_FAILURE"
        if "OUT_OF_REACHABILITY" in attempt_outcomes:
            return False, "OUT_OF_REACHABILITY"
        if "PLANNING_FAILURE" in attempt_outcomes:
            return False, "PLANNING_FAILURE"

        return False, "PLANNING_FAILURE"

    def _get_segmap(
        self,
        rgb,
        depth,
        K,
        camera_extrinsic,
        object_coord,
        segmentation,
        target_model_id,
    ):
        """Get segmentation mask for the target object."""
        H, W = rgb.shape[:2]
        world_point = object_coord[0]
        T_cam_world = np.linalg.inv(camera_extrinsic)
        p_cam_h = T_cam_world @ np.array(
            [world_point[0], world_point[1], world_point[2], 1.0],
            dtype=np.float32,
        )
        z_cam = p_cam_h[2]
        if z_cam <= 0:
            return None

        uv = K @ p_cam_h[:3]
        u = int(np.round(uv[0] / z_cam))
        v = int(np.round(uv[1] / z_cam))

        if not (0 <= u < W and 0 <= v < H):
            return None

        mask = None
        if target_model_id is not None and segmentation is not None:
            scene = self.robot.robot_env.env.unwrapped.scene
            seg_id = None
            for actor in scene.actors.values():
                if target_model_id in actor.name:
                    seg_id = actor._objs[0].per_scene_id
                    break
            if seg_id is not None:
                mask = mask_from_segmentation_id(segmentation, seg_id)
            else:
                mask = self.sam_client.segment_point(rgb, (u, v))
        else:
            mask = segment_object(rgb, (u, v))

        if mask is not None and np.count_nonzero(mask) > 0:
            return mask.astype(np.uint8)
        return None

    def _prepare_pointclouds_for_grasping(
        self, depth, segmap, K, camera_extrinsic, joint_dict
    ):
        """Prepare full and segment point clouds for grasp prediction."""
        # 1. Full PCD
        full_mask = np.ones(depth.shape, dtype=bool)
        full_pcd_cam = get_pcd_from_mask(depth, full_mask, K)

        if full_pcd_cam is None or len(full_pcd_cam) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform to world to filter
        full_pcd_cam_h = np.hstack((full_pcd_cam, np.ones((full_pcd_cam.shape[0], 1))))
        full_pcd_world = (camera_extrinsic @ full_pcd_cam_h.T).T[:, :3]

        # Filter robot
        filtered_world_list, _ = self.robot.filter_points_on_robot_with_state(
            full_pcd_world, joint_dict, point_radius=0.04
        )
        filtered_world = np.array(filtered_world_list)

        # Back to camera
        if len(filtered_world) == 0:
            full_pcd_cam_filtered = np.empty((0, 3))
        else:
            filtered_world_h = np.hstack(
                (filtered_world, np.ones((filtered_world.shape[0], 1)))
            )
            T_cam_world = np.linalg.inv(camera_extrinsic)
            full_pcd_cam_filtered = (T_cam_world @ filtered_world_h.T).T[:, :3]

        # 2. Segment PCD
        object_pcd_cam = get_pcd_from_mask(depth, segmap.astype(bool), K)
        if object_pcd_cam is None:
            object_pcd_cam = np.empty((0, 3))

        return full_pcd_cam_filtered, object_pcd_cam
