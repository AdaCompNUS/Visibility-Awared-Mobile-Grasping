#!/usr/bin/env python3
"""
NavPreposeScheduler: A baseline scheduler that uses the PreposeSampler
(same as the main scheduler) but only uses the target base position for
navigation (like NavManipScheduler). This decouples navigation and manipulation.

Flow:
1. Tuck Arm
2. Use PreposeSampler to sample valid (base_config, arm_config) pairs
3. Navigate to only the base_config position (arm stays tucked)
4. Look at the object
5. Perform manipulation (perception + grasp)
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
from grasp_anywhere.samplers.prepose_sampler import PreposeSampler
from grasp_anywhere.stage_planners.grasp_stage import GraspPlanner
from grasp_anywhere.stage_planners.move_stage import MovePlanner
from grasp_anywhere.stage_planners.place_stage import PlacePlanner
from grasp_anywhere.utils.grasp_utils import select_diverse_grasps
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.navigation_utils import get_current_pcd, navigate
from grasp_anywhere.utils.perception_utils import get_pcd_from_mask
from grasp_anywhere.grasping_client.sam_client import SamClient, SamConfig
from grasp_anywhere.utils.seg_utils import mask_from_segmentation_id
from grasp_anywhere.utils.visualization_utils import visualize_grasps

# [torso, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
TUCK_JOINTS = [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0]


class NavPreposeScheduler:
    """
    Baseline scheduler that uses PreposeSampler for target selection but
    decouples navigation and manipulation (Nav -> Manip).
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
        self.grasp_planner = None
        self.place_planner = None
        self.move_planner = None
        self.failed_due_to_reachability = False

        self._initialize_environment_and_planners()

    def _initialize_environment_and_planners(self):
        """
        Initializes the planners and collision environment.
        """
        if not self.robot.is_sim and not rospy.core.is_initialized():
            rospy.init_node("grasp_interface_node", anonymous=True)

        self.robot.set_torso_height(0.3)
        log.info("Fetch robot initialized")

        # Load capability map (used by PreposeSampler for filtering)
        self.robot.capability_map = CapabilityMap.from_file(
            self.config["planning"]["capability_map_path"]
        )
        # Load reachability map (used by PreposeSampler for base filtering)
        self.robot.reachability_map = ReachabilityMap.from_file(
            self.config["planning"]["reachability_map_path"]
        )
        # Load torso map (used by PreposeSampler for torso height selection)
        self.robot.torso_map = TorsoMap.from_file(
            self.config["planning"]["torso_map_path"]
        )

        # Initialize planners
        self.grasp_planner = GraspPlanner(
            self.robot, enable_visualization=self.enable_visualization
        )
        self.place_planner = PlacePlanner(
            self.robot, enable_visualization=self.enable_visualization
        )
        self.move_planner = MovePlanner(self.robot)
        log.info("Planners initialized.")

        # Initialize Grasping Config
        from grasp_anywhere.grasping_client.config import GraspingConfig

        services = self.config.get("services", {})
        self.grasping_config = GraspingConfig(
            url=services.get("graspnet_url", "http://localhost:4003")
        )
        self.sam_client = SamClient(
            SamConfig(url=services.get("sam_url", "http://localhost:4001"))
        )

    def _move_arm_to_tuck(self):
        """Moves the arm to the safe tucked configuration."""

        from grasp_anywhere.utils.motion_utils import move_arm_with_replanning

        log.info("Moving arm to TUCK configuration...")
        result = move_arm_with_replanning(
            self.robot,
            TUCK_JOINTS,
            enable_gaze_control=False,
            enable_replanning=self.enable_replanning,
        )
        if result:
            log.info("Arm tucked successfully.")
            return True
        else:
            log.error("Failed to tuck arm.")
            return False

    def _navigate_to_pose(self, target_base_pose):
        """
        Navigates the robot base to a specific pose using the Base Planner (Hybrid A*).
        Keeps the arm in the fixed TUCK configuration.
        """
        target_base_pose = list(map(float, target_base_pose))
        log.info(f"Navigating to specific base pose: {target_base_pose}")

        success, msg = navigate(
            self.robot,
            target_base_pose,
            TUCK_JOINTS,
            enable_replanning=self.enable_replanning,
        )
        return success, msg

    def _create_prepose_generator(self, prepose_sampler, local_pcd, object_coord):
        """
        Creates a generator that wraps PreposeSampler.sample_and_validate().
        This allows multiple sampling attempts, similar to NavSampler.generator().
        """
        self.failed_due_to_reachability = False
        num_samples = prepose_sampler.num_samples  # Use same num_samples as sampler

        for _ in range(num_samples):
            base_config, arm_config = prepose_sampler.sample_and_validate(
                local_pcd, object_coord
            )

            if base_config is None or arm_config is None:
                # Sampling failed, check if due to reachability
                self.failed_due_to_reachability = (
                    prepose_sampler.failed_due_to_reachability
                )
                return  # Stop generation

            yield base_config, arm_config

    def grasp_anywhere(
        self, object_pcd, max_attempts: int = 1, target_model_id: Optional[str] = None
    ):
        """
        Nav+Manip Scheduler using PreposeSampler for target selection.

        1. Tuck Arm.
        2. Use PreposeSampler to find valid (base_config, arm_config) pairs via generator.
        3. Navigate to only the base_config (Base Motion, Arm Fixed at Tuck).
           If navigation fails, try next sample from generator.
        4. Look at the object (Head Motion).
        5. Detect and Grasp (Arm Motion, Fixed Base).

        Note: max_attempts is kept for API compatibility but not used in this baseline.
        """
        _ = max_attempts  # Suppress unused parameter warning
        # Interpret input strictly as a single coordinate (1x3)
        object_coord = np.asarray(object_pcd, dtype=np.float32).reshape(1, 3)
        log.info(
            f"Starting NAV+PREPOSE grasp_anywhere with target coordinate "
            f"{object_coord.flatten().tolist()}"
        )

        # --- Stage 0: Tuck Arm ---
        log.info("--- Stage 0: Tuck Arm ---")
        if not self._move_arm_to_tuck():
            return False, "TUCK_FAILURE"

        # --- Stage 1: Plan Navigation Target using PreposeSampler ---
        log.info("--- Stage 1: Plan Navigation Target (PreposeSampler) ---")

        # Update collision environment with LOCAL MAP
        local_pcd = get_current_pcd(self.robot)
        if local_pcd is not None:
            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(local_pcd, filter_robot=True)
            log.info(
                "Updated collision environment with local map for prepose planning."
            )
        else:
            local_pcd = np.empty((0, 3))
            log.warning("Could not get local PCD for prepose planning.")

        object_center = object_coord[0]

        # Use PreposeSampler (same as main scheduler)
        prepose_sampler = PreposeSampler(
            robot=self.robot,
            object_center_world=object_center,
            manipulation_radius=self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )

        # Create generator for sampling (similar to NavSampler.generator())
        config_generator = self._create_prepose_generator(
            prepose_sampler, local_pcd, object_coord
        )

        # --- Stage 2: Navigation (Base Motion Only) ---
        # Iterate through valid candidates adaptively (same pattern as NavManipScheduler)
        log.info("--- Stage 2: Navigation ---")

        nav_success = False
        nav_msg = "PLANNING_FAILURE"

        for base_config, arm_config in config_generator:
            log.info(f"PreposeSampler found base config: {base_config}")
            log.info(
                f"PreposeSampler found arm config: {arm_config} (will be ignored for nav)"
            )

            # Navigate to only the base position (ignore arm_config, keep TUCK)
            nav_success, nav_msg = self._navigate_to_pose(base_config)

            if nav_success:
                log.info("Navigation successful.")
                break
            else:
                log.warning(
                    f"Navigation to {base_config} failed: {nav_msg}. Sampling next target..."
                )

        if not nav_success:
            log.error(f"All navigation attempts failed. Last error: {nav_msg}")
            if self.failed_due_to_reachability:
                return False, "OUT_OF_REACHABILITY"
            return False, "NAVIGATION_FAILURE"

        # --- Stage 3: Look At (Head Motion) ---
        log.info("--- Stage 3: Look At Object ---")
        self.robot.point_head_at(object_coord[0].tolist())
        time.sleep(1.0)  # Wait for head to settle

        # --- Stage 4: Manipulation (Grasp) ---
        log.info("--- Stage 4: Manipulation (Grasp) ---")

        # 4a. Perception Update
        snapshot = self.robot.robot_env.get_sensor_snapshot()

        rgb = snapshot["rgb"]
        depth = snapshot["depth"]
        camera_intrinsic = snapshot["intrinsics"]
        joint_names, joint_positions = snapshot["joint_states"]
        joint_dict = dict(zip(joint_names, joint_positions))

        # Recalculate camera pose
        current_extrinsic = self.robot.compute_camera_pose_from_joints(joint_dict)

        # Update Scene
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

        # 4b. Grasp Detection
        # Calculate segmentation mask
        H, W = rgb.shape[:2]
        world_point = object_coord[0]
        T_cam_world = np.linalg.inv(current_extrinsic)
        p_cam_h = T_cam_world @ np.array(
            [world_point[0], world_point[1], world_point[2], 1.0],
            dtype=np.float32,
        )
        z_cam = p_cam_h[2]

        segmap = None
        if z_cam > 0:
            uv = camera_intrinsic @ p_cam_h[:3]
            u, v = int(np.round(uv[0] / z_cam)), int(np.round(uv[1] / z_cam))
            if 0 <= u < W and 0 <= v < H:
                # Use SAM or ground-truth segmentation
                if target_model_id is not None and "segmentation" in snapshot:
                    scene = self.robot.robot_env.env.unwrapped.scene
                    seg_id = None
                    for actor in scene.actors.values():
                        if target_model_id in actor.name:
                            seg_id = actor._objs[0].per_scene_id
                            break
                    if seg_id is not None:
                        segmap = mask_from_segmentation_id(
                            snapshot["segmentation"], seg_id
                        )
                    else:
                        segmap = self.sam_client.segment_point(rgb, (u, v))
                elif "segmentation" in snapshot:
                    segmap = segment_object(rgb, (u, v))
                else:
                    segmap = segment_object(rgb, (u, v))

        if segmap is None or np.count_nonzero(segmap) == 0:
            log.warning("Could not segment object. Perception Failure.")
            return False, "PERCEPTION_FAILURE"

        segmap = segmap.astype(np.uint8)

        # Prepare point clouds
        full_pcd_cam, segment_pcd_cam = self._prepare_pointclouds_for_grasping(
            depth, segmap, camera_intrinsic, current_extrinsic, joint_dict
        )

        # Predict Grasps
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
            log.warning("No grasps detected.")
            return False, "PERCEPTION_FAILURE"

        if self.enable_visualization:
            visualize_grasps(pred_grasps_cam, scores, rgb, depth, camera_intrinsic)

        grasp_poses_world = [current_extrinsic @ g for g in pred_grasps_cam]

        # 4c. Execute Grasp (No Base Motion)
        diverse_indices = select_diverse_grasps(grasp_poses_world, 3)

        for idx in diverse_indices:
            pose = grasp_poses_world[idx]
            log.info(f"Attempting grasp execution (Rank {idx})")

            success, msg = self.grasp_planner.run(
                pose,
                current_extrinsic,
                self.robot.scene.current_environment(),
                use_active_perception=False,
            )

            if success:
                log.info("Grasp successful!")
                return True, "GRASP_SUCCESS"
            else:
                log.warning(f"Grasp execution failed: {msg}")

        log.error("All grasp attempts failed.")
        return False, "GRASP_EXECUTION_FAILED"

    def _prepare_pointclouds_for_grasping(
        self, depth, segmap, K, camera_extrinsic, joint_dict
    ):
        """Prepare full and segment point clouds for grasp prediction."""
        # 1. Full PCD
        full_mask = np.ones(depth.shape, dtype=bool)
        full_pcd_cam = get_pcd_from_mask(depth, full_mask, K)

        if full_pcd_cam is None or len(full_pcd_cam) == 0:
            # Return empty arrays if no points
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
