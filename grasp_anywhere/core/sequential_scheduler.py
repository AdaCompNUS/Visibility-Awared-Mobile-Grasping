#!/usr/bin/env python3
import time

import numpy as np
import rospy
import yaml

from grasp_anywhere.data_collector.trajectory_collector import (
    episode_finish,
    episode_record,
)
from grasp_anywhere.dataclass.datacollector.config import DataCollectionConfig
from grasp_anywhere.dataclass.reachability_map import (
    CapabilityMap,
    ReachabilityMap,
)
from grasp_anywhere.dataclass.torso_map import TorsoMap
from grasp_anywhere.grasping_client.grasp_generation import predict_grasps
from grasp_anywhere.grasping_client.sam_client import SamClient, SamConfig
from grasp_anywhere.stage_planners.confirm_stage import ConfirmPlanner
from grasp_anywhere.stage_planners.grasp_stage import GraspPlanner
from grasp_anywhere.stage_planners.move_stage import MovePlanner
from grasp_anywhere.stage_planners.place_stage import PlacePlanner
from grasp_anywhere.stage_planners.point_stage import PointPlanner
from grasp_anywhere.stage_planners.prepose_stage import PreposePlanner
from grasp_anywhere.stage_planners.static_point_stage import StaticPointPlanner
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.perception_utils import get_pcd_from_mask
from grasp_anywhere.utils.seg_utils import mask_from_segmentation_id, mask_from_uv
from grasp_anywhere.utils.visualization_utils import (
    visualize_grasps,
)


class SequentialScheduler:
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
        self.place_planner = None
        self.move_planner = None
        self.point_planner = None
        self.confirm_planner = None
        self.static_point_planner = None

        self._initialize_environment_and_planners()

    def _initialize_environment_and_planners(self):
        """
        Initializes the planners, and collision environment.
        """
        if not self.robot.is_sim and not rospy.core.is_initialized():
            rospy.init_node("grasp_interface_node", anonymous=True)

        self.robot.set_torso_height(0.3)
        log.info("Fetch robot initialized")

        # Load capability map once (used for pre-pose filtering)
        self.robot.capability_map = CapabilityMap.from_file(
            self.config["planning"]["capability_map_path"]
        )
        # Load reachability map once (used to filter low-reachability base samples)
        self.robot.reachability_map = ReachabilityMap.from_file(
            self.config["planning"]["reachability_map_path"]
        )
        # Load torso map once (used to choose torso height before IKFast)
        self.robot.torso_map = TorsoMap.from_file(
            self.config["planning"]["torso_map_path"]
        )

        # Initialize planners
        self.prepose_planner = PreposePlanner(
            self.robot,
            self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )
        self.grasp_planner = GraspPlanner(
            self.robot, enable_visualization=self.enable_visualization
        )
        self.place_planner = PlacePlanner(
            self.robot, enable_visualization=self.enable_visualization
        )
        self.move_planner = MovePlanner(self.robot)
        self.point_planner = PointPlanner(
            self.robot,
            self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )
        self.confirm_planner = ConfirmPlanner(
            self.robot,
            self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )
        self.static_point_planner = StaticPointPlanner(
            self.robot,
            self.manipulation_radius,
            enable_visualization=self.enable_visualization,
        )
        log.info("Planners initialized.")

        # Initialize Grasping Config
        from grasp_anywhere.grasping_client.config import GraspingConfig

        services = self.config.get("services", {})
        self.grasping_config = GraspingConfig(
            url=services.get("graspnet_url", "http://localhost:4003")
        )

        # Initialize SAM Client
        self.sam_client = SamClient(
            SamConfig(url=services.get("sam_url", "http://localhost:4001"))
        )

    def grasp_anywhere(
        self, object_pcd, max_attempts: int = 1, target_model_id: str = None
    ):
        """
        Simple baseline: Pre-pose -> Grasp. No retries, no behavior tree.
        """
        # Interpret input strictly as a single coordinate (1x3)
        object_coord = np.asarray(object_pcd, dtype=np.float32).reshape(1, 3)
        log.info(
            f"Starting sequential grasp_anywhere"
            f" (Observe -> Pre-pose -> Grasp) with target coordinate {object_coord.flatten().tolist()}"
        )

        self.robot._active_episode_cfg = DataCollectionConfig(
            run_name="grasp_episode",
        )
        self.robot._active_episode_ds = None
        self.robot._active_episode_goal_xyz_world = object_coord[0].astype(np.float32)

        # --- Stage 0: Observe ---
        log.info("Starting OBSERVE stage.")
        # Try to move to a good observation point
        obs_success, obs_msg = self.move_planner.run_observe_object(
            object_coord,
            self.robot.scene.current_environment(),
            1.5 * self.manipulation_radius,
            self.enable_replanning,
            self.enable_pcd_alignment,
        )
        if not obs_success:
            log.warning(f"Observe stage failed: {obs_msg}")
            # Depending on strictness, we might fallback or fail.
            # Given sequential instructions, we fail.
            self._cleanup_episode()
            return False, "OBSERVE_FAILURE"

        # --- Stage 1: Pre-Pose ---
        log.info("Starting PRE-POSE stage.")

        # Snapshot for pre-pose environment update
        snapshot = self.robot.robot_env.get_sensor_snapshot()
        self._record_snapshot(snapshot)  # Record for data collection

        rgb = snapshot["rgb"]
        depth = snapshot["depth"]
        camera_intrinsic = snapshot["intrinsics"]
        joint_names, joint_positions = snapshot["joint_states"]
        joint_dict = dict(zip(joint_names, joint_positions))

        # Compute camera pose from joints (FK) just to be sure it matches snapshot
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

        # Plan Pre-pose
        config_generator, object_center = self.prepose_planner.plan(
            object_coord,
            current_extrinsic,
            self.robot.scene.current_environment,
        )

        prepose_successful = False
        if config_generator is not None:
            prepose_successful = self.prepose_planner.execute_prepose(
                config_generator, self.enable_replanning, self.enable_pcd_alignment
            )

        if not prepose_successful:
            log.warning("Pre-pose stage failed. Aborting.")
            self._cleanup_episode()
            if getattr(self.prepose_planner, "failed_due_to_reachability", False):
                return False, "OUT_OF_REACHABILITY"
            return False, "PLANNING_FAILURE"

        log.info("Pre-pose successful. Proceeding to GRASP stage.")
        time.sleep(2.0)
        self.robot.point_head_at(np.asarray(object_center, dtype=np.float32).tolist())
        rospy.sleep(2.0)

        # --- Stage 2: Grasp ---
        log.info("Starting GRASP stage.")

        # New snapshot after moving
        snapshot = self.robot.robot_env.get_sensor_snapshot()
        self._record_snapshot(snapshot)  # Record

        rgb = snapshot["rgb"]
        depth = snapshot["depth"]
        camera_intrinsic = snapshot["intrinsics"]
        if "segmentation" in snapshot:
            segmentation = snapshot["segmentation"]
        else:
            segmentation = None
        joint_names, joint_positions = snapshot["joint_states"]
        joint_dict = dict(zip(joint_names, joint_positions))

        current_extrinsic = self.robot.compute_camera_pose_from_joints(joint_dict)

        # Update scene again
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

        # Segmentation Logic (SAM or ID)
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
            log.warning("Segmentation failed. Aborting.")
            self._cleanup_episode()
            return False, "PERCEPTION_FAILURE"

        # Prepare point clouds for grasp prediction
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
            log.warning("No grasps predicted. Aborting.")
            self._cleanup_episode()
            return False, "PERCEPTION_FAILURE"

        if self.enable_visualization:
            visualize_grasps(pred_grasps_cam, scores, rgb, depth, camera_intrinsic)

        # Transform to world
        grasp_poses_world = [current_extrinsic @ g for g in pred_grasps_cam]

        # Simple selection: Take the best scoring grasp
        # Use argsort to find index of max score. Scores are usually confidence 0-1.
        # Assuming scores is a list or array.
        best_idx = np.argmax(scores)
        target_pose = grasp_poses_world[best_idx]
        log.info(f"Selected grasp with score {scores[best_idx]}")

        # Execute Grasp
        success, msg = self.grasp_planner.run(
            target_pose,
            current_extrinsic,
            self.robot.scene.current_environment(),
        )

        if success:
            log.info("Grasp execution successful.")
            _ = episode_finish(
                self.robot._active_episode_ds,
                cfg=self.robot._active_episode_cfg,
                dt=1.0 / float(self.robot._active_episode_cfg.record_hz),
            )
            self._cleanup_episode()
            return True, f"Grasp successful: {msg}"
        else:
            log.warning(f"Grasp execution failed: {msg}")
            self._cleanup_episode()
            if "Failed to find IK" in msg:
                return False, "IK_FAILED"
            return False, "GRASP_EXECUTION_FAILED"

    def _cleanup_episode(self):
        self.robot._active_episode_cfg = None
        self.robot._active_episode_ds = None
        self.robot._active_episode_goal_xyz_world = None

    def _record_snapshot(self, snapshot):
        rgb = snapshot["rgb"]
        depth = snapshot["depth"]
        # joint_names, joint_positions = snapshot["joint_states"]

        qpos = np.asarray(
            [
                *self.robot.get_base_params(),
                *self.robot.get_current_planning_joints(),
            ],
            dtype=np.float32,
        )
        joint_states_with_head = self.robot.get_current_planning_joints_with_head()
        if joint_states_with_head is None:
            head_qpos = np.zeros(2, dtype=np.float32)  # Fallback
        else:
            head_qpos = np.asarray(joint_states_with_head[-2:], dtype=np.float32)

        self.robot._active_episode_ds = episode_record(
            self.robot._active_episode_ds,
            cfg=self.robot._active_episode_cfg,
            goal_xyz_world=self.robot._active_episode_goal_xyz_world,
            rgb=rgb,
            depth=depth,
            qpos=qpos,
            head_qpos=head_qpos,
        )

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

        # Segment
        mask = None
        if target_model_id is not None and segmentation is not None:
            scene = self.robot.robot_env.env.unwrapped.scene
            seg_id = None
            for actor in scene.actors.values():
                if target_model_id in actor.name:
                    seg_id = actor._objs[0].per_scene_id
                    break
            mask = mask_from_segmentation_id(segmentation, seg_id)
        elif segmentation is not None:
            mask = mask_from_uv(segmentation, (u, v))
        else:
            mask = self.sam_client.segment_point(rgb, (u, v))

        if mask is not None and np.count_nonzero(mask) > 0:
            return mask.astype(np.uint8)
        return None

    def _prepare_pointclouds_for_grasping(
        self, depth, segmap, K, camera_extrinsic, joint_dict
    ):
        # 1. Full PCD
        full_mask = np.ones(depth.shape, dtype=bool)
        full_pcd_cam = get_pcd_from_mask(depth, full_mask, K)

        # Transform to world to filter
        full_pcd_cam_h = np.hstack((full_pcd_cam, np.ones((full_pcd_cam.shape[0], 1))))
        full_pcd_world = (camera_extrinsic @ full_pcd_cam_h.T).T[:, :3]

        # Filter robot
        filtered_world_list, _ = self.robot.filter_points_on_robot_with_state(
            full_pcd_world, joint_dict, point_radius=0.04
        )
        filtered_world = np.array(filtered_world_list)

        # Back to camera
        filtered_world_h = np.hstack(
            (filtered_world, np.ones((filtered_world.shape[0], 1)))
        )
        T_cam_world = np.linalg.inv(camera_extrinsic)
        full_pcd_cam_filtered = (T_cam_world @ filtered_world_h.T).T[:, :3]

        # 2. Segment PCD
        object_pcd_cam = get_pcd_from_mask(depth, segmap.astype(bool), K)

        return full_pcd_cam_filtered, object_pcd_cam
