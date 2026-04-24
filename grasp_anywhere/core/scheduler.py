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
from grasp_anywhere.data_collector.viz_data_collector import VizDataCollector
from grasp_anywhere.grasping_client.grasp_generation import predict_grasps
from grasp_anywhere.stage_planners.confirm_stage import ConfirmPlanner
from grasp_anywhere.stage_planners.grasp_stage import GraspPlanner
from grasp_anywhere.stage_planners.move_stage import MovePlanner
from grasp_anywhere.stage_planners.place_stage import PlacePlanner
from grasp_anywhere.stage_planners.point_stage import PointPlanner
from grasp_anywhere.stage_planners.prepose_stage import PreposePlanner
from grasp_anywhere.stage_planners.static_point_stage import StaticPointPlanner
from grasp_anywhere.utils.grasp_utils import select_diverse_grasps
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.perception_utils import get_pcd_from_mask
from grasp_anywhere.grasping_client.sam_client import SamClient, SamConfig
from grasp_anywhere.utils.seg_utils import mask_from_segmentation_id, mask_from_uv
from grasp_anywhere.utils.visualization_utils import (
    visualize_grasps,
    visualize_pcd,
)


class Scheduler:
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

        # We will move initialization logic here step-by-step
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

        # Do not load collision pointclouds here.
        # The collision environment will be taken from the robot's maintained scene.

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
        self.sam_client = SamClient(
            SamConfig(url=services.get("sam_url", "http://localhost:4001"))
        )

        # Initialize visualization data collector
        self.viz_collector = VizDataCollector()

    def grasp_anywhere(self, object_pcd, max_attempts=5, target_model_id: str = None):
        """
        Main loop for the grasping pipeline. It attempts to grasp an object given
        by a single 3D coordinate, retrying up to a max number of attempts.
        """
        self.viz_collector.reset()

        # Interpret input strictly as a single coordinate (1x3)
        object_coord = np.asarray(object_pcd, dtype=np.float32).reshape(1, 3)
        log.info(
            f"Starting grasp_anywhere with target coordinate {object_coord.flatten().tolist()}"
        )

        self.robot._active_episode_cfg = DataCollectionConfig(
            run_name="grasp_episode",
        )
        self.robot._active_episode_ds = None
        self.robot._active_episode_goal_xyz_world = object_coord[0].astype(np.float32)

        # Failure tracking
        self.attempt_outcomes = []

        for attempt in range(max_attempts):
            log.info(f"--- Grasping Attempt {attempt + 1}/{max_attempts} ---")

            # Check if robot is close enough to target for grasping
            base_x, base_y, _ = self.robot.get_base_params()
            target_x, target_y = object_coord[0, 0], object_coord[0, 1]
            distance_to_target = np.linalg.norm([target_x - base_x, target_y - base_y])

            if distance_to_target > 1.2 * self.manipulation_radius:
                log.info(
                    f"Target is {distance_to_target:.2f}m away (> {self.manipulation_radius}m)."
                    "Skipping GRASP stage, going to PRE-POSE."
                )
                current_outcome = (
                    "OUT_OF_REACHABILITY"  # Default if skipped and pre-pose fails
                )
            else:
                current_outcome = "PERCEPTION_FAILURE"  # Default if perception fails
                # --- Stage 1: Grasp ---
                self.viz_collector.set_stage("GRASP")
                log.info("Attempting GRASP stage...")
                # Get current sensor data for grasp prediction (atomic snapshot for sim)
                snapshot = self.robot.robot_env.get_sensor_snapshot()
                rgb = snapshot["rgb"]
                if "segmentation" in snapshot:
                    segmentation = snapshot["segmentation"]
                else:
                    segmentation = None
                depth = snapshot["depth"]
                camera_intrinsic = snapshot["intrinsics"]
                joint_names, joint_positions = snapshot["joint_states"]
                joint_dict = dict(zip(joint_names, joint_positions))

                qpos = np.asarray(
                    [
                        *self.robot.get_base_params(),
                        *self.robot.get_current_planning_joints(),
                    ],
                    dtype=np.float32,
                )
                # Get head qpos
                joint_states_with_head = (
                    self.robot.get_current_planning_joints_with_head()
                )
                if joint_states_with_head is None:
                    raise ValueError(
                        "Failed to retrieve joint states with head in Grasp Stage."
                    )
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

                # Compute camera pose from joint states snapshot (uses FK)
                camera_extrinsic = self.robot.compute_camera_pose_from_joints(
                    joint_dict
                )

                # Refresh maintained scene with synchronized robot filtering
                self.robot.scene.update(
                    depth,
                    camera_intrinsic,
                    camera_extrinsic,
                    joint_dict,
                    enable_icp_alignment=self.enable_pcd_alignment,
                )
                # Update planner collision environment to match maintained scene
                self.robot.clear_pointclouds()
                self.robot.add_pointcloud(
                    self.robot.scene.current_environment(),
                    filter_robot=True,
                    point_radius=0.03,
                )

                # Build a segmentation mask from the single world coordinate using SAM
                segmap = None
                object_pcd_for_planning = object_coord
                H, W = rgb.shape[:2]

                world_point = object_pcd_for_planning[0]
                T_cam_world = np.linalg.inv(camera_extrinsic)
                p_cam_h = T_cam_world @ np.array(
                    [world_point[0], world_point[1], world_point[2], 1.0],
                    dtype=np.float32,
                )
                z_cam = p_cam_h[2]
                if z_cam <= 0:
                    log.warning(
                        "Projected point is behind the camera; skipping SAM segmentation."
                    )
                else:
                    uv = camera_intrinsic @ p_cam_h[:3]
                    u = int(np.round(uv[0] / z_cam))
                    v = int(np.round(uv[1] / z_cam))
                    if 0 <= u < W and 0 <= v < H:
                        log.info(
                            f"Running SAM segmentation at pixel ({u}, {v}) from object center."
                        )

                        if target_model_id is not None and segmentation is not None:
                            # Look up per_scene_id from scene.actors by model_id
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
                            segmap = mask.astype(np.uint8)
                            # Extract object point cloud from the mask (camera frame), then to world frame
                            object_pcd_cam = get_pcd_from_mask(
                                depth, segmap.astype(bool), camera_intrinsic
                            )
                            if object_pcd_cam is not None and len(object_pcd_cam) > 0:
                                object_pcd_world_h = np.hstack(
                                    (
                                        object_pcd_cam,
                                        np.ones(
                                            (object_pcd_cam.shape[0], 1),
                                            dtype=object_pcd_cam.dtype,
                                        ),
                                    )
                                )
                                object_pcd_world = (
                                    camera_extrinsic @ object_pcd_world_h.T
                                ).T[:, :3]
                                object_pcd_for_planning = object_pcd_world
                                log.info(
                                    f"Extracted object point cloud via SAM with {len(object_pcd_world)} points."
                                )
                            else:
                                log.warning(
                                    "Failed to extract object point cloud from SAM mask."
                                )
                    else:
                        log.warning(
                            "Projected pixel out of image bounds; skipping SAM segmentation."
                        )
                if segmap is None or np.count_nonzero(segmap) == 0:
                    log.warning(
                        "Could not generate a valid segmap from the object coordinate."
                    )
                    # Fall through to pre-pose stage
                    current_outcome = "PERCEPTION_FAILURE"
                else:
                    # --- Point Cloud Context Prediction (No Transfer) ---
                    # 1. Get Full Point Cloud of the scene in Camera Frame
                    full_mask = np.ones(depth.shape, dtype=bool)
                    full_pcd_cam = get_pcd_from_mask(depth, full_mask, camera_intrinsic)

                    # Transform to world frame
                    full_pcd_cam_h = np.hstack(
                        (full_pcd_cam, np.ones((full_pcd_cam.shape[0], 1)))
                    )
                    full_pcd_world = (camera_extrinsic @ full_pcd_cam_h.T).T[:, :3]

                    # Filter using robot self-filter
                    (
                        filtered_world_list,
                        _,
                    ) = self.robot.filter_points_on_robot_with_state(
                        full_pcd_world, joint_dict, point_radius=0.04
                    )
                    filtered_world = np.array(filtered_world_list)

                    # Transform back to camera frame
                    filtered_world_h = np.hstack(
                        (filtered_world, np.ones((filtered_world.shape[0], 1)))
                    )
                    T_cam_world = np.linalg.inv(camera_extrinsic)
                    full_pcd_cam = (T_cam_world @ filtered_world_h.T).T[:, :3]

                    # Visualize the filtered point cloud
                    if self.enable_visualization:
                        visualize_pcd(
                            full_pcd_cam,
                            rgb,
                            depth,
                            camera_intrinsic,
                        )

                    segment_pcd_cam = object_pcd_cam

                    # --- Zoom In and Predict Grasps (using functional interface) ---
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

                    if pred_grasps_cam is not None and len(pred_grasps_cam) > 0:
                        if self.enable_visualization:
                            visualize_grasps(
                                pred_grasps_cam,
                                scores,
                                rgb,
                                depth,
                                camera_intrinsic,
                            )
                        # Transform grasp poses to world frame
                        grasp_poses_world = [
                            camera_extrinsic @ g for g in pred_grasps_cam
                        ]

                        max_tries = 3
                        # Select diverse grasps instead of just top-scored ones
                        diverse_indices = select_diverse_grasps(
                            grasp_poses_world, max_tries
                        )
                        total = len(diverse_indices)
                        for i, idx in enumerate(diverse_indices):
                            pose = grasp_poses_world[idx]
                            log.info(
                                f"Trying grasp pose {i + 1}/{total} (original rank {idx + 1})"
                            )

                            # Recapture scene to handle robot movement between attempts
                            snapshot = self.robot.robot_env.get_sensor_snapshot()
                            current_extrinsic = camera_extrinsic

                            if snapshot is not None:
                                depth = snapshot["depth"]
                                camera_intrinsic = snapshot["intrinsics"]
                                joint_names, joint_positions = snapshot["joint_states"]
                                joint_dict = dict(zip(joint_names, joint_positions))
                                rgb = snapshot["rgb"]

                                qpos = np.asarray(
                                    [
                                        *self.robot.get_base_params(),
                                        *self.robot.get_current_planning_joints(),
                                    ],
                                    dtype=np.float32,
                                )
                                # Get head qpos
                                joint_states_with_head = (
                                    self.robot.get_current_planning_joints_with_head()
                                )
                                if joint_states_with_head is None:
                                    raise ValueError(
                                        "Failed to retrieve joint states with head in Grasp Attempt."
                                    )
                                head_qpos = np.asarray(
                                    joint_states_with_head[-2:], dtype=np.float32
                                )

                                self.robot._active_episode_ds = episode_record(
                                    self.robot._active_episode_ds,
                                    cfg=self.robot._active_episode_cfg,
                                    goal_xyz_world=self.robot._active_episode_goal_xyz_world,
                                    rgb=rgb,
                                    depth=depth,
                                    qpos=qpos,
                                    head_qpos=head_qpos,
                                )

                                current_extrinsic = (
                                    self.robot.compute_camera_pose_from_joints(
                                        joint_dict
                                    )
                                )
                                self.robot.scene.update(
                                    depth,
                                    camera_intrinsic,
                                    current_extrinsic,
                                    joint_dict,
                                    enable_icp_alignment=self.enable_pcd_alignment,
                                )

                            success, msg = self.grasp_planner.run(
                                pose,
                                current_extrinsic,
                                self.robot.scene.current_environment(),
                            )
                            if success:
                                log.info(f"Grasp successful on attempt {attempt + 1}")
                                self._finalize_episode(True)
                                return True, f"Grasp successful: {msg}"
                            else:
                                log.warning(f"Grasp failed: {msg}")
                                # Clear the episode cache so next attempt starts fresh
                                self.robot._active_episode_ds = None
                                self.robot.point_head_at(object_coord[0].tolist())
                                time.sleep(1.5)
                                current_outcome = (
                                    "IK_FAILED"
                                    if "Failed to find IK" in msg
                                    else "GRASP_EXECUTION_FAILED"
                                )
                    else:
                        log.warning("No grasping poses detected")

                        current_outcome = "PERCEPTION_FAILURE"

                    # If we got here, we either had no candidates, or filtered all out (UNREACHABLE),
                    # or tried execution and failed (EXECUTION_ERROR/IK_FAILED).
                    # 'current_outcome' holds the reason.
                    log.warning(
                        f"Grasp attempt finished with outcome: {current_outcome}"
                    )

            # --- Stage 2: Pre-Pose ---
            self.viz_collector.set_stage("PRE-POSE")
            log.info("Grasp stage failed. Attempting PRE-POSE stage.")
            prepose_camera_pose = self.robot.get_camera_pose()
            # Refresh maintained scene right before pre-pose planning so the
            # environment point cloud provider reflects the latest observation.
            snapshot = self.robot.robot_env.get_sensor_snapshot()

            depth = snapshot["depth"]
            camera_intrinsic = snapshot["intrinsics"]
            joint_names, joint_positions = snapshot["joint_states"]
            joint_dict = dict(zip(joint_names, joint_positions))
            rgb = snapshot["rgb"]

            qpos = np.asarray(
                [
                    *self.robot.get_base_params(),
                    *self.robot.get_current_planning_joints(),
                ],
                dtype=np.float32,
            )
            # Get head qpos
            joint_states_with_head = self.robot.get_current_planning_joints_with_head()
            if joint_states_with_head is None:
                raise ValueError(
                    "Failed to retrieve joint states with head in Pre-Pose Stage."
                )
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
            # Compute camera pose from synchronized joint snapshot (FK)
            prepose_camera_pose = self.robot.compute_camera_pose_from_joints(joint_dict)
            self.robot.scene.update(
                depth,
                camera_intrinsic,
                prepose_camera_pose,
                joint_dict,
                enable_icp_alignment=self.enable_pcd_alignment,
            )
            # Sync VAMP collision environment before prepose sampling
            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(
                self.robot.scene.current_environment(),
                filter_robot=True,
                point_radius=0.03,
            )
            config_generator, object_center = self.prepose_planner.plan(
                object_coord,
                prepose_camera_pose,
                # Pass a provider to fetch the latest scene point cloud dynamically
                self.robot.scene.current_environment,
            )
            if config_generator is not None:
                move_success = self.prepose_planner.execute_prepose(
                    config_generator, self.enable_replanning, self.enable_pcd_alignment
                )
                if move_success:
                    log.info("Pre-pose move successful. Continuing to next attempt.")
                    time.sleep(2.0)
                    # Ensure head looks at the target object after pre-pose
                    self.robot.point_head_at(
                        np.asarray(object_center, dtype=np.float32).tolist()
                    )
                    rospy.sleep(2.0)

                    # Trigger Dynamic Benchmark obstacles (manipulation phase) - after prepose, looking at object
                    if hasattr(self.robot.robot_env, "benchmark_manager"):
                        self.robot.robot_env.benchmark_manager.spawn_manipulation_obstacles(
                            np.array(object_center)
                        )

                    self.attempt_outcomes.append(current_outcome)
                    continue  # Go to the next attempt in the for loop
                else:
                    if getattr(
                        self.prepose_planner, "failed_due_to_reachability", False
                    ):
                        # Only upgrade to OUT_OF_REACHABILITY if not already a more severe error
                        current_outcome = "OUT_OF_REACHABILITY"
                    else:
                        # Generic planning failure if not strictly reachability
                        if current_outcome in [
                            "PERCEPTION_FAILURE",
                            "OUT_OF_REACHABILITY",
                        ]:
                            current_outcome = "PLANNING_FAILURE"

            log.warning("Pre-pose stage failed.")

            # --- Stage 3: Observe ---
            self.viz_collector.set_stage("OBSERVE")
            log.info("Pre-pose failed. Attempting OBSERVE stage.")
            success, msg = self.move_planner.run_observe_object(
                object_coord,
                self.robot.scene.current_environment(),
                2 * self.manipulation_radius,
                self.enable_replanning,
                self.enable_pcd_alignment,
            )
            if success:
                log.info("Observe move successful. Continuing to next attempt.")
            else:
                log.error(f"Observe move failed: {msg}. This was the last resort.")

            # Record the final outcome of this attempt
            self.attempt_outcomes.append(current_outcome)

        # If loop finishes, all attempts have failed
        self._finalize_episode(False)

        log.info(f"All attempts failed. History: {self.attempt_outcomes}")

        # Priority Check
        if "GRASP_EXECUTION_FAILED" in self.attempt_outcomes:
            return False, "GRASP_EXECUTION_FAILED"
        if "IK_FAILED" in self.attempt_outcomes:
            return False, "IK_FAILED"
        if "PERCEPTION_FAILURE" in self.attempt_outcomes:
            return False, "PERCEPTION_FAILURE"
        if "OUT_OF_REACHABILITY" in self.attempt_outcomes:
            return False, "OUT_OF_REACHABILITY"
        if "PLANNING_FAILURE" in self.attempt_outcomes:
            return False, "PLANNING_FAILURE"

        return False, "PERCEPTION_FAILURE"

    def _finalize_episode(self, success: bool) -> None:
        """Finish the active data-collection episode (if any) and save viz data."""
        if (
            self.robot._active_episode_ds is not None
            and self.robot._active_episode_cfg is not None
        ):
            rgbd_path = episode_finish(
                self.robot._active_episode_ds,
                cfg=self.robot._active_episode_cfg,
                dt=1.0 / float(self.robot._active_episode_cfg.record_hz),
            )
            self.viz_collector.record_rgbd_path(rgbd_path)
        suffix = "success" if success else "failed"
        self.viz_collector.save(f"grasp_{suffix}")
        self.robot._active_episode_cfg = None
        self.robot._active_episode_ds = None
        self.robot._active_episode_goal_xyz_world = None

    def _create_segmap_from_pcd(self, pcd_world, image_shape, K, T_world_cam):
        """
        Projects a 3D point cloud in the world frame onto the image plane to create a segmentation mask.
        """
        if pcd_world is None or K is None or T_world_cam is None:
            return None

        T_cam_world = np.linalg.inv(T_world_cam)
        pcd_world_h = np.hstack((pcd_world, np.ones((pcd_world.shape[0], 1))))
        pcd_cam_h = (T_cam_world @ pcd_world_h.T).T
        pcd_cam = pcd_cam_h[:, :3]

        # Project points to image plane
        pcd_proj = (K @ pcd_cam.T).T
        pcd_pixels = pcd_proj[:, :2] / pcd_proj[:, 2:]
        pcd_pixels = pcd_pixels.astype(int)

        # Filter points that are within image bounds
        height, width = image_shape
        valid_indices = (
            (pcd_pixels[:, 0] >= 0)
            & (pcd_pixels[:, 0] < width)
            & (pcd_pixels[:, 1] >= 0)
            & (pcd_pixels[:, 1] < height)
        )
        valid_pixels = pcd_pixels[valid_indices]

        # Create segmap
        segmap = np.zeros(image_shape, dtype=np.uint8)
        segmap[valid_pixels[:, 1], valid_pixels[:, 0]] = 1
        return segmap
