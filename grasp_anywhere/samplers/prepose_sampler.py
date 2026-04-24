from pathlib import Path

import numpy as np
import open3d as o3d

from grasp_anywhere.checker.occlusion_checker import (
    SelfOcclusionChecker,
)
from grasp_anywhere.data_collector.prepose_collector import prepose_record
from grasp_anywhere.robot.ik.ikfast_api import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
from grasp_anywhere.robot.utils.transform_utils import (
    quaternion_from_matrix,
    transform_pose_to_base,
)
from grasp_anywhere.utils import reachability_utils
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.torso_utils import query_best_torso
from grasp_anywhere.utils.visualization_utils import (
    show_costmap,
    visualize_prepose_pipeline,
)


def _extract_position_orientation(pose_matrix):
    """Helper to extract position and orientation arrays from 4x4 numpy matrix."""
    position = pose_matrix[:3, 3].tolist()
    quaternion = quaternion_from_matrix(pose_matrix)
    return position, quaternion


def _farthest_point_sampling(points, num_samples):
    """Downsample points using farthest point sampling."""
    n = len(points)
    if n <= num_samples:
        return points

    sampled_indices = [np.random.randint(n)]
    distances = np.full(n, np.inf)

    for _ in range(num_samples - 1):
        last_idx = sampled_indices[-1]
        dist = np.linalg.norm(points - points[last_idx], axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices.append(np.argmax(distances))

    return points[sampled_indices]


def _get_next_data_index(save_dir):
    """Get the next available index for saving data."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(save_dir.glob("prepose_data_*.npy"))
    if not existing_files:
        return 0

    indices = []
    for f in existing_files:
        try:
            idx = int(f.stem.split("_")[-1])
            indices.append(idx)
        except ValueError:
            continue

    return max(indices) + 1 if indices else 0


class PreposeSampler:
    """
    Minimal pre-pose sampler: uniformly sample end-effector poses on a sphere
    around the object, then validate each candidate with:
    - collision checking via whole-body IK + planner validation
    - self-occlusion check using a simple forward-kinematics model
    The first valid candidate is returned.
    """

    def __init__(
        self,
        robot,
        object_center_world,
        manipulation_radius,
        num_samples=20,
        radius=0.25,
        enable_visualization=False,
    ):
        self.robot = robot
        self.object_center_world = object_center_world
        self.manipulation_radius = manipulation_radius
        self.num_samples = num_samples
        self.radius = radius
        self.enable_visualization = enable_visualization
        self.failed_due_to_reachability = False
        # Initialize self-occlusion checker
        self.occlusion_checker = SelfOcclusionChecker()

    def sample_and_validate(self, combined_points, object_pcd_world):
        """
        Runs the full sampling and validation pipeline.
        Args:
            combined_points (np.ndarray): Scene point cloud for visualization.
            object_pcd_world (o3d.geometry.PointCloud): Object cloud for visualization.

        Returns:
            A tuple of (base_config, arm_config) or (None, None).
        """
        self.failed_due_to_reachability = False
        # 1. Update base sampler from point cloud
        if self.robot.base_sampler.dynamic_costmap:
            self.robot.base_sampler.update_from_pointcloud(combined_points)
            if (
                self.enable_visualization
                and self.robot.base_sampler.costmap is not None
            ):
                show_costmap(self.robot.base_sampler.costmap)

        batch_size = 128

        # Rejection counters for summary
        reject_base_sample = 0
        reject_reachability = 0
        reject_capability = 0
        reject_ik = 0
        reject_collision = 0
        reject_occlusion = 0
        total_ee_candidates = 0

        for _ in range(self.num_samples):
            # 2. Sample Base Config
            base_config = self.robot.base_sampler.sample_base_pose(
                self.object_center_world, manipulation_radius=self.manipulation_radius
            )
            if base_config is None:
                reject_base_sample += 1
                continue

            # 2b. Filter base configs with low reachability (position-only map)
            base_reach = reachability_utils.query_reachability_score(
                self.robot.reachability_map,
                base_config,
                self.object_center_world.tolist(),
            )
            log.info(f"Base reachability score: {base_reach}")
            log.info(f"Base config: {base_config}")
            if base_reach <= 0.04:
                reject_reachability += 1
                continue

            # 2c. Choose torso height ONCE per base config using the object center (world frame).
            torso_lift = query_best_torso(
                self.robot.torso_map,
                np.asarray(self.object_center_world, dtype=np.float64),
                np.asarray(base_config, dtype=np.float64),
            )
            torso_lift = float(
                np.clip(
                    torso_lift,
                    float(JOINT_LIMITS_LOWER[0]),
                    float(JOINT_LIMITS_UPPER[0]),
                )
            )

            # 3. Sample Batch of EE Poses
            random_dirs = np.random.randn(batch_size, 3)
            # Normalize
            norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0  # Avoid division by zero
            random_dirs = random_dirs / norms

            ee_positions = self.object_center_world - random_dirs * self.radius

            for ee_pos in ee_positions:
                total_ee_candidates += 1
                # Compute target ee pose
                approach_vec = self.object_center_world - ee_pos
                approach_vec = approach_vec / np.linalg.norm(approach_vec)
                prepose_matrix = self._create_pose_matrix(ee_pos, approach_vec)
                pos, quat = _extract_position_orientation(prepose_matrix)
                # Pass position and quaternion [x, y, z, qx, qy, qz, qw]
                target_pose_7d = [
                    pos[0],
                    pos[1],
                    pos[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3],
                ]

                # 4. Filter with capability map
                score = reachability_utils.query_capability_score(
                    self.robot.capability_map, base_config, target_pose_7d
                )

                if score <= 0.04:
                    reject_capability += 1
                    continue

                if self.enable_visualization:
                    log.info(f"Visualizing prepose, score: {score}")
                    object_pcd_o3d = o3d.geometry.PointCloud()
                    object_pcd_o3d.points = o3d.utility.Vector3dVector(object_pcd_world)
                    visualize_prepose_pipeline(
                        base_config=base_config,  # [x, y, theta]
                        prepose_world=prepose_matrix,
                        combined_points=combined_points,
                        object_center_world=self.object_center_world,
                        sampling_radius=self.radius,
                        object_pcd_world=object_pcd_o3d,
                        is_candidate=True,
                    )

                # 5. Solve IK
                ik_solutions = self.robot.sample_ik(
                    pos,
                    quat,
                    base_config,
                    torso_lift=torso_lift,
                )

                if ik_solutions is None:
                    reject_ik += 1
                    continue

                # 6. Validate Whole Body Config (Collision and Occlusion)

                # Pre-calculate data needed for occlusion check
                cam_K = self.robot.get_camera_intrinsic()
                image_shape = (640, 480)
                target_center_base, _ = transform_pose_to_base(
                    world_pos=self.object_center_world.tolist(),
                    world_quat=[0.0, 0.0, 0.0, 1.0],
                    base_pos=[base_config[0], base_config[1], 0.0],
                    base_yaw=base_config[2],
                )

                valid_solution = None
                for ik_sol in ik_solutions:
                    # Check collision (returns True if in collision)
                    if self.robot.validate_whole_body_config(ik_sol, base_config):
                        reject_collision += 1
                        continue

                    # Check occlusion
                    joint_angles = np.array(ik_sol, dtype=np.float64)
                    is_occluded = self.occlusion_checker.check_occlusion(
                        joint_angles=joint_angles,
                        target_center_base=np.array(
                            target_center_base, dtype=np.float64
                        ),
                        target_radius=0.05,
                        cam_K=np.array(cam_K, dtype=np.float64),
                        image_shape=image_shape,
                    )

                    if is_occluded:
                        reject_occlusion += 1
                        continue

                    valid_solution = ik_sol
                    break

                if valid_solution is None:
                    continue

                # All checks passed
                # Save the first valid planned prepose for sampler training.
                # This uses the *current* pointcloud used for validation (combined_points),
                # cropped around the object center and converted to object-centered frame.
                prepose_record(
                    combined_points=np.asarray(combined_points, dtype=np.float32),
                    object_center_world=np.asarray(
                        self.object_center_world, dtype=np.float32
                    ),
                    radius=float(self.radius),
                    prepose_matrix=np.asarray(prepose_matrix, dtype=np.float32),
                    base_config=np.asarray(base_config, dtype=np.float32),
                    arm_config=np.asarray(valid_solution, dtype=np.float32),
                )
                return base_config, valid_solution

        # Print rejection summary
        log.error(
            f"Prepose sampling failed. Rejections: "
            f"base={reject_base_sample}, reach={reject_reachability}, "
            f"cap={reject_capability}/{total_ee_candidates}, ik={reject_ik}, "
            f"collision={reject_collision}, occlusion={reject_occlusion}"
        )
        # If every usable base sample failed the reachability gate, call it out-of-reachability.
        usable_base = self.num_samples - reject_base_sample
        if (usable_base == 0) or (
            usable_base > 0
            and reject_reachability == usable_base
            and total_ee_candidates == 0
        ):
            self.failed_due_to_reachability = True
        return None, None

    def _save_prepose_data(self, ee_pose_world, combined_points):
        """Save end-effector pose and nearby point cloud data."""
        save_dir = "resources/benchmark/prepose_data"

        # Extract points within 5*radius of object center
        points = np.asarray(combined_points)
        distances = np.linalg.norm(points - self.object_center_world, axis=1)
        nearby_mask = distances <= (5 * self.radius)
        nearby_points = points[nearby_mask]

        # Transform to object-centered frame
        ee_pose_obj = ee_pose_world.copy()
        ee_pose_obj[:3, 3] -= self.object_center_world

        points_obj = nearby_points - self.object_center_world

        # Process point cloud to 4096 dimensions
        num_points = len(points_obj)
        target_size = 4096

        if num_points == 0:
            points_4096 = np.zeros((target_size, 3))
        elif num_points < target_size:
            # Pad with zeros
            points_4096 = np.zeros((target_size, 3))
            points_4096[:num_points] = points_obj
        else:
            # Downsample using farthest point sampling
            points_4096 = _farthest_point_sampling(points_obj, target_size)

        # Get next index and save
        idx = _get_next_data_index(save_dir)
        save_path = Path(save_dir)

        # Save as NPY
        data_dict = {
            "ee_pose": ee_pose_obj,
            "points": points_4096,
            "object_center_world": self.object_center_world,
            "radius": self.radius,
        }
        np.save(save_path / f"prepose_data_{idx}.npy", data_dict)

        # Save as PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_4096)
        o3d.io.write_point_cloud(str(save_path / f"prepose_pcd_{idx}.ply"), pcd)

        log.info(f"Saved prepose data to index {idx}")

    def _create_pose_matrix(self, position, approach_vector):
        """Creates a 4x4 pose matrix for the Fetch gripper."""
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
