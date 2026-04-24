import numpy as np
import open3d as o3d
import tf.transformations as tfs

from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.visualization_utils import visualize_place_pipeline


def _extract_position_orientation(pose_matrix):
    """Helper to extract position and orientation arrays from 4x4 numpy matrix."""
    position = pose_matrix[:3, 3].tolist()
    quaternion = tfs.quaternion_from_matrix(pose_matrix).tolist()
    return position, quaternion


class HeuristicPlaceSampler:
    """
    Samples feasible end-effector poses for placing an object on a planar surface.

    The process is as follows:
    1.  The placement surface is modeled as a plane from the input point cloud.
    2.  Candidate points are sampled from the placement point cloud.
    3.  For each point, a grasp pose is generated, with the gripper pointing downwards,
        aligned with the plane normal.
    4.  Each candidate is validated with whole-body IK. The first valid pose is returned.
    """

    def __init__(
        self,
        robot,
        placement_pcd_world,
        num_samples=20,
        gripper_height_offset=0.2,
        manipulation_radius=0.5,
    ):
        self.robot = robot
        self.placement_pcd_world = placement_pcd_world
        self.num_samples = num_samples
        self.gripper_height_offset = gripper_height_offset
        self.manipulation_radius = manipulation_radius

    def sample_and_validate(self, combined_points, debug=True):
        """
        Runs the full sampling and validation pipeline.
        Args:
            combined_points (np.ndarray): Scene point cloud for collision checking.
            debug (bool): If True, visualizes the pipeline.

        Returns:
            A tuple of (base_config, arm_config) or (None, None).
        """
        if self.placement_pcd_world is None or len(self.placement_pcd_world) < 3:
            log.error("ERROR: Placement point cloud is empty or has too few points.")
            return None, None

        # --- 1. Fit a plane to the placement point cloud ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.placement_pcd_world)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        [a, b, c, d] = plane_model
        plane_normal = np.array([a, b, c])

        # Ensure normal points upwards
        if np.dot(plane_normal, np.array([0, 0, 1])) < 0:
            plane_normal = -plane_normal

        log.info(f"Found plane with normal: {plane_normal}")

        # --- 2. Sample from inliers and validate ---
        inlier_points = self.placement_pcd_world[inliers]
        if len(inlier_points) == 0:
            log.info(
                "ERROR: Could not find a dominant plane in the placement point cloud."
            )
            return None, None

        # Downsample the inlier points to ensure a more uniform spatial distribution for sampling.
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
        downsampled_pcd = inlier_pcd.voxel_down_sample(voxel_size=0.05)  # 5cm grid
        downsampled_pcd.paint_uniform_color([0, 1, 0])  # Green for inliers
        sampling_points = np.asarray(downsampled_pcd.points)

        if len(sampling_points) == 0:
            log.warning(
                "WARNING: Downsampling placement inliers resulted in zero points. Falling back to original inliers."
            )
            sampling_points = inlier_points

        # Give higher probability to points closer to the center of the plane.
        if len(sampling_points) > 1:
            centroid = np.mean(sampling_points, axis=0)
            distances = np.linalg.norm(sampling_points - centroid, axis=1)

            # Use a Gaussian-like weighting. Sigma is the stdev of distances.
            std_dev = np.std(distances) / 5.0
            if std_dev > 1e-6:
                weights = np.exp(-0.5 * (distances / std_dev) ** 2)
                probabilities = weights / np.sum(weights)
            else:
                probabilities = None  # All points are equidistant, sample uniformly.
        else:
            probabilities = None  # Not enough points, sample uniformly.

        for i in range(self.num_samples):
            # --- 3. Generate a candidate pose ---
            # Sample a point, biased towards the center.
            if probabilities is not None and len(probabilities) == len(sampling_points):
                idx = np.random.choice(len(sampling_points), p=probabilities)
                target_point = sampling_points[idx]
            else:
                target_point = sampling_points[np.random.choice(len(sampling_points))]

            # Find a base configuration to reach this point
            base_config = self.robot.base_sampler.sample_base_pose(
                target_point, self.manipulation_radius
            )
            if base_config is None:
                log.warning(
                    f"WARNING: Sample {i+1}/{self.num_samples}: Could not find a "
                    f"valid base config for the target point. Skipping."
                )
                continue

            # Add a small offset to avoid collision with the surface
            target_position = target_point + plane_normal * self.gripper_height_offset

            # Gripper's approach should be "forward" relative to the robot base
            base_pos_2d = np.array([base_config[0], base_config[1]])
            target_pos_2d = target_point[:2]
            forward_dir = target_pos_2d - base_pos_2d

            # Add a bit of randomness to the direction
            angle_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            c, s = np.cos(angle_offset), np.sin(angle_offset)
            rot_matrix = np.array([[c, -s], [s, c]])
            forward_dir_rotated = rot_matrix @ forward_dir

            forward_dir_3d = np.array(
                [forward_dir_rotated[0], forward_dir_rotated[1], 0]
            )
            if np.linalg.norm(forward_dir_3d) > 1e-6:
                forward_dir_3d /= np.linalg.norm(forward_dir_3d)
            else:
                log.warning(
                    "WARNING: Forward direction is zero vector, skipping sample."
                )
                continue

            place_pose_matrix = self._create_pose_matrix(
                target_position, forward_dir_3d, plane_normal
            )

            log.info(f"Sample {i+1}/{self.num_samples}: Checking a placement pose.")

            if debug:
                visualize_place_pipeline(
                    place_pose_world=place_pose_matrix,
                    combined_points=combined_points,
                    placement_pcd_world=downsampled_pcd,
                    # placement_inliers_world=downsampled_pcd,
                    is_candidate=True,
                    sample_info=f"Sample {i+1}/{self.num_samples}",
                )

            # --- 4. Validate the candidate ---
            position, orientation = _extract_position_orientation(place_pose_matrix)
            ik_solution = self.robot.sample_wb_ik(
                position,
                orientation,
                max_attempts=30,
                manipulation_radius=self.manipulation_radius,
            )

            if ik_solution:
                log.info(f"Found a valid place pose after {i+1} attempts.")
                return ik_solution["base_config"], ik_solution["arm_config"]

        log.error("ERROR: Failed to find a valid place pose after all samples.")
        return None, None

    def _create_pose_matrix(self, position, forward_direction, plane_normal):
        """Creates a 4x4 pose matrix for the Fetch gripper for placing."""
        # Gripper's z-axis should be opposite to the plane normal for a downward-facing palm.
        # However, the Fetch gripper's z-axis points out of the palm. To place,
        # we need the palm facing down, so the gripper's z-axis must point up.
        z_axis = plane_normal / np.linalg.norm(plane_normal)

        # Gripper's x-axis should be the forward direction, projected onto the plane
        # and made orthogonal to z_axis.
        x_axis = forward_direction - np.dot(forward_direction, z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-6:
            # If forward_direction is aligned with z_axis, pick an arbitrary x in the plane
            ref_vec = np.array([1, 0, 0])
            if abs(np.dot(ref_vec, z_axis)) > 0.99:
                ref_vec = np.array([0, 1, 0])
            x_axis = ref_vec - np.dot(ref_vec, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)

        # Gripper's y-axis is the cross product to form a right-handed system.
        y_axis = np.cross(z_axis, x_axis)

        pose_matrix = np.eye(4)
        pose_matrix[:3, 0] = x_axis
        pose_matrix[:3, 1] = y_axis
        pose_matrix[:3, 2] = z_axis
        pose_matrix[:3, 3] = position
        return pose_matrix
