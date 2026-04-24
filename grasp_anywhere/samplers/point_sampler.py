import numpy as np
import open3d as o3d
import tf.transformations as tfs

from grasp_anywhere.utils.logger import log


def _extract_position_orientation(pose_matrix):
    """Helper to extract position and orientation arrays from 4x4 numpy matrix."""
    position = pose_matrix[:3, 3].tolist()
    quaternion = tfs.quaternion_from_matrix(pose_matrix).tolist()
    return position, quaternion


class PointSampler:
    """
    Encapsulates sampling and validation process for finding a pointing pose.

    The process is as follows:
    1. Find a single valid base configuration to reach the object's vicinity.
    2. Generate a top-down pointing pose.
    3. Validate the pose with whole-body IK, trying both 'left' and 'right' arm seeds.
    """

    def __init__(
        self,
        robot,
        object_center_world,
        manipulation_radius,
        num_samples=2,  # Reduced from 20, reflects left/right attempts
        radius=0.3,
        cone_angle=0.0,  # Cone angle not used for top-down
        enable_visualization=False,
    ):
        self.robot = robot
        self.object_center_world = object_center_world
        self.manipulation_radius = manipulation_radius
        self.num_samples = num_samples
        self.radius = radius
        self.cone_angle = cone_angle
        self.enable_visualization = enable_visualization

    def sample_and_validate(self, camera_pose, combined_points, object_pcd_world):
        """
        Generates and validates a top-down pointing pose.
        Args:
            camera_pose (np.ndarray): Current 4x4 pose matrix of the camera.
            combined_points (np.ndarray): Scene point cloud for collision checking.
            object_pcd_world (np.ndarray): Object cloud for collision checking.

        Returns:
            A tuple of (base_config, arm_config) or (None, None).
        """
        log.info("Searching for a valid base configuration...")
        base_config = self.robot.base_sampler.sample_base_pose(
            self.object_center_world, self.manipulation_radius
        )
        if base_config is None:
            log.info(
                "ERROR: Could not find a valid base configuration to reach the object."
            )
            return None, None
        log.info(f"Found a promising base config: {base_config[:2]}.")

        # --- 2. Generate top-down pointing pose ---
        approach_vec = np.array([0.0, 0.0, -1.0])
        point_pos = self.object_center_world + np.array([0.0, 0.0, self.radius])
        point_pose_matrix = self._create_pose_matrix(point_pos, approach_vec)

        # Convert numpy array to Open3D point cloud for visualization
        object_pcd_o3d = o3d.geometry.PointCloud()
        object_pcd_o3d.points = o3d.utility.Vector3dVector(object_pcd_world)

        position, orientation = _extract_position_orientation(point_pose_matrix)

        ik_solution = self.robot.sample_wb_ik(
            position,
            orientation,
            max_attempts=30,
            manipulation_radius=self.manipulation_radius,
        )

        if ik_solution:
            log.info("Found a valid pointing pose with arm config.")
            return ik_solution["base_config"], ik_solution["arm_config"]

        log.info("ERROR: Failed to find a valid pointing pose for top-down approach.")
        return None, None

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

    def _generate_arm_seed_from_side(self, side):
        """Generates a normalized 8-joint arm seed for IK."""
        seed = [0.5] * 8
        if side == "left":
            seed[1] = 1.0
        else:
            seed[1] = 0.0
        return seed
