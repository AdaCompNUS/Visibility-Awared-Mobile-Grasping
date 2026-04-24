from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# The sphere data is extracted from resources/fetch_ext/fetch_spherized.urdf
# It is embedded here to reduce dependencies.
# Structure: {link_name: [{'center': [x, y, z], 'radius': r}, ...]}
SPHERE_DATA = {
    "base_link": [
        {"center": [-0.12, 0.0, 0.182], "radius": 0.24},
        {"center": [0.225, 0.0, 0.31], "radius": 0.066},
        {"center": [0.08, -0.06, 0.16], "radius": 0.22},
        {"center": [0.215, -0.07, 0.31], "radius": 0.066},
        {"center": [0.185, -0.135, 0.31], "radius": 0.066},
        {"center": [0.13, -0.185, 0.31], "radius": 0.066},
        {"center": [0.065, -0.2, 0.31], "radius": 0.066},
        {"center": [0.01, -0.2, 0.31], "radius": 0.066},
        {"center": [0.08, 0.06, 0.16], "radius": 0.22},
        {"center": [0.215, 0.07, 0.31], "radius": 0.066},
        {"center": [0.185, 0.135, 0.31], "radius": 0.066},
        {"center": [0.13, 0.185, 0.31], "radius": 0.066},
        {"center": [0.065, 0.2, 0.31], "radius": 0.066},
        {"center": [0.01, 0.2, 0.31], "radius": 0.066},
    ],
    "torso_lift_link": [
        {"center": [-0.1, -0.05, 0.15], "radius": 0.15},
        {"center": [-0.1, 0.05, 0.15], "radius": 0.15},
        {"center": [-0.1, 0.05, 0.3], "radius": 0.15},
        {"center": [-0.1, 0.05, 0.45], "radius": 0.15},
        {"center": [-0.1, -0.05, 0.45], "radius": 0.15},
        {"center": [-0.1, -0.05, 0.3], "radius": 0.15},
    ],
    "torso_lift_link_collision_2": [
        {"center": [0.1, 0.0, 0.24], "radius": 0.07},
    ],
    "head_pan_link": [
        {"center": [0.0, 0.0, 0.06], "radius": 0.15},
        {"center": [0.145, 0.0, 0.058], "radius": 0.05},
        {"center": [0.145, -0.0425, 0.058], "radius": 0.05},
        {"center": [0.145, 0.0425, 0.058], "radius": 0.05},
        {"center": [0.145, 0.085, 0.058], "radius": 0.05},
        {"center": [0.145, -0.085, 0.058], "radius": 0.05},
        {"center": [0.0625, -0.115, 0.03], "radius": 0.03},
        {"center": [0.088, -0.115, 0.03], "radius": 0.03},
        {"center": [0.1135, -0.115, 0.03], "radius": 0.03},
        {"center": [0.139, -0.115, 0.03], "radius": 0.03},
        {"center": [0.0625, -0.115, 0.085], "radius": 0.03},
        {"center": [0.088, -0.115, 0.085], "radius": 0.03},
        {"center": [0.1135, -0.115, 0.085], "radius": 0.03},
        {"center": [0.139, -0.115, 0.085], "radius": 0.03},
        {"center": [0.16, -0.115, 0.075], "radius": 0.03},
        {"center": [0.168, -0.115, 0.0575], "radius": 0.03},
        {"center": [0.16, -0.115, 0.04], "radius": 0.03},
        {"center": [0.0625, 0.115, 0.03], "radius": 0.03},
        {"center": [0.088, 0.115, 0.03], "radius": 0.03},
        {"center": [0.1135, 0.115, 0.03], "radius": 0.03},
        {"center": [0.139, 0.115, 0.03], "radius": 0.03},
        {"center": [0.0625, 0.115, 0.085], "radius": 0.03},
        {"center": [0.088, 0.115, 0.085], "radius": 0.03},
        {"center": [0.1135, 0.115, 0.085], "radius": 0.03},
        {"center": [0.139, 0.115, 0.085], "radius": 0.03},
        {"center": [0.16, 0.115, 0.075], "radius": 0.03},
        {"center": [0.168, 0.115, 0.0575], "radius": 0.03},
        {"center": [0.16, 0.115, 0.04], "radius": 0.03},
    ],
    "shoulder_pan_link": [
        {"center": [0.0, 0.0, 0.0], "radius": 0.055},
        {"center": [0.025, -0.015, 0.035], "radius": 0.055},
        {"center": [0.05, -0.03, 0.06], "radius": 0.055},
        {"center": [0.12, -0.03, 0.06], "radius": 0.055},
    ],
    "shoulder_lift_link": [
        {"center": [0.025, 0.04, 0.025], "radius": 0.04},
        {"center": [-0.025, 0.04, -0.025], "radius": 0.04},
        {"center": [0.025, 0.04, -0.025], "radius": 0.04},
        {"center": [-0.025, 0.04, 0.025], "radius": 0.04},
        {"center": [0.08, 0.0, 0.0], "radius": 0.055},
        {"center": [0.11, 0.0, 0.0], "radius": 0.055},
        {"center": [0.14, 0.0, 0.0], "radius": 0.055},
    ],
    "upperarm_roll_link": [
        {"center": [-0.02, 0.0, 0.0], "radius": 0.055},
        {"center": [0.03, 0.0, 0.0], "radius": 0.055},
        {"center": [0.08, 0.0, 0.0], "radius": 0.055},
        {"center": [0.11, -0.045, 0.02], "radius": 0.03},
        {"center": [0.11, -0.045, -0.02], "radius": 0.03},
        {"center": [0.155, -0.045, 0.02], "radius": 0.03},
        {"center": [0.155, -0.045, -0.02], "radius": 0.03},
        {"center": [0.13, 0.0, 0.0], "radius": 0.055},
    ],
    "elbow_flex_link": [
        {"center": [0.02, 0.045, 0.02], "radius": 0.03},
        {"center": [0.02, 0.045, -0.02], "radius": 0.03},
        {"center": [-0.02, 0.045, 0.02], "radius": 0.03},
        {"center": [-0.02, 0.045, -0.02], "radius": 0.03},
        {"center": [0.08, 0.0, 0.0], "radius": 0.055},
        {"center": [0.14, 0.0, 0.0], "radius": 0.055},
    ],
    "forearm_roll_link": [
        {"center": [0.0, 0.0, 0.0], "radius": 0.055},
        {"center": [0.05, -0.06, 0.02], "radius": 0.03},
        {"center": [0.05, -0.06, -0.02], "radius": 0.03},
        {"center": [0.1, -0.06, 0.02], "radius": 0.03},
        {"center": [0.1, -0.06, -0.02], "radius": 0.03},
        {"center": [0.15, -0.06, 0.02], "radius": 0.03},
        {"center": [0.15, -0.06, -0.02], "radius": 0.03},
    ],
    "wrist_flex_link": [
        {"center": [0.0, 0.0, 0.0], "radius": 0.055},
        {"center": [0.06, 0.0, 0.0], "radius": 0.055},
        {"center": [0.02, 0.045, 0.02], "radius": 0.03},
        {"center": [0.02, 0.045, -0.02], "radius": 0.03},
        {"center": [-0.02, 0.045, 0.02], "radius": 0.03},
        {"center": [-0.02, 0.045, -0.02], "radius": 0.03},
    ],
    "wrist_roll_link": [
        {"center": [-0.03, 0.0, 0.0], "radius": 0.055},
        {"center": [0.0, 0.0, 0.0], "radius": 0.055},
    ],
    "gripper_link": [
        {"center": [-0.07, 0.02, 0.0], "radius": 0.05},
        {"center": [-0.07, -0.02, 0.0], "radius": 0.05},
        {"center": [-0.1, 0.02, 0.0], "radius": 0.05},
        {"center": [-0.1, -0.02, 0.0], "radius": 0.05},
    ],
    "r_gripper_finger_link": [
        {"center": [0.017, -0.0085, -0.005], "radius": 0.012},
        {"center": [0.017, -0.0085, 0.005], "radius": 0.012},
        {"center": [0.0, -0.0085, -0.005], "radius": 0.012},
        {"center": [0.0, -0.0085, 0.005], "radius": 0.012},
        {"center": [-0.017, -0.0085, -0.005], "radius": 0.012},
        {"center": [-0.017, -0.0085, 0.005], "radius": 0.012},
    ],
    "l_gripper_finger_link": [
        {"center": [0.017, 0.0085, -0.005], "radius": 0.012},
        {"center": [0.017, 0.0085, 0.005], "radius": 0.012},
        {"center": [0.0, 0.0085, -0.005], "radius": 0.012},
        {"center": [0.0, 0.0085, 0.005], "radius": 0.012},
        {"center": [-0.017, 0.0085, -0.005], "radius": 0.012},
        {"center": [-0.017, 0.0085, 0.005], "radius": 0.012},
    ],
    "torso_fixed_link": [
        {"center": [-0.1, -0.07, 0.35], "radius": 0.12},
        {"center": [-0.1, 0.07, 0.35], "radius": 0.12},
        {"center": [-0.1, -0.07, 0.2], "radius": 0.12},
        {"center": [-0.1, 0.07, 0.2], "radius": 0.12},
        {"center": [-0.1, 0.07, 0.07], "radius": 0.12},
        {"center": [-0.1, -0.07, 0.07], "radius": 0.12},
    ],
}


def _create_transform_matrix(translation, rotation_matrix):
    """Create 4x4 transformation matrix from translation and rotation."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


class SelfOcclusionChecker:
    """
    Checks for self-occlusion of the robot with respect to a target object.
    This class is ROS-independent and computes link poses from joint angles
    using an internal forward kinematics model. All calculations are in base_link frame.
    """

    def __init__(self):
        """Initializes the SelfOcclusionChecker with Fetch kinematic parameters."""
        self.sphere_data = {
            link: [
                {"center": np.array(s["center"]), "radius": s["radius"]}
                for s in spheres
            ]
            for link, spheres in SPHERE_DATA.items()
        }

        # Fetch robot kinematic parameters from URDF
        self.torso_base_offset = np.array([-0.086875, 0.0, 0.37743])
        self.shoulder_pan_offset = np.array([0.119525, 0.0, 0.34858])
        self.shoulder_lift_offset = np.array([0.117, 0.0, 0.06])
        self.upperarm_roll_offset = np.array([0.219, 0.0, 0.0])
        self.elbow_flex_offset = np.array([0.133, 0.0, 0.0])
        self.forearm_roll_offset = np.array([0.197, 0.0, 0.0])
        self.wrist_flex_offset = np.array([0.1245, 0.0, 0.0])
        self.wrist_roll_offset = np.array([0.1385, 0.0, 0.0])
        self.gripper_offset = np.array([0.16645, 0.0, 0.0])

        # Head and auxiliary offsets
        self.torso_fixed_offset = np.array([-0.086875, 0, 0.377425])
        self.head_pan_offset = np.array([0.053125, 0, 0.603001])
        self.head_tilt_offset = np.array([0.14253, 0, 0.057999])
        self.camera_mount_offset = np.array([0.05, 0, 0])  # Approx. forward from head
        self.r_gripper_finger_offset = np.array([0, 0.065425, 0])
        self.l_gripper_finger_offset = np.array([0, -0.065425, 0])

    def _forward_kinematics(self, joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes forward kinematics for all relevant Fetch links in base_link frame.
        Args:
            joint_angles: (8,) numpy array of joint values in radians.
        Returns:
            A dictionary mapping link names to their 4x4 pose matrices.
        """
        link_poses = {}

        # Extract joint angles (assuming radians)
        (
            torso_lift,
            shoulder_pan,
            shoulder_lift,
            upperarm_roll,
            elbow_flex,
            forearm_roll,
            wrist_flex,
            wrist_roll,
        ) = joint_angles

        # Start with identity at base_link
        T_base = np.eye(4)
        link_poses["base_link"] = T_base
        link_poses["torso_fixed_link"] = T_base @ _create_transform_matrix(
            self.torso_fixed_offset, np.eye(3)
        )

        # 1. Base to torso_lift_link
        torso_translation = self.torso_base_offset.copy()
        torso_translation[2] += torso_lift
        T_torso = _create_transform_matrix(torso_translation, np.eye(3))
        T = T_base @ T_torso
        link_poses["torso_lift_link"] = T
        link_poses["torso_lift_link_collision_2"] = T

        # Head link (child of torso)
        T_head_pan = T @ _create_transform_matrix(self.head_pan_offset, np.eye(3))
        link_poses["head_pan_link"] = T_head_pan
        T_head_tilt = T_head_pan @ _create_transform_matrix(
            self.head_tilt_offset, np.eye(3)
        )
        link_poses["head_tilt_link"] = T_head_tilt

        # 2. Torso to shoulder_pan_link
        T = (
            T
            @ _create_transform_matrix(self.shoulder_pan_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("z", shoulder_pan).as_matrix()
            )
        )
        link_poses["shoulder_pan_link"] = T

        # 3. Shoulder_pan to shoulder_lift_link
        T = (
            T
            @ _create_transform_matrix(self.shoulder_lift_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("y", shoulder_lift).as_matrix()
            )
        )
        link_poses["shoulder_lift_link"] = T

        # 4. Shoulder_lift to upperarm_roll_link
        T = (
            T
            @ _create_transform_matrix(self.upperarm_roll_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("x", upperarm_roll).as_matrix()
            )
        )
        link_poses["upperarm_roll_link"] = T

        # 5. Upperarm_roll to elbow_flex_link
        T = (
            T
            @ _create_transform_matrix(self.elbow_flex_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("y", elbow_flex).as_matrix()
            )
        )
        link_poses["elbow_flex_link"] = T

        # 6. Elbow_flex to forearm_roll_link
        T = (
            T
            @ _create_transform_matrix(self.forearm_roll_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("x", forearm_roll).as_matrix()
            )
        )
        link_poses["forearm_roll_link"] = T

        # 7. Forearm_roll to wrist_flex_link
        T = (
            T
            @ _create_transform_matrix(self.wrist_flex_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("y", wrist_flex).as_matrix()
            )
        )
        link_poses["wrist_flex_link"] = T

        # 8. Wrist_flex to wrist_roll_link
        T = (
            T
            @ _create_transform_matrix(self.wrist_roll_offset, np.eye(3))
            @ _create_transform_matrix(
                np.zeros(3), R.from_euler("x", wrist_roll).as_matrix()
            )
        )
        link_poses["wrist_roll_link"] = T

        # 9. Wrist_roll to gripper_link (end-effector)
        T = T @ _create_transform_matrix(self.gripper_offset, np.eye(3))
        link_poses["gripper_link"] = T

        # Gripper fingers (fixed children of gripper_link)
        link_poses["r_gripper_finger_link"] = T @ _create_transform_matrix(
            self.r_gripper_finger_offset, np.eye(3)
        )
        link_poses["l_gripper_finger_link"] = T @ _create_transform_matrix(
            self.l_gripper_finger_offset, np.eye(3)
        )

        return link_poses

    def _compute_camera_pose_in_base(
        self, joint_angles: np.ndarray, target_center_base: np.ndarray
    ) -> np.ndarray:
        """
        Compute camera pose (4x4) in base_link frame positioned on the robot head,
        looking at the target center.
        """
        link_poses = self._forward_kinematics(joint_angles)
        # Camera position at head tilt link plus a small forward offset along head x-axis
        T_head_tilt = link_poses["head_tilt_link"]
        cam_pos = (T_head_tilt @ np.append(self.camera_mount_offset, 1))[:3]

        # Build look-at rotation
        forward = target_center_base - cam_pos
        norm_f = np.linalg.norm(forward)
        if norm_f < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / norm_f
        up = np.array([0.0, 0.0, 1.0])
        # If forward is nearly parallel to up, tweak up
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        # Camera axes: x=right, y=up, z=forward
        R_cb = np.column_stack((right, up, forward))
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R_cb
        T_cam_base[:3, 3] = cam_pos
        return T_cam_base

    def _get_world_spheres(
        self, link_poses: Dict[str, np.ndarray]
    ) -> List[Dict[str, any]]:
        """Transforms robot spheres to the base frame given the poses of the links."""
        world_spheres = []
        for link_name, spheres in self.sphere_data.items():
            if link_name in link_poses:
                link_pose = link_poses[link_name]
                for sphere in spheres:
                    center_h = np.append(sphere["center"], 1)
                    world_center = (link_pose @ center_h)[:3]
                    world_spheres.append(
                        {"center": world_center, "radius": sphere["radius"]}
                    )
        return world_spheres

    def check_occlusion(
        self,
        joint_angles: np.ndarray,
        target_center_base: np.ndarray,
        target_radius: float,
        cam_K: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> bool:
        """
        Checks for occlusion between the robot and a target sphere.
        All computations are in base_link frame; camera is on the head, looking at target.
        Args:
            joint_angles: (8,) numpy array of joint values.
            target_center_base: [x,y,z] of the target sphere center in base_link frame.
            target_radius: Target sphere radius (meters).
            cam_K: 3x3 camera intrinsic matrix.
            image_shape: (height, width) of the output image.
        Returns:
            True if robot mask intersects target mask; otherwise False.
        """
        link_poses = self._forward_kinematics(joint_angles)
        robot_spheres_base = self._get_world_spheres(link_poses)

        # Camera pose in base
        T_cam_base = self._compute_camera_pose_in_base(joint_angles, target_center_base)
        base_to_cam_R = T_cam_base[:3, :3].T
        base_to_cam_t = -base_to_cam_R @ T_cam_base[:3, 3]

        # Transform spheres to camera frame
        robot_spheres_cam = []
        for s in robot_spheres_base:
            pb = s["center"]
            pc = base_to_cam_R @ pb + base_to_cam_t
            robot_spheres_cam.append({"center": pc, "radius": s["radius"]})
        target_center_cam = base_to_cam_R @ target_center_base + base_to_cam_t
        target_sphere_cam = {"center": target_center_cam, "radius": target_radius}

        h, w = image_shape
        robot_mask = self._create_mask_from_spheres(robot_spheres_cam, cam_K, h, w)
        target_mask = self._create_mask_from_spheres([target_sphere_cam], cam_K, h, w)
        return bool(np.any(robot_mask & target_mask))

    def _create_mask_from_spheres(
        self, spheres: List[Dict[str, any]], K: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for sphere in spheres:
            center, radius = sphere["center"], sphere["radius"]
            if center[2] <= 0:
                continue
            u = int(fx * center[0] / center[2] + cx)
            v = int(fy * center[1] / center[2] + cy)
            pixel_radius = max(1, int(fx * radius / center[2]))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(mask, (u, v), pixel_radius, 1, -1)

        return mask.astype(bool)


if __name__ == "__main__":
    import open3d as o3d

    # --- Example joint configuration (8-DoF) ---
    mock_joint_angles = np.array(
        [
            0.2,  # torso_lift
            0.5,  # shoulder_pan
            -0.8,  # shoulder_lift
            0.1,  # upperarm_roll
            1.2,  # elbow_flex
            -0.2,  # forearm_roll
            0.8,  # wrist_flex
            0.1,  # wrist_roll
        ]
    )

    # Target object in base_link frame
    target_center_base = np.array([0.8, 0.2, 0.7])
    target_radius = 0.05

    # Camera intrinsics
    img_shape = (480, 640)
    cam_K = np.array([[554.25, 0, 320.5], [0, 554.25, 240.5], [0, 0, 1]])

    checker = SelfOcclusionChecker()

    # --- 3D visualization in base frame ---
    link_poses = checker._forward_kinematics(mock_joint_angles)
    robot_spheres_base = checker._get_world_spheres(link_poses)

    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    ]
    for s in robot_spheres_base:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=s["radius"])
        m.translate(s["center"])
        m.paint_uniform_color([0.1, 0.1, 0.7])
        geometries.append(m)
    tgt = o3d.geometry.TriangleMesh.create_sphere(radius=target_radius)
    tgt.translate(target_center_base)
    tgt.paint_uniform_color([0.7, 0.1, 0.1])
    geometries.append(tgt)

    T_cam_base = checker._compute_camera_pose_in_base(
        mock_joint_angles, target_center_base
    )
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    cam_frame.transform(T_cam_base)
    geometries.append(cam_frame)

    print("Visualizing 3D scene (base_link frame). Press 'q' to close.")
    o3d.visualization.draw_geometries(geometries)

    # --- 2D projection in camera frame ---
    is_occ = checker.check_occlusion(
        joint_angles=mock_joint_angles,
        target_center_base=target_center_base,
        target_radius=target_radius,
        cam_K=cam_K,
        image_shape=img_shape,
    )

    # For display, regenerate masks
    link_poses = checker._forward_kinematics(mock_joint_angles)
    robot_spheres_base = checker._get_world_spheres(link_poses)
    R_cb = T_cam_base[:3, :3]
    t_cb = T_cam_base[:3, 3]
    base_to_cam_R = R_cb.T
    base_to_cam_t = -base_to_cam_R @ t_cb
    robot_spheres_cam = [
        {"center": base_to_cam_R @ s["center"] + base_to_cam_t, "radius": s["radius"]}
        for s in robot_spheres_base
    ]
    target_center_cam = base_to_cam_R @ target_center_base + base_to_cam_t

    robot_mask = checker._create_mask_from_spheres(robot_spheres_cam, cam_K, *img_shape)
    target_mask = checker._create_mask_from_spheres(
        [{"center": target_center_cam, "radius": target_radius}], cam_K, *img_shape
    )

    h, w = img_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[robot_mask] = [255, 0, 0]
    vis[target_mask] = [0, 0, 255]
    vis[robot_mask & target_mask] = [255, 0, 255]

    title = f"2D Projection (Occlusion: {is_occ})"
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
