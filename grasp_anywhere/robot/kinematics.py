from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

# Fetch robot kinematic parameters from URDF
TORSO_BASE_OFFSET = np.array([-0.086875, 0.0, 0.37743])
SHOULDER_PAN_OFFSET = np.array([0.119525, 0.0, 0.34858])
SHOULDER_LIFT_OFFSET = np.array([0.117, 0.0, 0.06])
UPPERARM_ROLL_OFFSET = np.array([0.219, 0.0, 0.0])
ELBOW_FLEX_OFFSET = np.array([0.133, 0.0, 0.0])
FOREARM_ROLL_OFFSET = np.array([0.197, 0.0, 0.0])
WRIST_FLEX_OFFSET = np.array([0.1245, 0.0, 0.0])
WRIST_ROLL_OFFSET = np.array([0.1385, 0.0, 0.0])
GRIPPER_OFFSET = np.array([0.16645, 0.0, 0.0])
# Head and auxiliary offsets
TORSO_FIXED_OFFSET = np.array([-0.086875, 0, 0.377425])
HEAD_PAN_OFFSET = np.array([0.053125, 0, 0.603001])
HEAD_TILT_OFFSET = np.array([0.14253, 0, 0.057999])
R_GRIPPER_FINGER_OFFSET = np.array([0, 0.065425, 0])
L_GRIPPER_FINGER_OFFSET = np.array([0, -0.065425, 0])


def _create_transform_matrix(translation, rotation_matrix):
    """Create 4x4 transformation matrix from translation and rotation."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def forward_kinematics(joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes forward kinematics for all relevant Fetch links in base_link frame.
    Args:
        joint_angles: (10,) numpy array of joint values in radians for
                        [torso, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex,
                        forearm_roll, wrist_flex, wrist_roll, head_pan, head_tilt].
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
        head_pan,
        head_tilt,
    ) = joint_angles

    # Start with identity at base_link
    T_base = np.eye(4)
    link_poses["base_link"] = T_base
    link_poses["torso_fixed_link"] = T_base @ _create_transform_matrix(
        TORSO_FIXED_OFFSET, np.eye(3)
    )

    # 1. Base to torso_lift_link
    torso_translation = TORSO_BASE_OFFSET.copy()
    torso_translation[2] += torso_lift
    T_torso = _create_transform_matrix(torso_translation, np.eye(3))
    T = T_base @ T_torso
    link_poses["torso_lift_link"] = T

    # Head link (child of torso)
    T_head_pan = (
        T
        @ _create_transform_matrix(HEAD_PAN_OFFSET, np.eye(3))
        @ _create_transform_matrix(np.zeros(3), R.from_euler("z", head_pan).as_matrix())
    )
    link_poses["head_pan_link"] = T_head_pan

    T_head_tilt = (
        T_head_pan
        @ _create_transform_matrix(HEAD_TILT_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("y", head_tilt).as_matrix()
        )
    )
    link_poses["head_tilt_link"] = T_head_tilt

    # 2. Torso to shoulder_pan_link
    T = (
        T
        @ _create_transform_matrix(SHOULDER_PAN_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("z", shoulder_pan).as_matrix()
        )
    )
    link_poses["shoulder_pan_link"] = T

    # 3. Shoulder_pan to shoulder_lift_link
    T = (
        T
        @ _create_transform_matrix(SHOULDER_LIFT_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("y", shoulder_lift).as_matrix()
        )
    )
    link_poses["shoulder_lift_link"] = T

    # 4. Shoulder_lift to upperarm_roll_link
    T = (
        T
        @ _create_transform_matrix(UPPERARM_ROLL_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("x", upperarm_roll).as_matrix()
        )
    )
    link_poses["upperarm_roll_link"] = T

    # 5. Upperarm_roll to elbow_flex_link
    T = (
        T
        @ _create_transform_matrix(ELBOW_FLEX_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("y", elbow_flex).as_matrix()
        )
    )
    link_poses["elbow_flex_link"] = T

    # 6. Elbow_flex to forearm_roll_link
    T = (
        T
        @ _create_transform_matrix(FOREARM_ROLL_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("x", forearm_roll).as_matrix()
        )
    )
    link_poses["forearm_roll_link"] = T

    # 7. Forearm_roll to wrist_flex_link
    T = (
        T
        @ _create_transform_matrix(WRIST_FLEX_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("y", wrist_flex).as_matrix()
        )
    )
    link_poses["wrist_flex_link"] = T

    # 8. Wrist_flex to wrist_roll_link
    T = (
        T
        @ _create_transform_matrix(WRIST_ROLL_OFFSET, np.eye(3))
        @ _create_transform_matrix(
            np.zeros(3), R.from_euler("x", wrist_roll).as_matrix()
        )
    )
    link_poses["wrist_roll_link"] = T

    # 9. Wrist_roll to gripper_link (end-effector)
    T = T @ _create_transform_matrix(GRIPPER_OFFSET, np.eye(3))
    link_poses["gripper_link"] = T

    # Gripper fingers (fixed children of gripper_link)
    link_poses["r_gripper_finger_link"] = T @ _create_transform_matrix(
        R_GRIPPER_FINGER_OFFSET, np.eye(3)
    )
    link_poses["l_gripper_finger_link"] = T @ _create_transform_matrix(
        L_GRIPPER_FINGER_OFFSET, np.eye(3)
    )

    return link_poses
