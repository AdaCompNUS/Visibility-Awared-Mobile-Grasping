from typing import Optional

import numpy as np

import ikfast_fetch

# Joint limits for Fetch robot (8 joints)
# Order: torso_lift_joint, shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint,
# elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint
JOINT_LIMITS_LOWER = np.array(
    [0.0, -1.6056, -1.221, -np.pi, -2.251, -np.pi, -2.16, -np.pi]
)
JOINT_LIMITS_UPPER = np.array(
    [0.38615, 1.6056, 1.518, np.pi, 2.251, np.pi, 2.16, np.pi]
)


def is_solution_valid(solution):
    """Check if a solution is within joint limits."""
    solution = np.array(solution)
    return np.all(solution >= JOINT_LIMITS_LOWER) and np.all(
        solution <= JOINT_LIMITS_UPPER
    )


def compute_ik(end_effector_pose, free_params: Optional[list[float]] = None):
    """
    Compute inverse kinematics solutions.

    Args:
        end_effector_pose: 4x4 transformation matrix or [position, rotation_matrix]
        free_params: List of free parameters 'torso_lift_joint', 'shoulder_lift_joint'

    Returns:
        List of joint angle solutions (each solution has 8 joints)
    """
    if free_params is None:
        # Randomly sample free parameters if not provided.
        # Expected order: [torso_lift_joint, shoulder_lift_joint].
        torso_lift = np.random.uniform(JOINT_LIMITS_LOWER[0], JOINT_LIMITS_UPPER[0])
        shoulder_lift = np.random.uniform(JOINT_LIMITS_LOWER[2], JOINT_LIMITS_UPPER[2])
        free_params = [torso_lift, shoulder_lift]

    if isinstance(end_effector_pose, np.ndarray) and end_effector_pose.shape == (4, 4):
        position = end_effector_pose[:3, 3].tolist()
        rotation = end_effector_pose[:3, :3].tolist()  # Keep as 3x3 matrix
    else:
        position, rotation = end_effector_pose
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                rotation = rotation.tolist()  # Keep as 3x3 matrix
            else:
                rotation = rotation.reshape(3, 3).tolist()  # Reshape if flattened
        else:
            # Assume it's a flattened list, reshape to 3x3
            rotation = np.array(rotation).reshape(3, 3).tolist()

    solutions = ikfast_fetch.get_ik(rotation, position, free_params)

    # Filter out invalid solutions
    if solutions:
        solutions = [sol for sol in solutions if is_solution_valid(sol)]

    return solutions


def compute_fk(joint_angles):
    """
    Compute forward kinematics.

    Args:
        joint_angles: List of 8 joint angles

    Returns:
        [position, rotation_matrix] where position is [x,y,z] and rotation_matrix is 3x3
    """
    result = ikfast_fetch.get_fk(joint_angles)
    position = result[0]
    rotation = np.array(result[1]).reshape(3, 3)
    return [position, rotation]
