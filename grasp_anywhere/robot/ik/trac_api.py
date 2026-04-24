import os
from typing import List, Optional, Tuple

import numpy as np
import tf.transformations as transformations

from grasp_anywhere.robot.ik.trac_ik_solver import TracIKSolver

_trac_solver_full: Optional[TracIKSolver] = None
_trac_solver_arm: Optional[TracIKSolver] = None
_trac_solver_arm_only: Optional[TracIKSolver] = None
_trac_auto_initialized: bool = False


def _default_urdf_path() -> str:
    # Resolve to project resources path: <repo_root>/resources/fetch_ext/fetch ext.urdf
    here = os.path.dirname(__file__)
    # Go up three levels from grasp_anywhere/robot/ik -> repo root
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo_root, "resources", "fetch_ext", "fetch ext.urdf")


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def _auto_init_if_needed() -> None:
    global _trac_auto_initialized
    if _trac_auto_initialized:
        return
    # Allow override via env var
    urdf_path_env = os.environ.get("FETCH_URDF_PATH")
    urdf_path = urdf_path_env if urdf_path_env else _default_urdf_path()
    urdf_str = _read_text_file(urdf_path)
    try:
        # Initialize all common modes by default
        trac_init_fixed_base(urdf_string=urdf_str, urdf_path=None)
        trac_init_arm_only(urdf_string=urdf_str, urdf_path=None)
        # Floating base is optional; initialize as well for completeness
        trac_init_floating_base(urdf_string=urdf_str, urdf_path=None)
        _trac_auto_initialized = True
    except Exception:
        # Defer initialization to explicit calls if automatic init fails
        _trac_auto_initialized = False


def _ensure_full() -> TracIKSolver:
    if _trac_solver_full is None:
        raise RuntimeError(
            "TRAC-IK floating-base not initialized. Call trac_init_floating_base()."
        )
    return _trac_solver_full


def _ensure_arm() -> TracIKSolver:
    if _trac_solver_arm is None:
        raise RuntimeError(
            "TRAC-IK fixed-base not initialized. Call trac_init_fixed_base()."
        )
    return _trac_solver_arm


def _ensure_arm_only() -> TracIKSolver:
    if _trac_solver_arm_only is None:
        raise RuntimeError(
            "TRAC-IK arm-only not initialized. Call trac_init_arm_only()."
        )
    return _trac_solver_arm_only


def trac_init_floating_base(
    urdf_string: Optional[str] = None,
    urdf_path: Optional[str] = "resources/fetch_ext/fetch ext.urdf",
    timeout: float = 0.2,
    epsilon: float = 1e-6,
) -> None:
    global _trac_solver_full
    if urdf_string is None and urdf_path is None:
        urdf_path = _default_urdf_path()
        urdf_string = _read_text_file(urdf_path)
    _trac_solver_full = TracIKSolver(
        base_link="world_link",
        ee_link="gripper_link",
        urdf_string=urdf_string,
        urdf_path=urdf_path,
        timeout=timeout,
        epsilon=epsilon,
    )


def trac_init_fixed_base(
    urdf_string: Optional[str] = None,
    urdf_path: Optional[str] = "resources/fetch_ext/fetch ext.urdf",
    timeout: float = 0.2,
    epsilon: float = 1e-6,
) -> None:
    global _trac_solver_arm
    if urdf_string is None and urdf_path is None:
        urdf_path = _default_urdf_path()
        urdf_string = _read_text_file(urdf_path)
    _trac_solver_arm = TracIKSolver(
        base_link="base_link",
        ee_link="gripper_link",
        urdf_string=urdf_string,
        urdf_path=urdf_path,
        timeout=timeout,
        epsilon=epsilon,
    )


def trac_init_arm_only(
    urdf_string: Optional[str] = None,
    urdf_path: Optional[str] = "resources/fetch_ext/fetch ext.urdf",
    timeout: float = 0.2,
    epsilon: float = 1e-6,
) -> None:
    global _trac_solver_arm_only
    if urdf_string is None and urdf_path is None:
        urdf_path = _default_urdf_path()
        urdf_string = _read_text_file(urdf_path)
    _trac_solver_arm_only = TracIKSolver(
        base_link="torso_lift_link",
        ee_link="gripper_link",
        urdf_string=urdf_string,
        urdf_path=urdf_path,
        timeout=timeout,
        epsilon=epsilon,
    )


def trac_limits_floating_base() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    _auto_init_if_needed()
    return _ensure_full().joint_limits


def trac_limits_fixed_base() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    _auto_init_if_needed()
    return _ensure_arm().joint_limits


def trac_limits_arm_only() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    _auto_init_if_needed()
    return _ensure_arm_only().joint_limits


def trac_solve_floating_base_full(
    seed_full: List[float],
    position_world: List[float],
    orientation_world_xyzw: List[float],
) -> Optional[List[float]]:
    _auto_init_if_needed()
    return _ensure_full().get_ik_near_seed(
        seed_full, position_world, orientation_world_xyzw
    )


def trac_solve_fixed_base_arm(
    seed_arm8: List[float],
    base_config_xytheta: List[float],
    position_world: List[float],
    orientation_world_xyzw: List[float],
) -> Optional[List[float]]:
    _auto_init_if_needed()
    base_x, base_y, base_theta = base_config_xytheta
    T_world_base = transformations.euler_matrix(0, 0, base_theta)
    T_world_base[0:3, 3] = [base_x, base_y, 0]
    T_world_ee = transformations.quaternion_matrix(orientation_world_xyzw)
    T_world_ee[0:3, 3] = position_world
    T_base_world = transformations.inverse_matrix(T_world_base)
    T_base_ee = np.dot(T_base_world, T_world_ee)
    pos_b = transformations.translation_from_matrix(T_base_ee)
    quat_b = transformations.quaternion_from_matrix(T_base_ee)
    return _ensure_arm().get_ik_near_seed(
        seed_arm8,
        [float(pos_b[0]), float(pos_b[1]), float(pos_b[2])],
        [float(quat_b[0]), float(quat_b[1]), float(quat_b[2]), float(quat_b[3])],
    )


def trac_solve_arm_only(
    seed_arm7: List[float],
    base_config_xytheta: List[float],
    torso_position: float,
    position_world: List[float],
    orientation_world_xyzw: List[float],
) -> Optional[List[float]]:
    _auto_init_if_needed()
    base_x, base_y, base_theta = base_config_xytheta
    T_world_base = transformations.euler_matrix(0, 0, base_theta)
    T_world_base[0:3, 3] = [base_x, base_y, 0]
    torso_lift_link_offset = [-0.086875, 0, 0.37743]
    T_base_torso = transformations.translation_matrix(torso_lift_link_offset)
    T_base_torso[2, 3] += torso_position
    T_world_torso = np.dot(T_world_base, T_base_torso)
    T_world_ee = transformations.quaternion_matrix(orientation_world_xyzw)
    T_world_ee[0:3, 3] = position_world
    T_torso_world = transformations.inverse_matrix(T_world_torso)
    T_torso_ee = np.dot(T_torso_world, T_world_ee)
    pos_t = transformations.translation_from_matrix(T_torso_ee)
    quat_t = transformations.quaternion_from_matrix(T_torso_ee)
    return _ensure_arm_only().get_ik_near_seed(
        seed_arm7,
        [float(pos_t[0]), float(pos_t[1]), float(pos_t[2])],
        [float(quat_t[0]), float(quat_t[1]), float(quat_t[2]), float(quat_t[3])],
    )


# Attempt automatic initialization on import
_auto_init_if_needed()
