import os
from typing import List, Optional, Tuple

import numpy as np

# ROS-free TRAC-IK bindings, built from third_party/pytracik (pybind11 + CMake).
# Replaces the former `from trac_ik_python.trac_ik import IK` ROS dependency.
import pytracik

_SOLVE_TYPES = {
    "Speed": pytracik.SolveType.Speed,
    "Distance": pytracik.SolveType.Distance,
    "Manip1": pytracik.SolveType.Manip1,
    "Manip2": pytracik.SolveType.Manip2,
}


class TracIKSolver:
    """
    TRAC-IK based solver wrapper: accepts seed, solves nearest IK.

    Backed by the vendored, ROS-free ``pytracik`` module (third_party/pytracik)
    instead of the ``trac_ik_python`` ROS package. The public interface is
    unchanged.
    """

    def __init__(
        self,
        base_link: str,
        ee_link: str,
        urdf_string: Optional[str] = None,
        urdf_path: Optional[str] = None,
        timeout: float = 0.2,
        epsilon: float = 1e-6,
    ) -> None:
        if urdf_string is None and urdf_path is not None:
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            with open(urdf_path, "r") as f:
                urdf_string = f.read()
        if urdf_string is None:
            raise ValueError("Either urdf_string or urdf_path must be provided.")

        self._solver = pytracik.TRAC_IK(
            base_link,
            ee_link,
            urdf_string,
            timeout,
            epsilon,
            _SOLVE_TYPES["Distance"],
        )

        self._lower_limits: Tuple[float, ...] = tuple(
            pytracik.get_joint_lower_bounds(self._solver)
        )
        self._upper_limits: Tuple[float, ...] = tuple(
            pytracik.get_joint_upper_bounds(self._solver)
        )

    @property
    def num_joints(self) -> int:
        return len(self._lower_limits)

    @property
    def joint_limits(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        return self._lower_limits, self._upper_limits

    def get_ik_near_seed(
        self,
        seed: List[float],
        position: List[float],
        orientation_xyzw: List[float],
    ) -> Optional[List[float]]:
        if len(seed) != self.num_joints:
            raise ValueError(
                f"Seed length {len(seed)} does not match solver DOF {self.num_joints}."
            )

        # pytracik.ik returns [status, q0, q1, ...]; status < 0 means no solution.
        result = np.asarray(
            pytracik.ik(
                self._solver,
                np.asarray(seed, dtype=np.float64),
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(orientation_xyzw[0]),
                float(orientation_xyzw[1]),
                float(orientation_xyzw[2]),
                float(orientation_xyzw[3]),
            )
        ).reshape(-1)

        if result.size >= 1 and result[0] >= 0:
            return result[1:].tolist()
        return None
