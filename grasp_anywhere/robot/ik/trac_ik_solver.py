import os
from typing import List, Optional, Tuple

from trac_ik_python.trac_ik import IK


class TracIKSolver:
    """
    TRAC-IK based solver wrapper: accepts seed, solves nearest IK.
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

        self._solver = IK(
            base_link,
            ee_link,
            urdf_string=urdf_string,
            timeout=timeout,
            epsilon=epsilon,
            solve_type="Distance",
        )

        limits = self._solver.get_joint_limits()
        self._lower_limits: Tuple[float, ...] = limits[0]
        self._upper_limits: Tuple[float, ...] = limits[1]

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

        sol = self._solver.get_ik(
            seed,
            position[0],
            position[1],
            position[2],
            orientation_xyzw[0],
            orientation_xyzw[1],
            orientation_xyzw[2],
            orientation_xyzw[3],
        )
        if sol:
            return list(sol)
        return None
