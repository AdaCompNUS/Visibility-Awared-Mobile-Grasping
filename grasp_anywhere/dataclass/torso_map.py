from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TorsoMap:
    xyz_step: float
    best_torso: np.ndarray  # (M,) float32/float64 meters
    xyz_to_index: dict[tuple[int, int, int], int]

    @classmethod
    def from_file(cls, path: str) -> "TorsoMap":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        required = [
            "xyz_step",
            "xyz_ijk",
            "best_torso",
        ]
        for k in required:
            if k not in payload:
                raise KeyError(
                    f"Torso map missing key '{k}'. Present keys: {list(payload.keys())}"
                )

        xyz_ijk = np.asarray(payload["xyz_ijk"], dtype=np.int32)
        best_torso = np.asarray(payload["best_torso"], dtype=np.float32)

        if xyz_ijk.ndim != 2 or xyz_ijk.shape[1] != 3:
            raise ValueError(f"xyz_ijk must be (M,3). Got {xyz_ijk.shape}")
        if best_torso.shape[0] != xyz_ijk.shape[0]:
            raise ValueError("best_torso length mismatch")

        # Note: M can be large; this is still typically faster than KDTree for exact-bin queries.
        xyz_to_index: dict[tuple[int, int, int], int] = {
            tuple(v.tolist()): i for i, v in enumerate(xyz_ijk)
        }

        return cls(
            xyz_step=float(payload["xyz_step"]),
            best_torso=best_torso,
            xyz_to_index=xyz_to_index,
        )
