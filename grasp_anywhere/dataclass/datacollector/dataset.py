from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EpisodeDataset:
    """
    Handle container for an open episode HDF5 file and its datasets.

    Kept as a dataclass so integration code can pass a single object around, while
    the actual logic remains functional.
    """

    path: Path
    h5: object  # h5py.File
    n: int

    # obs
    ds_rgb: object  # h5py.Dataset
    ds_depth: object  # h5py.Dataset

    # state
    ds_qpos: object  # h5py.Dataset
    ds_head_qpos: object  # h5py.Dataset
    ds_ee_pose_world: object  # h5py.Dataset

    # action (populated at finalize)
    g_action: object  # h5py.Group


@dataclass
class EpisodeCache:
    """
    In-memory episode buffer. We cache all observations and states, then compute
    actions and write a single HDF5 file at the end (no extendable datasets).
    """

    path: Path
    goal_xyz_world: object  # np.ndarray (3,)

    # cached per-timestep
    rgbs: object  # list[np.ndarray]
    depths: object  # list[np.ndarray]
    qpos: object  # list[np.ndarray]
    head_qpos: object  # list[np.ndarray]
