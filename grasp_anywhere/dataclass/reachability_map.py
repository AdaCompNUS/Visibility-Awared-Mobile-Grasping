import pickle
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import KDTree


@dataclass(frozen=True)
class CapabilityMap:
    """Dataclass to store forward reachability map data.

    Attributes:
        poses: Array of poses (N x 6) containing [x, y, z, roll, pitch, yaw]
        scores: Array of reachability scores (N,) for each pose
        kdtree: KDTree built from pose positions for efficient spatial queries

    Example:
        # Load reachability map
        capability_map = CapabilityMap.from_file("path/to/capability_map.pkl")

        # Access the data
        print(f"Loaded {len(capability_map.poses)} poses")
        print(f"First pose: {capability_map.poses[0]}")
        print(f"First score: {capability_map.scores[0]}")
    """

    poses: np.ndarray
    scores: np.ndarray
    kdtree: KDTree

    @classmethod
    def from_file(cls, map_path: str) -> "CapabilityMap":
        """Load reachability map from pickle file.

        Args:
            map_path: Path to the forward reachability map pickle file

        Returns:
            CapabilityMap instance with loaded data

        Example:
            capability_map = CapabilityMap.from_file("resources/capability_map.pkl")
        """
        with open(map_path, "rb") as f:
            data: np.ndarray = pickle.load(f)

        poses = data[:, :6]
        scores = data[:, 7]

        # Build KDTree using position-only (x, y, z) for efficient spatial queries
        kdtree = KDTree(poses[:, :3])

        return cls(poses=poses, scores=scores, kdtree=kdtree)


@dataclass(frozen=True)
class ReachabilityMap:
    """Dataclass to store a sparse, reachability voxel map.

    This map marginalizes out end-effector orientation and only stores whether
    the end-effector *position* (x, y, z) is reachable (with an associated score).

    Storage is sparse: only reachable voxels are stored.

    Attributes:
        origin: (3,) xyz origin of the voxel grid in the reachability-map frame.
        voxel_size: Edge length of each voxel (meters).
        voxels_ijk: (M, 3) integer voxel indices for reachable voxels.
        scores: (M,) reachability score per voxel (typically max over orientations).
    """

    origin: np.ndarray
    voxel_size: float
    voxels_ijk: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_file(cls, map_path: str) -> "ReachabilityMap":
        """Load a reachability voxel map from a pickle file."""
        with open(map_path, "rb") as f:
            data = pickle.load(f)

        origin = np.asarray(data.get("origin"), dtype=np.float32)
        voxel_size = float(data.get("voxel_size"))
        voxels_ijk = np.asarray(data.get("voxels_ijk"), dtype=np.int32)
        scores = np.asarray(data.get("scores"), dtype=np.float32)

        return cls(
            origin=origin,
            voxel_size=voxel_size,
            voxels_ijk=voxels_ijk,
            scores=scores,
        )


@dataclass(frozen=True)
class InverseReachabilityMap:
    """Dataclass to store inverse reachability map (IRM) data.

    The inverse reachability map stores transformations that allow querying
    which base poses can reach a given target position.

    Attributes:
        inv_transf_batch: Batch of inverse transformations (N x 4 x 4 torch tensors)
        M_scores: Reachability scores corresponding to each transformation (N, torch tensor)

    Example:
        # Load inverse reachability map
        irm = InverseReachabilityMap.from_file("path/to/inv_reach_map.pkl")

        # Access the data
        print(f"Loaded {len(irm.inv_transf_batch)} transformations")
        print(f"First transformation shape: {irm.inv_transf_batch[0].shape}")
        print(f"First score: {irm.M_scores[0]}")
    """

    inv_transf_batch: torch.Tensor
    M_scores: torch.Tensor

    @classmethod
    def from_file(cls, irm_map_path: str) -> "InverseReachabilityMap":
        """Load inverse reachability map from pickle file.

        Args:
            irm_map_path: Path to the inverse reachability map pickle file

        Returns:
            InverseReachabilityMap instance with loaded data

        Example:
            irm = InverseReachabilityMap.from_file("resources/inv_reach_map.pkl")
        """
        with open(irm_map_path, "rb") as f:
            irm_dict = pickle.load(f)

        inv_transf_batch = irm_dict["inv_transf_batch"]
        M_scores = irm_dict["M_scores"]

        return cls(inv_transf_batch=inv_transf_batch, M_scores=M_scores)
