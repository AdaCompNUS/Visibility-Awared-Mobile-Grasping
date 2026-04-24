from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from grasp_anywhere.data_collector.h5_utils import (
    h5_create_file,
)
from grasp_anywhere.dataclass.datacollector.config import DataCollectionConfig
from grasp_anywhere.dataclass.datacollector.dataset import EpisodeCache
from grasp_anywhere.robot.kinematics import forward_kinematics


def episode_make_path_simple(cfg: DataCollectionConfig, *, ts_ns: int) -> Path:
    fname = f"episode_{int(ts_ns)}.h5"
    return cfg.run_dir() / fname


def ee_pose_world_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """
    Compute EE pose in world frame from a "whole-body" qpos vector:
      qpos = [x, y, yaw, torso, 7 arm]  -> shape (11,)

    This uses the existing Fetch kinematics model (base_link frame) and applies
    the world base pose (x,y,yaw).
    """
    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if int(q.shape[0]) != 11:
        raise ValueError("Expected qpos shape (11,) = [x,y,yaw,torso,7arm].")

    x, y, yaw = float(q[0]), float(q[1]), float(q[2])
    torso_arm = q[3:11].astype(np.float32)  # (8,)
    # forward_kinematics expects 10 joints: [torso, 7 arm, head_pan, head_tilt]
    joint10 = np.concatenate([torso_arm, np.zeros((2,), dtype=np.float32)], axis=0)
    poses = forward_kinematics(joint10)
    T_base_ee = poses["gripper_link"].astype(np.float32)

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    T_world_base = np.eye(4, dtype=np.float32)
    T_world_base[0, 0] = c
    T_world_base[0, 1] = -s
    T_world_base[1, 0] = s
    T_world_base[1, 1] = c
    T_world_base[0, 3] = x
    T_world_base[1, 3] = y
    return (T_world_base @ T_base_ee).astype(np.float32)


def delta_ee_action_6d(T0: np.ndarray, T1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Action over the interval: delta position + 6D rotation representation.

    Rotation 6D is the first two rows of R_delta, flattened:
      R_delta = R0^T R1
      rot6d = R_delta[:2, :].reshape(6)
    """
    A = np.asarray(T0, dtype=np.float32)
    B = np.asarray(T1, dtype=np.float32)
    dpos = (B[:3, 3] - A[:3, 3]).astype(np.float32)

    R0 = A[:3, :3]
    R1 = B[:3, :3]
    R_delta = (R0.T @ R1).astype(np.float32)
    rot6d = R_delta[:2, :].reshape(6).astype(np.float32)
    return dpos, rot6d


def episode_create(
    cfg: DataCollectionConfig,
    *,
    goal_xyz_world: np.ndarray,
    qpos_dim: int,
    rgb_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
) -> EpisodeCache:
    # qpos_dim/rgb_shape/depth_shape are kept in the signature to make the caller
    # explicit about shapes, but we cache arrays in memory and write once at the end.
    _ = qpos_dim, rgb_shape, depth_shape
    path = episode_make_path_simple(cfg, ts_ns=time.time_ns())
    return EpisodeCache(
        path=path,
        goal_xyz_world=np.asarray(goal_xyz_world, dtype=np.float32).reshape(3),
        rgbs=[],
        depths=[],
        qpos=[],
        head_qpos=[],
    )


def episode_append_state(
    ds: EpisodeCache,
    rgb: np.ndarray,
    depth: np.ndarray,
    qpos: np.ndarray,
    head_qpos: np.ndarray,
) -> None:
    ds.rgbs.append(np.asarray(rgb, dtype=np.uint8))
    ds.depths.append(np.asarray(depth, dtype=np.float32))
    ds.qpos.append(np.asarray(qpos, dtype=np.float32).reshape(-1))
    ds.head_qpos.append(np.asarray(head_qpos, dtype=np.float32).reshape(-1))


def episode_finalize_to_h5(
    ds: EpisodeCache, cfg: DataCollectionConfig, *, dt: float
) -> Path:
    """
    Compute EE state and delta actions for the whole episode and write one HDF5 file.
    """
    if cfg.disabled:
        return ds.path

    T = int(len(ds.qpos))
    if T < 2:
        raise ValueError("Need at least 2 timesteps to compute delta actions.")

    rgbs = np.stack(ds.rgbs, axis=0)  # (T,H,W,3)
    depths = np.stack(ds.depths, axis=0)  # (T,H,W)
    qpos = np.stack(ds.qpos, axis=0)  # (T,D)
    head_qpos = np.stack(ds.head_qpos, axis=0)  # (T, 2)

    ee = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        ee[i] = ee_pose_world_from_qpos(qpos[i])

    delta_pos = np.zeros((T - 1, 3), dtype=np.float32)
    delta_rot6d = np.zeros((T - 1, 6), dtype=np.float32)
    for i in range(T - 1):
        dpos, rot6d = delta_ee_action_6d(ee[i], ee[i + 1])
        delta_pos[i] = dpos
        delta_rot6d[i] = rot6d

    h5 = h5_create_file(ds.path)
    g_goal = h5.create_group("goal")
    g_obs = h5.create_group("obs")
    g_state = h5.create_group("state")
    g_action = h5.create_group("action")

    g_goal.create_dataset(
        "target_xyz_world", data=np.asarray(ds.goal_xyz_world, dtype=np.float32)
    )

    g_obs.create_dataset("rgb", data=rgbs)
    g_obs.create_dataset("depth", data=depths)

    g_state.create_dataset("qpos", data=qpos)
    g_state.create_dataset("head_qpos", data=head_qpos)
    g_state.create_dataset("ee_pose_world", data=ee)

    g_action.create_dataset("delta_ee_pos", data=delta_pos)
    g_action.create_dataset("delta_ee_rot_6d", data=delta_rot6d)
    g_action.create_dataset("dt", data=np.full((T - 1,), float(dt), dtype=np.float32))

    h5.flush()
    h5.close()
    return ds.path


def episode_record(
    ds: EpisodeCache | None,
    *,
    cfg: DataCollectionConfig,
    goal_xyz_world: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    qpos: np.ndarray,
    head_qpos: np.ndarray,
) -> EpisodeCache:
    """
    Single-call interface for per-timestep episode collection.

    - If `ds` is None: create a new in-memory episode cache (timestamp-named file path).
    - Always appends the provided (rgb, depth, qpos) as the current timestep.
    """
    if ds is None:
        ds = episode_create(
            cfg,
            goal_xyz_world=np.asarray(goal_xyz_world, dtype=np.float32).reshape(3),
            qpos_dim=int(np.asarray(qpos).reshape(-1).shape[0]),
            rgb_shape=tuple(np.asarray(rgb).shape),
            depth_shape=tuple(np.asarray(depth).shape),
        )

    if not cfg.disabled:
        episode_append_state(ds, rgb=rgb, depth=depth, qpos=qpos, head_qpos=head_qpos)

    return ds


def episode_finish(ds: EpisodeCache, *, cfg: DataCollectionConfig, dt: float) -> Path:
    """
    Finalize an episode cache: compute EE states + delta actions and write HDF5.
    """
    return episode_finalize_to_h5(ds, cfg, dt=dt)
