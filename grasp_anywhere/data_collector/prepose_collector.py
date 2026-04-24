from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from grasp_anywhere.data_collector.h5_utils import (
    h5_create_file,
)
from grasp_anywhere.dataclass.datacollector.config import DataCollectionConfig


def prepose_make_path_simple(cfg: DataCollectionConfig, *, ts_ns: int) -> Path:
    fname = f"prepose_{int(ts_ns)}.h5"
    return cfg.run_dir() / fname


def prepose_write_h5(
    cfg: DataCollectionConfig,
    base_config: np.ndarray,
    arm_config: np.ndarray,
    ee_pose_world: np.ndarray,
    ee_pose_obj: np.ndarray,
    points_obj: np.ndarray,
    target_xyz_world: np.ndarray,
    object_center_world: np.ndarray,
    radius: float,
) -> Path:
    """
    Minimal prepose writer for sampler training.
    - No scene/task/seed metadata.
    - Stores object-centered pointcloud + the corresponding prepose label.
    """
    ts_ns = time.time_ns()
    path = prepose_make_path_simple(cfg, ts_ns=ts_ns)
    h5 = h5_create_file(path)

    g_meta = h5.create_group("meta")
    g_robot = h5.create_group("robot")
    g_env = h5.create_group("env")

    g_meta.create_dataset(
        "target_xyz_world", data=np.asarray(target_xyz_world, dtype=np.float32)
    )
    g_meta.create_dataset(
        "object_center_world", data=np.asarray(object_center_world, dtype=np.float32)
    )
    g_meta.create_dataset("radius", data=np.asarray(float(radius), dtype=np.float32))

    g_robot.create_dataset(
        "base_config", data=np.asarray(base_config, dtype=np.float32)
    )
    g_robot.create_dataset("arm_config", data=np.asarray(arm_config, dtype=np.float32))
    g_robot.create_dataset(
        "ee_pose_world", data=np.asarray(ee_pose_world, dtype=np.float32)
    )
    g_robot.create_dataset(
        "ee_pose_obj", data=np.asarray(ee_pose_obj, dtype=np.float32)
    )

    g_env.create_dataset("points_obj", data=np.asarray(points_obj, dtype=np.float32))

    h5.flush()
    h5.close()
    return path


def prepose_record(
    *,
    combined_points: np.ndarray,
    object_center_world: np.ndarray,
    radius: float,
    prepose_matrix: np.ndarray,
    base_config: np.ndarray,
    arm_config: np.ndarray,
    run_name: str = "preposes",
) -> Path:
    """
    Convenience wrapper used by the prepose sampler:
    - Crops points within 5*radius around object_center_world
    - Converts both points and EE pose to object-centered frame
    - Writes one HDF5 file under `data/<run_name>/`
    """
    cfg = DataCollectionConfig(run_name=run_name)
    points = np.asarray(combined_points, dtype=np.float32)
    center = np.asarray(object_center_world, dtype=np.float32).reshape(3)
    r = float(radius)

    distances = np.linalg.norm(points - center, axis=1)
    nearby_points = points[distances <= (5.0 * r)]
    points_obj = nearby_points - center

    ee_pose_world = np.asarray(prepose_matrix, dtype=np.float32)
    ee_pose_obj = ee_pose_world.copy()
    ee_pose_obj[:3, 3] -= center

    return prepose_write_h5(
        cfg,
        base_config=np.asarray(base_config, dtype=np.float32),
        arm_config=np.asarray(arm_config, dtype=np.float32),
        ee_pose_world=ee_pose_world,
        ee_pose_obj=ee_pose_obj,
        points_obj=points_obj,
        target_xyz_world=center,
        object_center_world=center,
        radius=r,
    )
