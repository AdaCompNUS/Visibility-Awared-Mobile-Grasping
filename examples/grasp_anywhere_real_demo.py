#!/usr/bin/env python3
import time
from typing import List, Tuple

import numpy as np

from grasp_anywhere.core.scheduler import Scheduler
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.stage_planners.move_stage import MovePlanner
from grasp_anywhere.utils.gui_utils import ClickPointCollector
from grasp_anywhere.utils.logger import log


def _backproject_click_to_world(
    click_uv: Tuple[int, int],
    depth: np.ndarray,
    K: np.ndarray,
    camera_extrinsic: np.ndarray,
) -> np.ndarray:
    """
    Convert a single image-space click into a 3D world coordinate using depth and intrinsics.
    Returns a 3D point (shape (3,)) in world coordinates.
    """
    u, v = int(click_uv[0]), int(click_uv[1])
    z = float(depth[v, u])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    p_cam_h = np.array([x, y, z, 1.0], dtype=np.float32)
    p_world_h = camera_extrinsic @ p_cam_h
    return p_world_h[:3]


def _observe_from_locations(
    robot: Fetch, move_planner: MovePlanner, locations: List[str]
):
    """
    Move through locations, capture sensor data, return a list of observations.
    Each observation is a dict with keys: name, rgb, depth, K, T_wc.
    """
    observations = []
    for loc in locations:
        log.info(f"Moving to observation location: {loc}")
        collision_points = robot.scene.current_environment()
        collision_points = (
            collision_points
            if isinstance(collision_points, np.ndarray)
            else np.empty((0, 3), dtype=np.float32)
        )
        ok, msg = move_planner.run(
            loc, collision_points, enable_replanning=True, enable_pcd_alignment=True
        )
        log.info(msg)

        # Wait for a few seconds
        time.sleep(2.0)

        rgb = robot.get_rgb()
        depth = robot.get_depth()
        K = robot.get_camera_intrinsic()
        T_wc = robot.get_camera_pose()
        observations.append(
            {"name": loc, "rgb": rgb, "depth": depth, "K": K, "T_wc": T_wc}
        )

    return observations


def main():
    log.info("=== Grasp-Anywhere Real Robot Demo ===")

    # 1) Initialize robot and scheduler (real robot by default)
    fetch_robot = Fetch(config_path="grasp_anywhere/configs/real_fetch.yaml")
    scheduler = Scheduler(
        robot=fetch_robot, config_path="grasp_anywhere/configs/real_fetch.yaml"
    )

    # 2) Predefined observation locations (use all known MovePlanner locations)
    move_planner = MovePlanner(fetch_robot)
    observation_locations = list(move_planner.locations.keys())
    log.info(f"Observation locations: {observation_locations}")

    # 3) Collect observations
    observations = _observe_from_locations(
        fetch_robot, move_planner, observation_locations
    )

    # 4) Let user choose a view to click. If no point is selected, go back to selection.
    while True:
        print("\nAvailable observation views:")
        for idx, obs in enumerate(observations):
            print(f"  [{idx}] {obs['name']}")

        user_in = input(
            "Enter the index of the view to select the object from: "
        ).strip()
        chosen_idx = int(user_in)

        chosen = observations[chosen_idx]
        rgb = np.array(chosen["rgb"], dtype=np.uint8)
        depth = np.array(chosen["depth"], dtype=float)
        K = np.array(chosen["K"], dtype=float)
        T_wc = np.array(chosen["T_wc"], dtype=float)

        # 5) Collect a single positive click from this view
        print("\nA window will open. Left-click to add a positive point.")
        print("Press 's' to finish, 'r' to reset, 'q' to quit.")
        point_collector = ClickPointCollector()
        points, labels = point_collector.collect_points(rgb)
        if points is None:
            print("User quit point selection. Exiting.")
            return
        if len(points) == 0:
            print("No points selected. Returning to observation selection.")
            continue

        selected_uv = (int(points[0][0]), int(points[0][1]))
        break

    # 6) Backproject to world coordinate
    object_world = _backproject_click_to_world(selected_uv, depth, K, T_wc)

    log.info(f"Selected world coordinate: {object_world.tolist()}")

    # 7) Run grasp-anywhere pipeline with this coordinate
    success, message = scheduler.grasp_anywhere(object_world.reshape(1, 3))
    print(f"Grasp result: success={success}, message={message}")


if __name__ == "__main__":
    main()
