#!/usr/bin/env python3
"""
Real Robot Grasping Experiment Script

This script allows running the Grasp-Anywhere system on a real Fetch robot.
It uses interactive pointing (clicking on an image) to specify the target object.
Success/failure results are tracked and saved to a JSON file.
"""
import argparse
import json
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml

from grasp_anywhere.core.nav_manip_scheduler import NavManipScheduler
from grasp_anywhere.core.scheduler import Scheduler
from grasp_anywhere.core.sequential_scheduler import SequentialScheduler
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.utils.gui_utils import ClickPointCollector
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.owl_utils import resolve_target_label


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

    # Check bounds
    if u < 0 or u >= depth.shape[1] or v < 0 or v >= depth.shape[0]:
        log.warning(f"Click ({u}, {v}) out of depth bounds {depth.shape}")
        return None

    z = float(depth[v, u])
    if z <= 0 or np.isnan(z):
        log.warning(f"Invalid depth at click location ({u}, {v}): {z}")
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    p_cam_h = np.array([x, y, z, 1.0], dtype=np.float32)
    p_world_h = camera_extrinsic @ p_cam_h
    return p_world_h[:3]


def run_real_robot():
    """
    Main loop for real robot grasping experiments.
    """
    parser = argparse.ArgumentParser(description="Run Real Robot Grasping Experiment")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="grasp_anywhere/configs/real_fetch.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config_path = args.config
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir,
        f"real_robot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    # 1. Initialize Robot & Scheduler
    log.info("Initializing Real Fetch Robot...")
    fetch_robot = Fetch(config_path=config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    scheduler_type = config.get("planning", {}).get("scheduler_type", "default")
    if scheduler_type == "sequential":
        log.info("Using SequentialScheduler")
        scheduler = SequentialScheduler(robot=fetch_robot, config_path=config_path)
    elif scheduler_type == "nav_manip":
        log.info("Using NavManipScheduler (Decoupled Navigation & Manipulation)")
        scheduler = NavManipScheduler(robot=fetch_robot, config_path=config_path)
    else:
        log.info("Using Default Scheduler")
        scheduler = Scheduler(robot=fetch_robot, config_path=config_path)

    point_collector = ClickPointCollector()

    results = {
        "timestamp": datetime.now().isoformat(),
        "tasks": [],
        "summary": {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "collision_failures": 0,
            "grasping_failures": 0,
            "out_of_reachability": 0,
            "perception_failure": 0,
            "planning_failures": 0,
            "ik_failures": 0,
        },
    }

    log.info(f"Results will be saved to: {results_file}")

    while True:
        current_task_idx = results["summary"]["total_tasks"] + 1
        log.info("\n" + "=" * 60)
        log.info(f"TASK {current_task_idx}")
        log.info("=" * 60)

        # --- PRE-OBSERVATION ---
        log.info("\n[Step 1] Initial Positioning")
        location = input(
            "Enter observation location ('table', 'kitchen_counter', 'workstation', "
            "'coffee_table', 'sofa') or press Enter to skip: "
        ).strip()
        if location:
            log.info(f"Moving to location: {location}")
            success, msg = scheduler.move_planner.run(
                location, fetch_robot.scene.current_environment()
            )
            log.info(f"Move result: {msg}")
            if not success:
                log.warning("Initial move failed. Proceeding with current pose.")
            else:
                time.sleep(3.0)  # Increased settle time for head pointing

        # --- TARGET POINTING ---
        log.info("\n[Step 2] Target Selection")
        log.info("Capturing current view...")
        rgb = fetch_robot.robot_env.get_rgb()
        depth = fetch_robot.robot_env.get_depth()
        K = fetch_robot.robot_env.get_camera_intrinsics()
        T_wc = fetch_robot.robot_env.get_camera_pose()

        if rgb is None or depth is None:
            log.error("Failed to capture RGB-D data. Check robot connection.")
            continue

        log.info("A window will open. Click on the target object.")
        points, labels = point_collector.collect_points(rgb)

        if not points or len(points) == 0:
            log.info("No selection made. Skipping task.")
            continue

        # Convert click to 3D world coordinate
        selected_uv = (int(points[0][0]), int(points[0][1]))
        object_world = _backproject_click_to_world(selected_uv, depth, K, T_wc)

        if object_world is None:
            log.error("Could not backproject click. Possibly invalid depth at target.")
            continue

        log.info(f"Target selected at World Coordinates: {object_world.tolist()}")

        # Identify the object label using Owl-ViT immediately
        full_object_list = ["bottle", "apple"]
        target_label = resolve_target_label(
            scheduler.owl_client, rgb, full_object_list, selected_uv
        )
        if target_label is None:
            log.warning(
                "Could not identify specific object. Refining with all candidates."
            )
        else:
            log.info(f"Identified object as: {target_label}")

        # --- EXECUTION ---
        log.info("\n[Step 3] Execution")
        task_result = {
            "task_id": current_task_idx,
            "target_world": object_world.tolist(),
            "text_prompt": target_label,
            "success": False,
            "failure_reason": None,
            "message": "",
        }

        # Run Grasp-Anywhere (same core logic as ManiSkill benchmark)
        success, message = scheduler.grasp_anywhere(
            object_world.reshape(1, 3),
            max_attempts=5,
            observation_delay=5.0,
            text_prompt=target_label,
        )

        task_result["success"] = success
        task_result["message"] = message

        # Metrics and results summary (mimic ManiSkill benchmark logic)
        if success:
            results["summary"]["successful_tasks"] += 1
        else:
            results["summary"]["failed_tasks"] += 1
            if "IK_FAILED" in message:
                task_result["failure_reason"] = "ik_failure"
                results["summary"]["ik_failures"] += 1
            elif "OUT_OF_REACHABILITY" in message:
                task_result["failure_reason"] = "out_of_reachability"
                results["summary"]["out_of_reachability"] += 1
            elif "PERCEPTION_FAILURE" in message:
                task_result["failure_reason"] = "perception_failure"
                results["summary"]["perception_failure"] += 1
            elif "GRASP_EXECUTION_FAILED" in message:
                task_result["failure_reason"] = "grasping_failure"
                results["summary"]["grasping_failures"] += 1
            else:
                task_result["failure_reason"] = "planning_failure"
                results["summary"]["planning_failures"] += 1

        results["summary"]["total_tasks"] += 1
        results["tasks"].append(task_result)

        log.info(
            f"Task {current_task_idx} result: {'SUCCESS' if success else 'FAILED'}"
        )
        log.info(f"Message: {message}")

        # Save results incrementally
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        log.info(
            f"Current Record: {results['summary']['successful_tasks']}/{results['summary']['total_tasks']} success"
        )

        # --- REPEAT ---
        cont = input("\nRun another task? (y/n): ").strip().lower()
        if cont != "y":
            break

    # Final Save and Summary
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    total = results["summary"]["total_tasks"]
    if total > 0:
        log.info("\n" + "=" * 60)
        log.info("REAL ROBOT RECAP")
        log.info(f"Total Tasks: {total}")
        log.info(
            f"Success: {results['summary']['successful_tasks']} | Failed: {results['summary']['failed_tasks']}"
        )
        log.info(f"IK Failures: {results['summary']['ik_failures']}")
        log.info(f"Perception Failures: {results['summary']['perception_failure']}")
        log.info(f"Outcome Rate: {(results['summary']['successful_tasks']/total):.1%}")
        log.info(f"Full results saved to: {os.path.abspath(results_file)}")
        log.info("=" * 60)


if __name__ == "__main__":
    run_real_robot()
