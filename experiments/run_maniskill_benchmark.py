#!/usr/bin/env python3
"""
ManiSkill Grasping Benchmark

Success Criteria:
  1. Gripper must be within 0.08m of target object
  2. No non-gripper robot collisions with environment

Evaluation: Single attempt per object, real-time monitoring during execution
Collision Detection: Integrated into environment step loop, stops motion on collision or success
"""
import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import uuid
from datetime import datetime

import yaml

# Fix library path to use conda environment's libraries
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}"
        # Re-exec the script with updated LD_LIBRARY_PATH
        os.execv(sys.executable, [sys.executable] + sys.argv)


def check_gripper_touching_object(
    sim_env, target_object_name: str, distance_threshold: float = 0.08
) -> bool:
    """
    Check if gripper is close enough to target object.
    Uses Euclidean distance between gripper links and target center.
    """
    import numpy as np
    import sapien

    robot = sim_env.agent.robot

    # Get all gripper links
    gripper_links: list[sapien.Link] = []
    for link in robot.links:
        link_name_lower: str = link.name.lower()
        if "gripper" in link_name_lower or "finger" in link_name_lower:
            gripper_links.append(link)

    # Find target object
    target_actor: sapien.Entity
    for actor in sim_env.env.unwrapped.scene.get_all_actors():
        if target_object_name in actor.name:
            target_actor = actor
            break

    # Get target position
    target_pos: np.ndarray = np.asarray(target_actor.pose.p, dtype=np.float32)

    # Check distance from each gripper link to target
    for gripper_link in gripper_links:
        gripper_pos: np.ndarray = np.asarray(gripper_link.pose.p, dtype=np.float32)

        distance: float = float(np.linalg.norm(gripper_pos - target_pos))

        if distance <= distance_threshold:
            return True

    return False


def init_worker(gpu_queue):
    import os

    # Get assigned GPU ID from queue
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def process_single_scene(args):
    (
        scene_id,
        scene_data,
        config_path,
        force_no_render,
        save_trajectory,
        trajectory_dir,
    ) = args

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if force_no_render:
        render_mode = None
    else:
        render_mode = config.get("debug", {}).get("render_mode", "human")
        if render_mode == "None":
            render_mode = None

    import numpy as np
    import sapien

    from grasp_anywhere.core.closed_loop_scheduler import ClosedLoopScheduler
    from grasp_anywhere.core.nav_manip_scheduler import NavManipScheduler
    from grasp_anywhere.core.nav_prepose_scheduler import NavPreposeScheduler
    from grasp_anywhere.core.scheduler import Scheduler
    from grasp_anywhere.core.sequential_scheduler import SequentialScheduler
    from grasp_anywhere.envs.maniskill.maniskill_env_mpc import ManiSkillEnv
    from grasp_anywhere.robot.fetch import Fetch
    from grasp_anywhere.utils.logger import log

    # Initialize ManiSkill environment
    sim_env: ManiSkillEnv = ManiSkillEnv(
        env_id="ReplicaCAD_SceneManipulation-v1",
        robot_uids="fetch",
        render_mode=render_mode,
    )
    from mani_skill.utils.building import actors

    # Initialize local results variable to mimic global structure for compatibility with existing code
    results = {
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
        }
    }

    log.info(f"Processing scene: {scene_id}")

    scene_results = {"tasks": [], "scene_success_rate": 0.0}
    # results["scenes"][scene_id] = scene_results # Cannot set global dict in worker

    # Get scene seed
    seed = scene_data.get("seed")
    if seed is None:
        m = re.search(r"(\d+)$", scene_id)
        seed = int(m.group(1)) if m else 0

    # Get canonical map path
    canonical_map_path = scene_data.get("canonical_map_path")
    if not canonical_map_path:
        sim_env.close()
        return scene_id, scene_results, results["summary"]

    canonical_map_path = os.path.expanduser(canonical_map_path)
    if not os.path.isabs(canonical_map_path):
        canonical_map_path = os.path.normpath(
            os.path.join(os.getcwd(), canonical_map_path)
        )

    if not os.path.exists(canonical_map_path):
        sim_env.close()
        return scene_id, scene_results, results["summary"]

    static_pcd_paths = [canonical_map_path]
    grasp_tasks = scene_data.get("grasp_tasks", [])

    # For each grasp task, run with trials and evaluation
    for task_idx, task in enumerate(grasp_tasks):
        # if task_idx != 18:
        #     continue
        log.info(f"Task {task_idx+1}/{len(grasp_tasks)}: {task['model_id']}")

        task_result = {
            "task_id": task_idx,
            "model_id": task["model_id"],
            "position": task["position"],
            "collision_detected": False,
            "gripper_touched_object": False,
            "success": False,
            "failure_reason": None,
        }

        # Reset environment for this task
        sim_env.reset(seed=seed)

        # Place ALL YCB objects in the scene
        object_actors = {}
        for i, t in enumerate(grasp_tasks):
            model_id = t["model_id"]
            position = np.asarray(t["position"], dtype=np.float32).reshape(-1, 3)[0]
            orientation = np.asarray(t["orientation"], dtype=np.float32).reshape(-1, 4)[
                0
            ]

            builder = actors.get_actor_builder(sim_env.env.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=position, q=orientation)
            actor_name = f"ycb_{model_id}_{uuid.uuid4().hex[:8]}"
            builder.build(name=actor_name)

            if i == task_idx:
                object_actors[model_id] = actor_name

        # Initialize robot and scheduler
        # Apply benchmark configuration to the environment's manager
        benchmark_cfg = config.get("benchmark", {})

        sim_env.benchmark_manager.enabled = bool(
            benchmark_cfg.get("enable_dynamic_challenges", False)
        )
        sim_env.benchmark_manager.nav_trigger_distance = float(
            benchmark_cfg.get("nav_trigger_distance", 1.7)
        )
        # Pass dynamic obstacle config if available for the current task
        dynamic_obstacle_config = task.get("dynamic_obstacle")
        if dynamic_obstacle_config:
            sim_env.benchmark_manager.set_current_obstacle_config(
                dynamic_obstacle_config
            )
        else:
            sim_env.benchmark_manager.set_current_obstacle_config(None)

        # Note: nav_trigger_velocity was removed from manager by user so we don't set it here.

        fetch_robot = Fetch(
            config_path=config_path,
            robot_env=sim_env,
            static_pcd_paths=static_pcd_paths,
        )
        scheduler_type = config.get("planning", {}).get("scheduler_type", "default")
        if scheduler_type == "sequential":
            log.info("Using SequentialScheduler")
            scheduler = SequentialScheduler(robot=fetch_robot, config_path=config_path)
        elif scheduler_type == "nav_manip":
            log.info("Using NavManipScheduler (Decoupled Navigation & Manipulation)")
            scheduler = NavManipScheduler(robot=fetch_robot, config_path=config_path)
        elif scheduler_type == "nav_prepose":
            log.info("Using NavPreposeScheduler (Nav+Manip with PreposeSampler)")
            scheduler = NavPreposeScheduler(robot=fetch_robot, config_path=config_path)
        elif scheduler_type == "closed_loop":
            log.info("Using ClosedLoopScheduler (Minimal Prepose->Grasp Loop)")
            scheduler = ClosedLoopScheduler(robot=fetch_robot, config_path=config_path)
        else:
            log.info("Using Default Scheduler")
            scheduler = Scheduler(robot=fetch_robot, config_path=config_path)

        # Use the benchmark-provided world position as the object point cloud (1x3)
        object_pcd: np.ndarray = np.asarray(task["position"], dtype=np.float32).reshape(
            -1, 3
        )

        # Get target object name and start monitoring
        target_actor_name: str = object_actors[task["model_id"]]
        monitor_cfg = config.get("monitor", {})
        sim_env.start_monitoring(
            target_actor_name,
            hold_seconds=float(monitor_cfg.get("hold_seconds", 0.0)),
            contact_force_threshold=float(
                monitor_cfg.get("contact_force_threshold", 0.001)
            ),
        )

        # Start trajectory recording if enabled
        if save_trajectory:
            sim_env.start_trajectory_recording()

        # Execute grasp attempt (single trial)
        success: bool
        message: str
        success, message = scheduler.grasp_anywhere(
            object_pcd, max_attempts=5, target_model_id=target_actor_name
        )

        # Arm the hold timer AFTER the grasp stage (which includes lifting back to pre-grasp).
        sim_env.arm_hold_monitoring()

        # Wait at the lifted pose before stopping monitoring so lift-and-hold can be evaluated.
        # Keep this intentionally simple: sleep in wall-time while the env background loop keeps stepping.
        time.sleep(
            float(
                monitor_cfg.get(
                    "post_grasp_wait_s",
                    float(monitor_cfg.get("hold_seconds", 0.0)) + 0.2,
                )
            )
        )

        # Stop monitoring and get results
        sim_env.stop_monitoring()
        has_collision: bool
        has_success: bool
        has_hold_success: bool
        (
            has_collision,
            has_success,
            has_hold_success,
            collision_pairs,
        ) = sim_env.get_hold_monitoring_results()

        task_result["collision_detected"] = has_collision
        task_result["collision_pairs"] = collision_pairs
        task_result["gripper_touched_object"] = bool(has_success)
        task_result["success"] = bool(has_hold_success) and not has_collision
        task_result["hold_success"] = bool(has_hold_success)

        # Save trajectory if enabled
        if save_trajectory:
            trajectory = sim_env.stop_trajectory_recording()
            # Save trajectory to .npy file
            if trajectory is not None and trajectory_dir is not None:
                traj_filename = f"{scene_id}_task{task_idx:03d}_{task['model_id']}.npy"
                traj_path = os.path.join(trajectory_dir, traj_filename)
                np.save(traj_path, trajectory)
                task_result["trajectory_file"] = traj_filename

        # Determine final task outcome
        if task_result["success"]:
            results["summary"]["successful_tasks"] += 1
        else:
            results["summary"]["failed_tasks"] += 1
            if has_collision:
                task_result["failure_reason"] = "collision"
                results["summary"]["collision_failures"] += 1
            elif success:  # Grasp stage returns success but hold stage fails
                task_result["failure_reason"] = "grasping_failure"
                results["summary"]["grasping_failures"] += 1
            elif "IK_FAILED" in message:
                task_result["failure_reason"] = "ik_failure"
                results["summary"]["ik_failures"] += 1
            elif "OUT_OF_REACHABILITY" in message:
                task_result["failure_reason"] = "out_of_reachability"
                results["summary"]["out_of_reachability"] += 1
            elif "PERCEPTION_FAILURE" in message:
                task_result["failure_reason"] = "perception_failure"
                results["summary"]["perception_failure"] += 1
            else:
                task_result["failure_reason"] = "planning_failure"
                results["summary"]["planning_failures"] += 1

        results["summary"]["total_tasks"] += 1
        scene_results["tasks"].append(task_result)

        # Calculate scene success rate incrementally
        if scene_results["tasks"]:
            scene_success_count = sum(1 for t in scene_results["tasks"] if t["success"])
            scene_results["scene_success_rate"] = scene_success_count / len(
                scene_results["tasks"]
            )

        log.info(
            f"Task {task_idx+1} result: {'SUCCESS' if task_result['success'] else 'FAILED'} | "
            f"Collision Free: {not has_collision} | Hold success: {task_result['hold_success']}"
        )

        # Save results incrementally (Disabled in worker)
        # with open(results_file, "w") as f:
        #     json.dump(results, f, indent=2)

    sim_env.close()
    return scene_id, scene_results, results["summary"]


def run_benchmark() -> None:
    """
    Run ManiSkill benchmark with real-time collision and success monitoring.

    Logic:
    1. For each object: Reset environment, place all objects
    2. Start monitoring, execute grasp (single attempt), stop monitoring
    3. Monitoring runs in environment step loop, stops motion on collision or success
    4. SUCCESS if gripper touched AND no collision, otherwise FAIL
    5. Move to next object

    Success: Gripper contacts target OR distance <= 0.08m, AND no non-gripper collisions
    Collision: Non-gripper robot link contacts environment (gripper contacts ignored)
    """
    from grasp_anywhere.utils.logger import log

    # log.info("Starting ManiSkill Benchmark")
    # Load benchmark data
    benchmark_path: str = "resources/grasp_benchmark.json"
    with open(benchmark_path, "r") as f:
        benchmark_data: dict[str, dict | list | str | int | float | None] = json.load(f)

    # Initialize results tracking
    results: dict[str, str | dict | int] = {
        "timestamp": datetime.now().isoformat(),
        "config": {},
        "scenes": {},
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

    # Create parser
    parser = argparse.ArgumentParser(description="Run ManiSkill Benchmark")
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run benchmark in parallel using multiprocessing (offline mode)",
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=10,
        help="Number of parallel processes to use (default: 10)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="grasp_anywhere/configs/maniskill_fetch.yaml",
        help="Path to configuration file (default: grasp_anywhere/configs/maniskill_fetch.yaml)",
    )
    parser.add_argument(
        "-t",
        "--save-trajectory",
        action="store_true",
        help="Save executed trajectories for each task",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        default="2,3",
        help="Comma-separated list of GPU IDs to use (default: 2,3)",
    )
    args = parser.parse_args()

    config_path: str = args.config

    # Load config to derive method name for results folder
    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)

    # Derive method name from config
    scheduler_type = run_config.get("planning", {}).get("scheduler_type", "default")
    enable_dynamic = run_config.get("benchmark", {}).get(
        "enable_dynamic_challenges", False
    )
    velocity_weight = run_config.get("gaze", {}).get("velocity_weight", 1.0)

    # Map scheduler type to method prefix
    if scheduler_type == "default":
        # Check if velocity awareness is disabled
        if velocity_weight == 0.0:
            method_prefix = "novel"
        else:
            method_prefix = "ours"
    elif scheduler_type == "sequential":
        method_prefix = "seq"
    elif scheduler_type == "nav_manip":
        method_prefix = "nam"
    elif scheduler_type == "nav_prepose":
        method_prefix = "nap"  # Nav+Prepose
    elif scheduler_type == "closed_loop":
        method_prefix = "clp"  # Closed-Loop Planning
    else:
        method_prefix = scheduler_type

    method_suffix = "dyn" if enable_dynamic else "static"
    method_name = f"{method_prefix}_{method_suffix}"

    # Prepare results file with method-based folder structure
    # Create a unique run folder for this benchmark execution
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    results_dir = os.path.join("results", method_name)
    run_dir = os.path.join(results_dir, f"run_{run_timestamp}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Create trajectory directory if saving trajectories
    trajectory_dir = None
    if args.save_trajectory:
        trajectory_dir = os.path.join(run_dir, "trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)

    results_file = os.path.join(run_dir, "benchmark_results.json")
    results["config"] = {
        "config_path": config_path,
        "scheduler_type": scheduler_type,
        "method_name": method_name,
        "nav_trigger_distance": run_config.get("benchmark", {}).get(
            "nav_trigger_distance", 1.7
        ),
    }
    log.info(f"Method: {method_name} | Results will be saved to: {run_dir}")

    args_list = [
        (
            scene_id,
            scene_data,
            config_path,
            args.parallel,
            args.save_trajectory,
            trajectory_dir,
        )
        for scene_id, scene_data in benchmark_data.items()
    ]

    if args.parallel:
        # Use multiprocessing to process scenes in parallel
        num_processes = args.num_processes

        # Distribute processes across specified GPUs
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        gpu_queue = multiprocessing.Queue()
        for i in range(num_processes):
            gpu_queue.put(gpus[i % len(gpus)])

        log.info(
            f"Running benchmark with {num_processes} parallel processes distributed on GPUs {gpus}..."
        )

        with multiprocessing.Pool(
            processes=num_processes, initializer=init_worker, initargs=(gpu_queue,)
        ) as pool:
            for scene_id, s_results, s_summary in pool.imap(
                process_single_scene, args_list
            ):
                results["scenes"][scene_id] = s_results

                # Update summary statistics
                for k, v in s_summary.items():
                    if k in results["summary"]:
                        results["summary"][k] += v

                # Save results incrementally
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
    else:
        # Run sequentially (allows visualization)
        log.info("Running benchmark sequentially...")
        for scene_args in args_list:
            # if scene_args[0] != "scene_3":
            #     print(f"Skipping scene {scene_args[0]}")
            #     continue
            scene_id, s_results, s_summary = process_single_scene(scene_args)
            results["scenes"][scene_id] = s_results

            # Update summary statistics
            for k, v in s_summary.items():
                if k in results["summary"]:
                    results["summary"][k] += v

            # Save results incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Calculate and print summary
    overall_success_rate = (
        (results["summary"]["successful_tasks"] / results["summary"]["total_tasks"])
        if results["summary"]["total_tasks"] > 0
        else 0.0
    )

    log.info("\n" + "=" * 60)
    log.info(
        f"Total: {results['summary']['total_tasks']} | "
        f"Success: {results['summary']['successful_tasks']} | Failed: {results['summary']['failed_tasks']}"
    )
    log.info(
        f"Collision failures: {results['summary']['collision_failures']} | "
        f"Grasping failures: {results['summary']['grasping_failures']} | "
        f"Out of reachability: {results['summary']['out_of_reachability']} | "
        f"Planning failures: {results['summary']['planning_failures']} | "
        f"IK failures: {results['summary']['ik_failures']}"
    )
    log.info(f"Success rate: {overall_success_rate:.1%}")
    log.info("=" * 60)

    # Final save
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Final results saved to: {results_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_benchmark()
