#!/usr/bin/env python3
import json
import math
import os
import sys
import termios
import time
import tty
import uuid

import numpy as np
import sapien

# Fix library path to use conda environment's libraries (mirrors run_maniskill_benchmark.py)
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}"
        os.execv(sys.executable, [sys.executable] + sys.argv)


from mani_skill.utils.building import actors

from grasp_anywhere.envs.maniskill.maniskill_env_mpc import ManiSkillEnv
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.utils.logger import log

HELP_TEXT = (
    "\nTeleop controls (Whole-Body; absolute targets):\n"
    "  Base (absolute target updated per key):\n"
    "    w/s: forward/back  |  a/d: yaw left/right  |  Space: stop queue\n"
    "  Torso: i/k up/down\n"
    "  Head: z/x pan left/right  |  c/v tilt up/down\n"
    "  Arm joints (+/-):\n"
    "    q/a: shoulder pan   |  e/d: upperarm roll\n"
    "    r/f: shoulder lift  |  t/g: elbow flex\n"
    "    y/h: forearm roll   |  u/j: wrist flex\n"
    "    o/l: wrist roll\n"
    "  Gripper: [ : open  |  ] : close\n"
    "  h: help  |  ESC/Ctrl-C: quit\n"
)


def getch() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def update_target_base_relative(
    target_base, dx_body: float, dy_body: float, dth: float
):
    """Update an absolute base target by a body-frame delta, in-place."""
    tx, ty, tth = target_base
    dx_world = dx_body * math.cos(tth) - dy_body * math.sin(tth)
    dy_world = dx_body * math.sin(tth) + dy_body * math.cos(tth)
    target_base[0] = tx + dx_world
    target_base[1] = ty + dy_world
    target_base[2] = tth + dth


def send_wb_absolute(fetch: Fetch, target_joints, target_base):
    """Send absolute whole-body motion: arm_path/base_configs use absolute waypoints."""
    # Only send the target waypoint, not the current state
    arm_path = [list(target_joints)]
    base_configs = [list(target_base)]
    fetch.start_whole_body_motion(arm_path, base_configs)


def main():
    log.info("--- ManiSkill Fetch Whole-Body Teleop ---")

    # Load benchmark scene 0
    benchmark_path = "resources/grasp_benchmark.json"
    if not os.path.exists(benchmark_path):
        log.error(f"Benchmark file not found: {benchmark_path}")
        return

    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    scene_id = "scene_0"
    if scene_id not in benchmark_data:
        log.error(f"Scene {scene_id} not found in benchmark data")
        return

    scene_data = benchmark_data[scene_id]
    seed = scene_data.get("seed", 0)
    canonical_map_path = scene_data.get("canonical_map_path")

    if canonical_map_path:
        canonical_map_path = os.path.expanduser(canonical_map_path)
        if not os.path.isabs(canonical_map_path):
            canonical_map_path = os.path.normpath(
                os.path.join(os.getcwd(), canonical_map_path)
            )

    static_pcd_paths = (
        [canonical_map_path]
        if canonical_map_path and os.path.exists(canonical_map_path)
        else []
    )

    env_id = "ReplicaCAD_SceneManipulation-v1"
    sim_env = ManiSkillEnv(env_id=env_id, robot_uids="fetch", render_mode="human")

    # Reset environment with seed
    sim_env.reset(seed=seed)

    # Place objects from scene_0
    grasp_tasks = scene_data.get("grasp_tasks", [])
    spawned_actor_names = []
    for t in grasp_tasks:
        model_id = t["model_id"]
        position = np.asarray(t["position"], dtype=np.float32).reshape(-1, 3)[0]
        orientation = np.asarray(t["orientation"], dtype=np.float32).reshape(-1, 4)[0]

        builder = actors.get_actor_builder(sim_env.env.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=position, q=orientation)
        actor_name = f"ycb_{model_id}_{uuid.uuid4().hex[:8]}"
        builder.build(name=actor_name)
        spawned_actor_names.append(actor_name)

    config_path = "grasp_anywhere/configs/maniskill_fetch.yaml"
    # Pass static_pcd_paths to Fetch
    fetch = Fetch(
        config_path=config_path, robot_env=sim_env, static_pcd_paths=static_pcd_paths
    )

    # Step sizes
    step_torso = 0.02  # meters
    step_joint = 0.05  # radians
    step_forward = 0.10  # meters in body frame
    step_yaw = 0.10  # radians
    step_head = 0.05  # radians

    print(HELP_TEXT)
    print("Press keys to control. Viewer renders; terminal captures input.")

    # Joint index map in planning vector [torso, j1..j7]
    TORSO = 0
    SHOULDER_PAN = 1
    SHOULDER_LIFT = 2
    UPPERARM_ROLL = 3
    ELBOW_FLEX = 4
    FOREARM_ROLL = 5
    WRIST_FLEX = 6
    WRIST_ROLL = 7

    # Initialize absolute targets from current state
    target_joints = fetch.get_current_planning_joints()
    if target_joints is None:
        target_joints = [0.0] * 8
    bx, by, bth = fetch.get_base_params()
    target_base = [bx, by, bth]
    # Head targets (absolute pan, tilt)
    joints_with_head = fetch.get_current_planning_joints_with_head()
    if joints_with_head is not None and len(joints_with_head) >= 10:
        target_head_pan = joints_with_head[-2]
        target_head_tilt = joints_with_head[-1]
    else:
        target_head_pan = 0.0
        target_head_tilt = 0.0

    # Start collision monitoring on a real object name (avoids StopIteration in start_monitoring)
    monitor_name = next(
        (n for n in spawned_actor_names if "006_mustard_bottle" in n), None
    )
    monitor_name = monitor_name or (
        spawned_actor_names[0] if spawned_actor_names else None
    )
    if monitor_name and hasattr(fetch.robot_env, "start_monitoring"):
        fetch.robot_env.start_monitoring(monitor_name)
        print(f"Collision monitoring enabled for: {monitor_name}")

    try:
        while True:
            c = getch()
            if c in ("\x03", "\x1b"):
                fetch.stop_whole_body_motion()
                print("Quit")
                break
            elif c == "h":
                print(HELP_TEXT)

            # Base target updates (absolute)
            elif c in ("w", "W"):
                update_target_base_relative(target_base, step_forward, 0.0, 0.0)
                send_wb_absolute(fetch, target_joints, target_base)
                print("Base target: forward")
            elif c in ("s", "S"):
                update_target_base_relative(target_base, -step_forward, 0.0, 0.0)
                send_wb_absolute(fetch, target_joints, target_base)
                print("Base target: backward")
            elif c in ("a", "A"):
                update_target_base_relative(target_base, 0.0, 0.0, step_yaw)
                send_wb_absolute(fetch, target_joints, target_base)
                print("Base target: yaw left")
            elif c in ("d", "D"):
                update_target_base_relative(target_base, 0.0, 0.0, -step_yaw)
                send_wb_absolute(fetch, target_joints, target_base)
                print("Base target: yaw right")
            elif c == " ":
                fetch.stop_whole_body_motion()
                print("WB queue: stop/clear")

            # Head (absolute)
            elif c in ("z", "Z"):
                target_head_pan += step_head
                fetch.move_head(target_head_pan, target_head_tilt, duration=0.15)
                print("Head pan target: left")
            elif c in ("x", "X"):
                target_head_pan -= step_head
                fetch.move_head(target_head_pan, target_head_tilt, duration=0.15)
                print("Head pan target: right")
            elif c in ("c", "C"):
                target_head_tilt -= step_head
                fetch.move_head(target_head_pan, target_head_tilt, duration=0.15)
                print("Head tilt target: up")
            elif c in ("v", "V"):
                target_head_tilt += step_head
                fetch.move_head(target_head_pan, target_head_tilt, duration=0.15)
                print("Head tilt target: down")

            # Torso
            elif c in ("i", "I"):
                target_joints[TORSO] += step_torso
                send_wb_absolute(fetch, target_joints, target_base)
                print("Torso target: up")
            elif c in ("k", "K"):
                target_joints[TORSO] -= step_torso
                send_wb_absolute(fetch, target_joints, target_base)
                print("Torso target: down")

            # Arm joints
            elif c in ("q", "Q"):
                target_joints[SHOULDER_PAN] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Shoulder pan target: +")
            elif c in ("a", "A"):
                target_joints[SHOULDER_PAN] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Shoulder pan target: -")
            elif c in ("r", "R"):
                target_joints[SHOULDER_LIFT] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Shoulder lift target: +")
            elif c in ("f", "F"):
                target_joints[SHOULDER_LIFT] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Shoulder lift target: -")
            elif c in ("e", "E"):
                target_joints[UPPERARM_ROLL] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Upperarm roll target: +")
            elif c in ("d", "D"):
                target_joints[UPPERARM_ROLL] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Upperarm roll target: -")
            elif c in ("t", "T"):
                target_joints[ELBOW_FLEX] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Elbow flex target: +")
            elif c in ("g", "G"):
                target_joints[ELBOW_FLEX] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Elbow flex target: -")
            elif c in ("y", "Y"):
                target_joints[FOREARM_ROLL] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Forearm roll target: +")
            elif c in ("h", "H"):
                target_joints[FOREARM_ROLL] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Forearm roll target: -")
            elif c in ("u", "U"):
                target_joints[WRIST_FLEX] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Wrist flex target: +")
            elif c in ("j", "J"):
                target_joints[WRIST_FLEX] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Wrist flex target: -")
            elif c in ("o", "O"):
                target_joints[WRIST_ROLL] += step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Wrist roll target: +")
            elif c in ("l", "L"):
                target_joints[WRIST_ROLL] -= step_joint
                send_wb_absolute(fetch, target_joints, target_base)
                print("Wrist roll target: -")

            # Gripper (not in whole-body path; use dedicated control)
            elif c == "[":
                fetch.control_gripper(position=1.0)
                print("Gripper: open")
            elif c == "]":
                fetch.control_gripper(position=0.0)
                print("Gripper: close")

            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        fetch.stop_whole_body_motion()
    finally:
        try:
            sim_env.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
