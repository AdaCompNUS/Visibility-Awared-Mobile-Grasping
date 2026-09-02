#!/usr/bin/env python3
"""Evaluate the released MS-HAB SAC Pick policy on OUR grasp benchmark.

Runs in the MAIN pixi environment (mani_skill 3.0.1) — the same simulator and
referee as every other number in results/. The policy is the verified
plain-torch port (adapter/) of mshab_checkpoints/rl/tidy_house/pick/all.

Protocol (Phase A, static, "oracle navigation"):
  - Same scenes/tasks as experiments/run_maniskill_benchmark.py
    (resources/grasp_benchmark.json: 20 scenes x 20 tasks).
  - Robot base is SPAWNED near the target — sampled facing it within
    [0.6, 1.6] m, collision-checked — replicating MS-HAB's Pick initial-state
    assumption (their training teleports navigation; spawn <=2 m facing target).
    This favors the baseline and must be reported as oracle-navigation.
  - Policy stepped at the env's control rate for --episode-steps (default 200,
    their episode budget), head action dims zeroed (stationary_head, as
    trained). If the gripper is in contact with the target near budget end, up
    to --hold-extra-steps more steps are allowed so the hold criterion can be
    evaluated (mirrors the main benchmark's post-grasp wait).
  - OUR success rule: >=2 s stable gripper-target hold (monitor_core
    update_hold_state on sim time) AND zero non-gripper collision above 0.001 N
    at any point (eval_monitor_contacts, same parameters as the benchmark).

Usage:
  pixi run python baseline/RL/eval_rl_on_benchmark.py --gpu 2 \
      [--scenes scene_0,scene_1] [--tasks 0,1,2] [--out baseline/RL/results]
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "adapter"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(REPO / "resources/grasp_benchmark.json"))
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "mshab_checkpoints/rl/tidy_house/pick/all/policy.pt"))
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--scenes", default=None, help="comma list, default all")
    ap.add_argument("--tasks", default=None, help="comma list of task idx, default all")
    ap.add_argument("--protocol", choices=["easy", "hard"], default="easy",
                    help="easy: oracle spawn 0.6-1.6m facing target (MS-HAB Pick "
                         "assumption). hard: the benchmark's exact robot start "
                         "(rest qpos, base at (-1,0), yaw 0 — same as our pipeline).")
    ap.add_argument("--episode-steps", type=int, default=None,
                    help="default: 200 (easy, MS-HAB Pick budget) / 1000 (hard, "
                         "MS-HAB Navigate horizon)")
    ap.add_argument("--hold-extra-steps", type=int, default=80)
    ap.add_argument("--spawn-min", type=float, default=0.6)
    ap.add_argument("--spawn-max", type=float, default=1.6)
    ap.add_argument("--spawn-attempts", type=int, default=60)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--force-threshold", type=float, default=0.001)
    ap.add_argument("--seed-offset", type=int, default=0, help="added to spawn rng seed")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--dynamic", action="store_true",
                    help="enable the benchmark's moving-pedestrian challenge "
                         "(DynamicBenchmarkManager, same parameters as the main "
                         "benchmark). Scored by the same physical rule; the "
                         "require_dynamic_path_change filter is NOT applied — "
                         "the RL policy has no replanning, so that metric is "
                         "undefined for it (reported via obstacle_spawned).")
    ap.add_argument("--record-video", action="store_true",
                    help="save a third-person mp4 per task (slower; use with --scenes/--tasks)")
    args = ap.parse_args()
    if args.episode_steps is None:
        args.episode_steps = 200 if args.protocol == "easy" else 1000

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("SAPIEN_NO_DISPLAY", "1")

    import gymnasium as gym
    import numpy as np
    import torch
    import sapien

    import mani_skill.envs  # noqa: F401  (registers env ids)
    from mani_skill.utils.building import actors

    from grasp_anywhere.benchmark.dynamic_benchmark_manager import (
        DynamicBenchmarkManager,
    )
    from grasp_anywhere.utils.monitor_core import (
        eval_monitor_contacts,
        update_hold_state,
    )
    from obs_builder import MshabObsBuilder
    from sac_policy import HEAD_ACTION_DIMS, load_actor

    device = "cuda"
    actor_net = load_actor(args.ckpt, device=device)

    with open(args.benchmark) as f:
        bench = json.load(f)
    scene_ids = args.scenes.split(",") if args.scenes else list(bench.keys())
    task_filter = (
        set(int(t) for t in args.tasks.split(",")) if args.tasks else None
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = f"{args.protocol}_{'dyn' if args.dynamic else 'static'}"
    # pid suffix: parallel scene-partition workers must not clobber each other
    run_name = time.strftime(f"rl_all_obj_{cond}_%Y%m%d_%H%M%S") + f"_p{os.getpid()}"

    class _EnvShim:
        """Adapter so DynamicBenchmarkManager can read scene + route.

        The manager expects the main pipeline's ManiSkillEnv (`.env` gym handle,
        `_merged_traj` planned route, `_last_waypoint_idx`). The RL policy has
        no planner, so the rollout loop feeds a synthetic straight-line
        robot->target route each step — used only for pedestrian placement
        geometry, exactly as the planned route is in the main benchmark.
        """

        def __init__(self, gym_env):
            self.env = gym_env
            self._merged_traj = []
            self._last_waypoint_idx = 0
    results = {"config": vars(args), "scenes": {}, "summary": {}}

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        robot_uids="fetch",
        obs_mode="rgb+depth+state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array" if args.record_video else None,
        sensor_configs={"width": 128, "height": 128},
    )
    uenv = env.unwrapped
    control_dt = 1.0 / uenv.control_freq
    print(f"[eval] control_freq={uenv.control_freq} sim_freq={uenv.sim_freq}")

    def _aim_render_camera(robot_xy, target_p):
        """Place the render camera above the robot--target midpoint, looking
        down at the target (~60 deg). The default ReplicaCAD render camera is a
        fixed room camera and is frequently occluded by walls; the midpoint of
        two in-room points with a steep look-down stays unoccluded."""
        if not args.record_video:
            return
        from mani_skill.utils import sapien_utils
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        mid = (robot_xy + target_p[:2]) / 2.0
        span = float(np.linalg.norm(target_p[:2] - robot_xy))
        # 3/4 side view above the midpoint; occluded regions (walls/doorway
        # voids) are cropped in post by the video writer below.
        d = target_p[:2] - robot_xy
        d = d / (np.linalg.norm(d) + 1e-6)
        perp = np.array([-d[1], d[0]])
        eye = np.array([
            mid[0] + 0.9 * perp[0] - 0.6 * d[0],
            mid[1] + 0.9 * perp[1] - 0.6 * d[1],
            2.1 + 0.25 * span,
        ])
        look = np.array([target_p[0], target_p[1], float(target_p[2])])
        pose = sapien_utils.look_at(eye, look)
        for cam in uenv.scene.human_render_cameras.values():
            cam.camera.set_local_pose(pose.sp if hasattr(pose, "sp") else pose)

    n_success = n_collision = n_no_hold = n_no_contact = n_spawn_fail = 0
    total = 0

    for scene_id in scene_ids:
        scene = bench[scene_id]
        seed = scene.get("seed", 0)
        rng = np.random.default_rng(10_000 + seed + args.seed_offset)
        scene_res = []

        for task_idx, task in enumerate(scene["grasp_tasks"]):
            if task_filter is not None and task_idx not in task_filter:
                continue
            t0 = time.time()
            env.reset(seed=seed, options=dict(reconfigure=True))

            # ---- build ALL task objects (mirrors run_maniskill_benchmark) ----
            target_name = None
            for i, t in enumerate(scene["grasp_tasks"]):
                pos = np.asarray(t["position"], dtype=np.float32).reshape(3)
                orn = np.asarray(t["orientation"], dtype=np.float32).reshape(4)
                builder = actors.get_actor_builder(uenv.scene, id=f"ycb:{t['model_id']}")
                builder.initial_pose = sapien.Pose(p=pos, q=orn)
                name = f"ycb_{t['model_id']}_{uuid.uuid4().hex[:8]}"
                built = builder.build(name=name)
                if i == task_idx:
                    target_name = name
                    target_actor = built
            target_xy = np.asarray(task["position"][:2], dtype=np.float64)

            # ---- dynamic challenge (same manager as the main benchmark) ----
            dyn_manager = None
            dyn_shim = None
            if args.dynamic and task.get("dynamic_obstacle"):
                dyn_shim = _EnvShim(env)
                dyn_manager = DynamicBenchmarkManager(dyn_shim)
                dyn_manager.enabled = True
                dyn_manager.set_current_obstacle_config(task["dynamic_obstacle"])
                dyn_manager.set_navigation_goal(
                    np.asarray(task["position"][:2], dtype=np.float32)
                )

            # ---- oracle-nav spawn: face target, [spawn_min, spawn_max] m ----
            rest_qpos = np.asarray(uenv.agent.keyframes["rest"].qpos, dtype=np.float32).copy()
            spawned = False
            if args.protocol == "hard":
                # Benchmark start: env reset already placed the robot at rest
                # qpos with base pose (-1, 0, 0.02), yaw 0 (ReplicaCAD scene
                # builder) — identical to our pipeline's start. Do not move it.
                q = rest_qpos  # base joint entries are 0; bookkeeping below
                spawned = True
            else:
              for _ in range(args.spawn_attempts):
                  r = rng.uniform(args.spawn_min, args.spawn_max)
                  ang = rng.uniform(-np.pi, np.pi)
                  bx, by = target_xy[0] + r * np.cos(ang), target_xy[1] + r * np.sin(ang)
                  yaw = float(np.arctan2(target_xy[1] - by, target_xy[0] - bx))
                  yaw += float(rng.uniform(-0.15, 0.15))
                  q = rest_qpos.copy()
                  q[0], q[1], q[2] = bx, by, yaw
                  uenv.agent.reset(q)
                  # settle & collision-check the spawn
                  zero = torch.zeros(env.action_space.shape, device=device)[None] \
                      if env.action_space.shape else None
                  ok = True
                  for _ in range(3):
                      env.step(torch.zeros((1, 13), device=device))
                      succ, coll = eval_monitor_contacts(
                          contacts=uenv.scene.get_contacts(),
                          target_object_name=target_name,
                          dt=uenv.scene.px.timestep,
                          force_threshold=args.force_threshold,
                          robot_name_substr="fetch",
                          base_link_exclude="base_link",
                      )
                      if coll:
                          ok = False
                          break
                  if ok:
                      spawned = True
                      break
                  uenv.agent.reset(rest_qpos)  # re-settle far away before retry
            record = {
                "task_id": task_idx,
                "model_id": task["model_id"],
                "spawned": spawned,
                "success": False,
                "collision": False,
                "hold_success": False,
                "touched": False,
                "steps": 0,
            }
            total += 1
            if not spawned:
                n_spawn_fail += 1
                record["failure_reason"] = "spawn_failure"
                scene_res.append(record)
                print(f"[{scene_id}/{task_idx}] SPAWN FAILURE ({task['model_id']})")
                continue

            base_xy = (
                np.asarray([-1.0, 0.0]) if args.protocol == "hard"
                else np.asarray([q[0], q[1]])
            )
            spawn_dist = float(np.linalg.norm(base_xy - target_xy))

            # ---- policy rollout under our referee ----
            _aim_render_camera(
                base_xy if args.protocol == "hard" else np.asarray([q[0], q[1]]),
                np.asarray(task["position"], dtype=np.float64),
            )
            obs_builder = MshabObsBuilder(env, target_actor)
            raw_obs = uenv.get_obs()
            collision = False
            touched = False
            hold_state = {"active": False, "t0": 0.0}
            hold_ok = False
            collision_pairs = set()
            max_steps = args.episode_steps
            step = 0
            min_tcp_dist = float("inf")
            frames = []
            obj_z0 = float(target_actor.pose.p.reshape(-1)[2].item())
            obj_z_max = obj_z0
            while step < max_steps:
                o = obs_builder.build(raw_obs)
                with torch.no_grad():
                    act = actor_net(
                        {k: v.to(device) for k, v in o["pixels"].items()},
                        o["state"].to(device),
                    )
                act = act.clone()
                for d in HEAD_ACTION_DIMS:  # stationary_head, as trained
                    act[:, d] = 0.0
                raw_obs, _, _, _, _ = env.step(act)
                step += 1
                tcp_p = obs_builder.tcp.pose.p.reshape(-1)[:3]
                tgt_p = target_actor.pose.p.reshape(-1)[:3]
                d = float((tcp_p - tgt_p).norm().item())
                min_tcp_dist = min(min_tcp_dist, d)
                obj_z_max = max(obj_z_max, float(tgt_p[2].item()))
                if dyn_manager is not None:
                    bp = self_base = obs_builder.base_link.pose
                    bx_, by_ = (
                        float(bp.p.reshape(-1)[0].item()),
                        float(bp.p.reshape(-1)[1].item()),
                    )
                    qw, qx, qy, qz = [float(v.item()) for v in bp.q.reshape(-1)[:4]]
                    byaw = float(np.arctan2(2 * (qw * qz + qx * qy),
                                            1 - 2 * (qy * qy + qz * qz)))
                    # synthetic straight-line route robot -> target (placement geometry)
                    seg = np.linalg.norm(target_xy - np.array([bx_, by_]))
                    npts = max(2, int(seg / 0.1))
                    route = np.linspace([bx_, by_], target_xy, npts)
                    dyn_shim._merged_traj = route
                    dyn_shim._last_waypoint_idx = 0
                    dyn_manager.update(
                        control_dt,
                        np.array([bx_, by_, byaw], dtype=np.float32),
                        np.array([float(act[0, -2].item()), float(act[0, -1].item())]),
                    )
                if args.record_video:
                    fr = env.render()
                    if hasattr(fr, "cpu"):
                        fr = fr.cpu().numpy()
                    frames.append(np.asarray(fr).reshape(fr.shape[-3:]))

                succ_ev, coll_ev = eval_monitor_contacts(
                    contacts=uenv.scene.get_contacts(),
                    target_object_name=target_name,
                    dt=uenv.scene.px.timestep,
                    force_threshold=args.force_threshold,
                    robot_name_substr="fetch",
                    base_link_exclude="base_link",
                )
                for pair in coll_ev:
                    collision = True
                    collision_pairs.add(pair)
                has_contact = len(succ_ev) > 0
                touched = touched or has_contact
                hold_state, ok_now = update_hold_state(
                    state=hold_state,
                    now_s=step * control_dt,
                    has_gripper_target_contact=has_contact,
                    hold_seconds=args.hold_seconds,
                )
                hold_ok = hold_ok or ok_now
                # grant extra time to finish an in-progress hold
                if (
                    step == args.episode_steps
                    and has_contact
                    and not hold_ok
                    and max_steps == args.episode_steps
                ):
                    max_steps = args.episode_steps + args.hold_extra_steps
                if hold_ok and collision:
                    break

            record.update(
                success=bool(hold_ok and not collision),
                collision=bool(collision),
                collision_pairs=sorted(list(collision_pairs))[:5],
                hold_success=bool(hold_ok),
                touched=bool(touched),
                steps=step,
                min_tcp_dist=round(min_tcp_dist, 4),
                spawn_dist=round(spawn_dist, 3),
                obj_lift_m=round(obj_z_max - obj_z0, 4),
            )
            if dyn_manager is not None:
                dm = dyn_manager.get_task_metrics()
                record["dynamic_metrics"] = {
                    k: dm.get(k)
                    for k in (
                        "obstacle_spawned", "spawn_robot_distance_m",
                        "minimum_robot_obstacle_distance_m",
                        "obstacle_distance_traveled_m",
                    )
                }
            if args.record_video and frames:
                import cv2
                vdir = out_dir / "videos"
                vdir.mkdir(exist_ok=True)
                tag = "success" if record["success"] else record.get("failure_reason", "fail")
                vp = vdir / f"{scene_id}_task{task_idx:02d}_{task['model_id']}_{tag}.mp4"
                stack = np.stack(frames).astype(np.uint8)
                # crop columns that are void (near-black) through the whole
                # episode — walls/doorways the camera cannot avoid
                colmean = stack.mean(axis=(0, 1, 3))
                good = np.where(colmean > 55)[0]
                if len(good) > 64:
                    x0, x1 = int(good.min()), int(good.max()) + 1
                    x1 -= (x1 - x0) % 2  # even width for the encoder
                    stack = stack[:, :, x0:x1]
                h, w = stack.shape[1:3]
                vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                for fr in stack:
                    vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                vw.release()
                record["video"] = str(vp)
                print(f"  video -> {vp} | obj lift {record['obj_lift_m']*100:.1f} cm")
            if record["success"]:
                n_success += 1
            elif collision:
                n_collision += 1
                record["failure_reason"] = "collision"
            elif touched:
                n_no_hold += 1
                record["failure_reason"] = "no_stable_hold"
            else:
                n_no_contact += 1
                record["failure_reason"] = "no_contact"
            scene_res.append(record)
            print(
                f"[{scene_id}/{task_idx}] {task['model_id']}: "
                f"{'SUCCESS' if record['success'] else record.get('failure_reason')} "
                f"(steps={step}, {time.time()-t0:.1f}s)"
            )

        results["scenes"][scene_id] = scene_res
        # incremental save
        results["summary"] = dict(
            total=total, success=n_success, collision=n_collision,
            no_stable_hold=n_no_hold, no_contact=n_no_contact,
            spawn_failure=n_spawn_fail,
            success_rate=(n_success / total if total else 0.0),
        )
        with open(out_dir / f"{run_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    print("SUMMARY", json.dumps(results["summary"], indent=2))
    env.close()


if __name__ == "__main__":
    main()
