"""TidyHouse-Pick demonstration dataset generator for the VLA baseline.

Mirrors mshab/utils/gen/gen_data.py (the upstream MS-HAB generation entrypoint)
but records RGB + depth + segmentation + 42-d state + executed actions per
episode into the SPEC schema (baseline/VLA/SPEC.md), while the policy receives
EXACTLY its training observation (2-cam raw-mm depth 3-stack + 42-d state)
through mshab's own wrappers.

Composition (same wrapper order as upstream make_env):
    PickSubtaskTrain-v0 (obs_mode="rgb+depth+segmentation")
      -> VLAPickRecorder (ours, innermost — sees raw sensor obs, like upstream's
         RecordEpisode which upstream also places innermost at obs_mode="rgbd")
      -> FetchDepthObservationWrapper (depth + 42-d state for the policy)
      -> FrameStack(3)
      -> FetchActionWrapper (stationary_head=True zeroes head dims — the
         recorder therefore sees the env-EXECUTED action)
      -> ManiSkillVectorEnv (ignore_terminations=True -> fixed 200-step
         episodes, auto-reset) -> VectorRecordEpisodeStatistics

Filtering: mshab.utils.label_dataset.get_episode_label_and_events with
valid labels ["straightforward_success"] (SUBTASK_TO_EPISODE_LABELS["pick"]),
exactly as upstream gen_data.py.

Run from the repo root (paths are repo-relative; dataset goes through the
baseline/VLA/data symlink):
    CUDA_VISIBLE_DEVICES=4 pixi run -e rl python baseline/VLA/datagen/gen_pick_dataset.py 002_master_chef_can
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces

import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.common import flatten_state_dict

from mshab.agents.sac import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.gen.gen_data import SUBTASK_TO_EPISODE_LABELS
from mshab.utils.label_dataset import get_episode_label_and_events

SEED = 2024  # same as upstream gen_data.py
MAX_EPISODE_STEPS = 200
INFO_KEYS = (  # everything get_pick_episode_label_and_events needs
    "success",
    "subtask_type",
    "is_grasped",
    "robot_target_pairwise_force",
    "robot_force",
    "robot_cumulative_force",
)


class VLAPickRecorder(gym.Wrapper):
    """Innermost wrapper: buffers raw sensor frames + state + executed actions,
    labels finished episodes with mshab's own labeler, and writes valid ones
    incrementally to the SPEC h5 (gzip). Interface points mirror upstream
    RecordEpisode: flush on reset (per env_idx), stop at max_trajectories."""

    def __init__(
        self,
        env,
        h5_path,
        obj_name,
        valid_episode_labels,
        max_trajectories,
    ):
        super().__init__(env)
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.h5_path, "w")
        self.obj_name = obj_name
        self.valid_episode_labels = valid_episode_labels
        self.max_trajectories = max_trajectories

        n = self.base_env.num_envs
        # per-env episode buffers
        self._frames = [[] for _ in range(n)]  # dicts of numpy arrays (T+1 incl reset)
        self._actions = [[] for _ in range(n)]  # (13,) f32, length T
        self._infos = [[] for _ in range(n)]  # dicts of scalars, length T (no reset info)
        self._ep_meta = [dict() for _ in range(n)]  # model_id / obj_id / target_seg_id

        self.num_saved = 0
        # stats over ALL completed (fully rolled out) episodes
        self.label_counts = defaultdict(int)
        self.completed_episodes = 0
        self.completed_success_once = 0
        self.completed_success_at_end = 0
        self.kept_lengths = []

    # ---------------------------------------------------------------- helpers
    @property
    def base_env(self):
        return self.env.unwrapped

    @property
    def reached_max_trajectories(self):
        return self.num_saved >= self.max_trajectories

    def _capture_frame(self, obs):
        """raw obs -> per-modality numpy batch arrays (cheap, one GPU->CPU copy)."""
        sd = obs["sensor_data"]
        state = torch.cat(
            [
                flatten_state_dict(obs["agent"], use_torch=True),
                flatten_state_dict(obs["extra"], use_torch=True),
            ],
            dim=1,
        )  # identical construction to FetchDepthObservationWrapper
        # world base pose (x, y, yaw) — SPEC amendment: the 42-d state strips the
        # base joints, but SG-VLA's global-robot-position aux decoder needs it
        pose = self.base_env.agent.base_link.pose
        p, q = pose.p, pose.q  # (N,3), (N,4) wxyz
        yaw = torch.atan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
        )
        base_pose = torch.stack([p[:, 0], p[:, 1], yaw], dim=1)
        return dict(
            fetch_head_rgb=common.to_numpy(sd["fetch_head"]["rgb"]).astype(
                np.uint8, copy=False
            ),
            fetch_hand_rgb=common.to_numpy(sd["fetch_hand"]["rgb"]).astype(
                np.uint8, copy=False
            ),
            fetch_head_depth=common.to_numpy(sd["fetch_head"]["depth"]).astype(
                np.uint16
            ),
            fetch_hand_depth=common.to_numpy(sd["fetch_hand"]["depth"]).astype(
                np.uint16
            ),
            fetch_head_seg=common.to_numpy(sd["fetch_head"]["segmentation"]).astype(
                np.uint16
            ),
            fetch_hand_seg=common.to_numpy(sd["fetch_hand"]["segmentation"]).astype(
                np.uint16
            ),
            state=common.to_numpy(state).astype(np.float32, copy=False),
            base_pose=common.to_numpy(base_pose).astype(np.float32, copy=False),
        )

    def _append_frame(self, frame_batch, env_idxs):
        for i in env_idxs:
            self._frames[i].append({k: v[i] for k, v in frame_batch.items()})

    def _capture_meta(self, env_idxs):
        uenv = self.base_env
        seg_ids = common.to_numpy(uenv.subtask_objs[0].per_scene_id)
        uids = uenv.task_plan[0].composite_subtask_uids
        for i in env_idxs:
            obj_id = uenv.base_task_plans[(uids[i],)].subtasks[0].obj_id
            self._ep_meta[i] = dict(
                obj_id=obj_id,
                model_id=obj_id.rsplit("-", 1)[0],
                target_seg_id=int(seg_ids[i]),
            )

    # ------------------------------------------------------------------ gym API
    def reset(self, *args, seed=None, options=dict(), **kwargs):
        if options is not None and "env_idx" in options:
            env_idxs = common.to_numpy(options["env_idx"]).tolist()
        else:
            env_idxs = list(range(self.base_env.num_envs))
        self._flush(env_idxs)
        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        # start new episode buffers for the reset envs
        frame_batch = self._capture_frame(obs)
        for i in env_idxs:
            self._frames[i] = []
            self._actions[i] = []
            self._infos[i] = []
        self._append_frame(frame_batch, env_idxs)
        self._capture_meta(env_idxs)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        act_np = common.to_numpy(action).astype(np.float32, copy=False)
        frame_batch = self._capture_frame(obs)
        info_np = {k: common.to_numpy(info[k]) for k in INFO_KEYS}
        n = self.base_env.num_envs
        self._append_frame(frame_batch, range(n))
        for i in range(n):
            self._actions[i].append(act_np[i])
            self._infos[i].append({k: info_np[k][i] for k in INFO_KEYS})
        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------ flush
    def _flush(self, env_idxs):
        for i in env_idxs:
            T = len(self._actions[i])
            if T == 0:
                continue
            # label exactly like upstream RecordEpisode: step infos only
            ep_infos = {
                k: np.stack([step[k] for step in self._infos[i]])
                for k in INFO_KEYS
            }
            ep_success = ep_infos["success"]
            label, _events, _events_verbose = get_episode_label_and_events(
                self.base_env.task_cfgs, ep_success, ep_infos
            )
            self.label_counts[label] += 1
            self.completed_episodes += 1
            self.completed_success_once += bool(ep_success.any())
            self.completed_success_at_end += bool(ep_success[-1])

            if self.reached_max_trajectories:
                continue
            if (
                self.valid_episode_labels is not None
                and label not in self.valid_episode_labels
            ):
                continue

            # SPEC alignment: store (o_t, a_t) for t = 0..T-1 (drop final obs)
            frames = self._frames[i][:T]
            grp = self._h5.create_group(f"traj_{self.num_saved}")
            for key in (
                "fetch_head_rgb",
                "fetch_hand_rgb",
                "fetch_head_depth",
                "fetch_hand_depth",
                "fetch_head_seg",
                "fetch_hand_seg",
            ):
                data = np.stack([f[key] for f in frames])
                grp.create_dataset(
                    key,
                    data=data,
                    dtype=data.dtype,
                    compression="gzip",
                    compression_opts=5,  # upstream RecordEpisode's setting
                    chunks=(min(T, 8), *data.shape[1:]),
                )
            state = np.stack([f["state"] for f in frames])
            base_pose = np.stack([f["base_pose"] for f in frames])
            action = np.stack(self._actions[i])
            grp.create_dataset(
                "state", data=state, dtype=np.float32, compression="gzip", compression_opts=5
            )
            grp.create_dataset(
                "base_pose", data=base_pose, dtype=np.float32, compression="gzip", compression_opts=5
            )
            grp.create_dataset(
                "action", data=action, dtype=np.float32, compression="gzip", compression_opts=5
            )
            meta = self._ep_meta[i]
            grp.attrs["model_id"] = meta["model_id"]
            grp.attrs["obj_id"] = meta["obj_id"]
            grp.attrs["target_seg_id"] = meta["target_seg_id"]
            grp.attrs["label"] = label
            grp.attrs["success"] = bool(ep_success.any())
            grp.attrs["success_at_end"] = bool(ep_success[-1])
            grp.attrs["elapsed_steps"] = T
            self.kept_lengths.append(T)
            self.num_saved += 1
        # free flushed buffers
        for i in env_idxs:
            self._frames[i] = []
            self._actions[i] = []
            self._infos[i] = []

    def finalize(self, extra_attrs=dict()):
        self._h5.attrs["object"] = self.obj_name
        self._h5.attrs["num_episodes"] = self.num_saved
        self._h5.attrs["control_mode"] = "pd_joint_delta_pos"
        self._h5.attrs["env_id"] = "PickSubtaskTrain-v0"
        self._h5.attrs["max_episode_steps"] = MAX_EPISODE_STEPS
        self._h5.attrs["depth_unit"] = "mm"
        self._h5.attrs["seed"] = SEED
        for k, v in extra_attrs.items():
            self._h5.attrs[k] = v
        self._h5.close()


def build_policy_act_fn(envs, cfg_path, ckpt_path, device):
    """Load the released per-object SAC policy exactly as upstream gen_data.py."""
    algo_cfg = parse_cfg(default_cfg_path=str(cfg_path)).algo
    assert algo_cfg.name == "sac", f"expected sac checkpoint, got {algo_cfg.name}"
    obs_space = envs.unwrapped.single_observation_space
    pixels_obs_space: spaces.Dict = obs_space["pixels"]
    model_pixel_obs_space = dict()
    for k, space in pixels_obs_space.items():
        shape, low, high, dtype = space.shape, space.low, space.high, space.dtype
        if len(shape) == 4:
            shape = (shape[0] * shape[1], shape[-2], shape[-1])
            low = low.reshape((-1, *low.shape[-2:]))
            high = high.reshape((-1, *high.shape[-2:]))
        model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
    model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
    policy = SACAgent(
        model_pixel_obs_space,
        obs_space["state"].shape,
        envs.unwrapped.single_action_space.shape,
        actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
        critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
        critic_layer_norm=algo_cfg.critic_layer_norm,
        critic_dropout=algo_cfg.critic_dropout,
        encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
        encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
        cnn_features=list(algo_cfg.cnn_features),
        cnn_filters=list(algo_cfg.cnn_filters),
        cnn_strides=list(algo_cfg.cnn_strides),
        cnn_padding=algo_cfg.cnn_padding,
        log_std_min=algo_cfg.actor_log_std_min,
        log_std_max=algo_cfg.actor_log_std_max,
        device=device,
    )
    policy.eval()
    policy.to(device)
    policy.load_state_dict(torch.load(str(ckpt_path), map_location=device)["agent"])

    def act(obs):
        with torch.no_grad():
            return policy.actor(
                obs["pixels"], obs["state"], compute_pi=False, compute_log_pi=False
            )[0]

    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_name")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--max-episodes", type=int, default=1000)
    ap.add_argument(
        "--max-rollout-episodes",
        type=int,
        default=6000,
        help="safety cap on total rolled-out episodes",
    )
    ap.add_argument("--out-dir", type=str, default="baseline/VLA/data/pick")
    ap.add_argument(
        "--ckpt-dir",
        type=str,
        default="baseline/RL/mshab_checkpoints/rl/tidy_house/pick",
    )
    args = ap.parse_args()

    t_start = time.time()

    # seeding — same as upstream gen_data.py
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    task, subtask, split = "tidy_house", "pick", "train"
    valid_episode_labels = SUBTASK_TO_EPISODE_LABELS[subtask]

    # env config: identical to upstream gen_data.py except obs_mode gains
    # segmentation (depth comes from the same combined texture -> unchanged),
    # configurable num_envs, and no video recording
    env_cfg = EnvConfig(
        env_id=f"{subtask.capitalize()}SubtaskTrain-v0",
        obs_mode="rgb+depth+segmentation",
        num_envs=args.num_envs,
        max_episode_steps=MAX_EPISODE_STEPS,
        record_video=False,
        info_on_video=False,
        debug_video=False,
        debug_video_gen=False,
        continuous_task=True,
        cat_state=True,
        cat_pixels=False,
        task_plan_fp=str(
            ASSET_DIR
            / f"scene_datasets/replica_cad_dataset/rearrange/task_plans/{task}/{subtask}/{split}/{args.obj_name}.json"
        ),
        spawn_data_fp=str(
            ASSET_DIR
            / f"scene_datasets/replica_cad_dataset/rearrange/spawn_data/{task}/{subtask}/{split}/spawn_data.pt"
        ),
        extra_stat_keys=[],
        env_kwargs=dict(
            require_build_configs_repeated_equally_across_envs=False,
            add_event_tracker_info=True,
            robot_force_mult=0.001,
            robot_force_penalty_min=0.2,
            target_randomization=False,
        ),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{args.obj_name}.h5"
    json_path = out_dir / f"{args.obj_name}.json"

    recorder_ref = {}

    def make_recorder(env):
        rec = VLAPickRecorder(
            env,
            h5_path=h5_path,
            obj_name=args.obj_name,
            valid_episode_labels=valid_episode_labels,
            max_trajectories=args.max_episodes,
        )
        recorder_ref["rec"] = rec
        return rec

    envs = make_env(env_cfg, video_path=None, wrappers=[make_recorder])
    rec: VLAPickRecorder = recorder_ref["rec"]

    ckpt_dir = Path(args.ckpt_dir) / args.obj_name
    act_fn = build_policy_act_fn(
        envs, ckpt_dir / "config.yml", ckpt_dir / "policy.pt", device
    )

    envs.reset()  # mirror upstream's initial unseeded reset
    obs, _ = envs.reset(seed=SEED)
    obs = to_tensor(obs, device=device, dtype="float")

    step_num = 0
    last_report = 0.0
    while not rec.reached_max_trajectories:
        action = act_fn(obs)
        obs, _, _, _, _ = envs.step(action)
        obs = to_tensor(obs, device=device, dtype="float")
        step_num += 1
        if time.time() - last_report > 60:
            last_report = time.time()
            print(
                f"[{args.obj_name}] step={step_num} kept={rec.num_saved}"
                f"/{args.max_episodes} completed={rec.completed_episodes}",
                flush=True,
            )
        if rec.completed_episodes >= args.max_rollout_episodes:
            print(f"[{args.obj_name}] hit rollout safety cap, stopping", flush=True)
            break

    wall_time = time.time() - t_start
    rec.finalize()
    envs.close()

    stats = dict(
        object=args.obj_name,
        kept_episodes=rec.num_saved,
        completed_episodes=rec.completed_episodes,
        keep_ratio=rec.num_saved / max(rec.completed_episodes, 1),
        label_histogram=dict(sorted(rec.label_counts.items())),
        success_once_rate=rec.completed_success_once / max(rec.completed_episodes, 1),
        success_at_end_rate=rec.completed_success_at_end
        / max(rec.completed_episodes, 1),
        valid_episode_labels=list(valid_episode_labels),
        mean_episode_length=float(np.mean(rec.kept_lengths)) if rec.kept_lengths else 0,
        num_envs=args.num_envs,
        seed=SEED,
        env_steps=step_num,
        wall_time_sec=round(wall_time, 1),
        h5_bytes=os.path.getsize(h5_path),
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        stationary_head=True,
        schema=dict(
            per_episode_datasets=dict(
                fetch_head_rgb="(T,128,128,3) uint8",
                fetch_hand_rgb="(T,128,128,3) uint8",
                fetch_head_depth="(T,128,128,1) uint16 mm",
                fetch_hand_depth="(T,128,128,1) uint16 mm",
                fetch_head_seg="(T,128,128,1) uint16 per-actor ids",
                fetch_hand_seg="(T,128,128,1) uint16 per-actor ids",
                state="(T,42) float32 (mshab order: qpos[3:], qvel[3:], tcp_pose_wrt_base, obj_pose_wrt_base, goal_pos_wrt_base, is_grasped)",
                base_pose="(T,3) float32 world base x, y, yaw (yaw from base link wxyz quaternion)",
                action="(T,13) float32 env-executed actions in [-1,1] (head dims zeroed)",
            ),
            per_episode_attrs=[
                "model_id",
                "obj_id",
                "target_seg_id",
                "label",
                "success",
                "success_at_end",
                "elapsed_steps",
            ],
            alignment="row t stores (o_t, a_t); o_0 is the reset observation; the post-episode observation o_T is not stored",
        ),
    )
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{args.obj_name}] DONE kept={rec.num_saved} wall={wall_time:.0f}s")
    print(json.dumps({k: v for k, v in stats.items() if k != "schema"}, indent=2))


if __name__ == "__main__":
    main()
