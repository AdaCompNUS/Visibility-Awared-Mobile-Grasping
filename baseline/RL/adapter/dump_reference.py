"""Dump reference (obs, action) pairs using MS-HAB's OWN code path.

Runs in the `rl` pixi environment. Creates the same eval env used for
checkpoint validation (PickSubtaskTrain-v0 + spawn data), loads the released
SAC checkpoint through mshab's classes, rolls deterministic episodes, and saves
per-step observations + actions plus environment metadata (joint order, link
names, control frequencies).

The saved npz certifies the plain-torch port in ../adapter/sac_policy.py: the
main-environment port must reproduce these actions bit-for-bit (fp32 tolerance)
from these observations.

Usage (from baseline/RL):
  pixi run -e rl python adapter/dump_reference.py [--steps 64] [--envs 4]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# Strict fp32 determinism for port certification: TF32 and cudnn autotuning
# otherwise make conv outputs differ across runs/processes at ~1e-2 scale
# (raw-mm depth inputs amplify reduced-precision paths).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from mshab.agents.sac import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.config import parse_cfg
from gymnasium import spaces

from mani_skill import ASSET_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--out", type=str, default="adapter/reference_pairs.npz")
    args = ap.parse_args()

    device = torch.device("cuda")
    rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

    env_cfg = EnvConfig(
        env_id="PickSubtaskTrain-v0",
        num_envs=args.envs,
        max_episode_steps=200,
        continuous_task=True,
        cat_state=True,
        cat_pixels=False,
        frame_stack=3,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=True,
        task_plan_fp=str(rearrange / "task_plans/tidy_house/pick/train/all.json"),
        spawn_data_fp=str(rearrange / "spawn_data/tidy_house/pick/train/spawn_data.pt"),
        record_video=False,
        info_on_video=False,
        extra_stat_keys=[],
        env_kwargs=dict(require_build_configs_repeated_equally_across_envs=False),
    )
    envs = make_env(env_cfg, video_path=None)
    uenv = envs.unwrapped
    obs, _ = envs.reset(seed=0, options=dict(reconfigure=True))

    # ---- metadata for the port ----
    active_joints = [j.name for j in uenv.agent.robot.active_joints]
    meta = dict(
        active_joints=np.array(active_joints),
        base_link=uenv.agent.base_link.name,
        tcp_link=uenv.agent.tcp.name,
        control_freq=uenv.control_freq,
        sim_freq=uenv.sim_freq,
        action_dim=int(envs.single_action_space.shape[0]),
        state_dim=int(obs["state"].shape[-1]),
    )
    print({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()})

    # ---- policy via mshab's own class, exactly as evaluate.py builds it ----
    algo_cfg = parse_cfg(
        default_cfg_path="mshab_checkpoints/rl/tidy_house/pick/all/config.yml"
    ).algo
    obs_space = uenv.single_observation_space
    pixels_space: spaces.Dict = obs_space["pixels"]
    model_pixel_obs_space = dict()
    for k, space in pixels_space.items():
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
        envs.single_action_space.shape,
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
    policy.load_state_dict(
        torch.load(
            "mshab_checkpoints/rl/tidy_house/pick/all/policy.pt", map_location=device
        )["agent"]
    )
    policy.to(device)

    from mshab.utils.array import to_tensor

    hand, head, states, actions = [], [], [], []
    for _ in range(args.steps):
        tobs = to_tensor(obs, device=device, dtype="float")
        with torch.no_grad():
            act = policy.actor(
                tobs["pixels"], tobs["state"], compute_pi=False, compute_log_pi=False
            )[0]
        hand.append(tobs["pixels"]["fetch_hand_depth"].cpu().numpy())
        head.append(tobs["pixels"]["fetch_head_depth"].cpu().numpy())
        states.append(tobs["state"].cpu().numpy())
        actions.append(act.cpu().numpy())
        obs, _, _, _, _ = envs.step(act)

    out = Path(args.out)
    np.savez_compressed(
        out,
        fetch_hand_depth=np.concatenate(hand),
        fetch_head_depth=np.concatenate(head),
        state=np.concatenate(states),
        action=np.concatenate(actions),
        **meta,
    )
    print(f"saved {out} | pairs={len(states) * args.envs}")
    envs.close()


if __name__ == "__main__":
    main()
