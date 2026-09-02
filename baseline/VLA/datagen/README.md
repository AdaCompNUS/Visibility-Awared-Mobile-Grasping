# VLA datagen — TidyHouse-Pick demonstration dataset

Generates the VLA training dataset (SPEC: `baseline/VLA/SPEC.md`) by rolling out
the RELEASED per-object MS-HAB SAC checkpoints
(`baseline/RL/mshab_checkpoints/rl/tidy_house/pick/<object>/`) with MS-HAB's own
trajectory filtering. 9 objects x 1000 kept episodes.

## Files

- `gen_pick_dataset.py` — generator for one object (one process, one GPU).
- `run_all.sh` — launches all 9 objects across GPUs 4-7 in two waves (5+4
  jobs; ~9 GB RSS each keeps total host RAM ~45 GB, under the 60 GB budget).
- `verify_dataset.py` — post-hoc SPEC verification (schema, decoded frames, stats).

Run from the repo root:

```bash
bash baseline/VLA/datagen/run_all.sh          # everything (~64 envs per job)
# or a single object:
CUDA_VISIBLE_DEVICES=4 pixi run -e rl python baseline/VLA/datagen/gen_pick_dataset.py 002_master_chef_can
pixi run -e rl python baseline/VLA/datagen/verify_dataset.py
```

Output: `baseline/VLA/data/pick/<object>.h5` + `<object>.json` (the `data`
symlink points at bulk storage; scripts only ever use the repo-relative path).

## How this maps onto upstream `mshab/utils/gen/gen_data.py`

Upstream generation = `make_env(EnvConfig(...))` with `RecordEpisode` passed as
the innermost wrapper, a per-object SAC policy stepped deterministically, and
episode filtering by `mshab.utils.label_dataset.get_episode_label_and_events`
with `SUBTASK_TO_EPISODE_LABELS["pick"] = ["straightforward_success"]`.

We keep every element of that pipeline and swap only the recorder:

| upstream (`gen_data.py`) | ours (`gen_pick_dataset.py`) |
|---|---|
| `EnvConfig(env_id="PickSubtaskTrain-v0", obs_mode="rgbd", max_episode_steps=200, continuous_task=True, cat_state=True, cat_pixels=False, frame_stack=3 (default), stationary_head=True (default), task_plan_fp/spawn_data_fp for `train/<object>`, env_kwargs: require_build_configs_repeated_equally_across_envs=False, add_event_tracker_info=True, robot_force_mult=0.001, robot_force_penalty_min=0.2, target_randomization=False)` | identical, except `obs_mode="rgb+depth+segmentation"` and `num_envs=64` (upstream: 252) |
| `RecordEpisode` innermost (records the RAW env obs — upstream's own trick: gen runs at `obs_mode="rgbd"` although policies were trained at `"depth"`, because the depth the policy sees is extracted later by `FetchDepthObservationWrapper`) | `VLAPickRecorder` innermost (records raw rgb/depth/seg/state + executed action) |
| policy: `SACAgent` built from per-object `config.yml`, weights `policy.pt`, deterministic `actor(..., compute_pi=False)[0]` | identical code |
| seeding: `SEED=2024` for random/numpy/torch, `reset()` then `reset(seed=SEED)` | identical |
| filter: label episodes, keep only `straightforward_success`, stop at `MAX_TRAJECTORIES=1000` | identical (`get_episode_label_and_events` on the same per-step info keys, over step infos excluding the reset info — same slicing as upstream `flush_trajectory`) |
| output: ManiSkill trajectory h5 (`obs/...`, gzip-5 images) | SPEC h5 (below), gzip-5 |

Wrapper order (unchanged from upstream `make_env`): recorder →
`FetchDepthObservationWrapper` → `FrameStack(3)` → `FetchActionWrapper`
(`stationary_head=True` zeroes head dims before the recorder sees the action)
→ `ManiSkillVectorEnv(ignore_terminations=True)` → episode stats wrapper.
Because terminations are ignored, every episode is exactly 200 steps and all
envs reset together (auto-reset), which is when flushing/labeling happens.

### Why the policy inputs are identical

- The `minimal` shader renders depth and segmentation from ONE combined
  texture (`PositionSegmentation`); rgb comes from `Color`. `obs_mode="rgbd"`
  (upstream gen) already renders both textures, so adding `+segmentation`
  changes nothing about the depth the policy receives.
- Verified empirically (`scratch probe`, 16 envs, seed 2024): with TF32
  disabled, upstream-config rollout vs ours — reset-time 42-d `state` and both
  cameras' depth are bit-identical; the recorder-rebuilt state matches the
  wrapper's `state` to 0.0; the newest stacked policy depth frame equals the
  raw sensor depth to 0.0.
- Residual difference: GPU rasterization has cross-process jitter of ±1 mm on
  a few depth pixels. This exists equally between two runs of the unmodified
  upstream config (measured), i.e. it is inherent to upstream generation, not
  introduced by us. It makes long rollouts diverge chaotically but is
  distributionally irrelevant (success rates match the released numbers; see
  per-object json stats).

## Output schema (per SPEC)

`<object>.h5`, one group `traj_<k>` per kept episode (k = 0..999):

| dataset | shape | dtype |
|---|---|---|
| `fetch_head_rgb`, `fetch_hand_rgb` | (T,128,128,3) | u8 |
| `fetch_head_depth`, `fetch_hand_depth` | (T,128,128,1) | u16, raw mm |
| `fetch_head_seg`, `fetch_hand_seg` | (T,128,128,1) | u16, per-actor seg ids |
| `state` | (T,42) | f32 (mshab order: qpos[3:], qvel[3:], tcp_pose_wrt_base, obj_pose_wrt_base, goal_pos_wrt_base, is_grasped) |
| `base_pose` | (T,3) | f32, world base x, y, yaw (SPEC amendment for the global-robot-position aux decoder; yaw = atan2(2(wz+xy), 1-2(y²+z²)) from the base link wxyz quaternion) |
| `action` | (T,13) | f32, env-executed (head dims zeroed) |

Alignment: row `t` is the pair `(o_t, a_t)`; `o_0` is the reset observation;
the terminal observation `o_T` is not stored. T = 200 for all episodes.

Attrs per episode: `model_id` (e.g. `002_master_chef_can`), `obj_id` (instance,
e.g. `002_master_chef_can-1`), `target_seg_id` (per-actor segmentation id of
the target in BOTH cameras' id maps — from `subtask_objs[0].per_scene_id`,
captured at episode reset), `label` (always `straightforward_success`),
`success` (success_once), `success_at_end`, `elapsed_steps`.

All image datasets gzip-5 (upstream's own compression), chunked 8 frames — a
4-frame history read touches at most 2 chunks.

`<object>.json`: keep ratio, label histogram over ALL rolled-out episodes,
success rates, wall time, file size, schema notes.

## Deviations from upstream / SPEC (all deliberate, none affect the policy)

1. `obs_mode="rgb+depth+segmentation"` instead of `"rgbd"` — needed for seg
   maps; depth unchanged (same texture), rgb unchanged.
2. `num_envs=64` instead of 252 — 2-3 generation jobs share one 24 GB GPU
   (4-7) and host-RAM budget is 60 GB total; upstream had a whole GPU per job.
   Affects only the interleaving of sampled task plans, not their distribution.
3. Recorder stores the SPEC schema directly (no ManiSkill trajectory h5 +
   conversion pass) — avoids double disk. Filtering/labeling/stop logic follow
   `RecordEpisode.flush_trajectory` (same info slicing, same valid labels,
   same 1000-trajectory stop).
4. Depth/seg stored as u16 (SPEC) — raw sensor values are int16 mm, always
   >= 0, so the cast is lossless.
5. Partial episodes at shutdown are discarded (upstream's flush-on-close is a
   no-op there as well once `max_trajectories` is reached).
6. No videos, no rewards, no env states recorded (SPEC does not ask for them).
7. `base_pose` (T,3) added per SPEC amendment (world base x, y, yaw) — present
   in ALL 9 objects' files.

## Actual generation run (2026-08-29)

All 9 objects were generated with `--num-envs 80`, seed 2024, on GPUs 4-7.
Wave 1 (002, 003, 004, 005, 007) ran as in `run_all.sh`; the second wave (008,
009, 010, 024) was launched per-GPU as soon as that GPU's wave-1 job finished
(same command/parameters as `run_all.sh`'s wave 2, just without the barrier —
object rollouts are independent and identically seeded, so the launch order
does not affect content). Never more than 5 concurrent jobs (~47 GB host RAM).
Per-object wall time 26-33 min; whole dataset ~63 min. Result: 9 x 1000
episodes, 121 GB total, success_once 75.2-84.0% per object, keep ratio
1000/1360-2160 rollouts. See `<object>.json` for per-object stats.
