# VLA Baseline SPEC (single source of truth for all agents)

SG-VLA-style policy (arXiv 2603.22760, no code released) reproduced at
published scale on the Pick task, trained on MS-HAB-pipeline data, evaluated
on our 400-task benchmark. Read this before writing any code. If you must
deviate, document the deviation in your files AND report it.

## Hard constraints
- GPUs **4-7 only** (`CUDA_VISIBLE_DEVICES` from {4,5,6,7}). Never touch 0-3.
- Total host RAM across ALL our processes <= 125 GB (half the server).
  Budgets: datagen <=60 GB, training <=40 GB, eval <=10 GB. No in-RAM dataset
  caching; h5 lazy reads.
- NO absolute storage paths anywhere in code, configs, or scripts. Always use
  the repo-relative symlinks baseline/VLA/data , baseline/VLA/ckpts ,
  baseline/VLA/hf_cache (they point into /home/storage/tianrun/ — big files
  cannot live on the repo partition, ~30 GB free).
- NO modifications outside baseline/VLA/ (no grasp_anywhere/, no experiments/,
  no site-packages patching, no baseline/RL/ edits). Root pixi.toml is managed
  by the orchestrator only.

## Environments
- `pixi run -e rl <cmd>`  — mshab + ManiSkill 3.0.0b18 (data generation).
- `pixi run -e vla <cmd>` — torch 2.4.1+cu124, transformers, mani_skill 3.0.1
  (training + benchmark evaluation). HF_HOME -> baseline/VLA/hf_cache (set by env activation).

## Action space (13-dim, all in [-1,1]) — identical to the RL baseline
[arm delta-pos x7 | gripper abs x1 | body delta-pos x3 (head_pan, head_tilt,
torso) | base x2 (fwd vel, ang vel)] under Fetch `pd_joint_delta_pos`.
Data is generated with stationary_head=True (head dims are zeroed by the env
wrapper); at eval time the policy's head dims are zeroed the same way.

## Dataset (generated, not downloaded)
- Source: mshab gen pipeline (`mshab/utils/gen/gen_data.py` logic) rolling out
  the RELEASED per-object RL checkpoints in
  baseline/RL/mshab_checkpoints/rl/tidy_house/pick/<object>/ , with their
  trajectory filtering (mshab/utils/label_dataset.py, pick-valid labels).
- Scale: 9 objects x 1000 kept episodes (their released standard).
- Layout: baseline/VLA/data/pick/<object>.h5 (+<object>.json
  stats). gzip-compressed. Per episode store EVERY step:
    fetch_head_rgb  (T,128,128,3) u8      fetch_hand_rgb  (T,128,128,3) u8
    fetch_head_depth(T,128,128,1) u16 mm  fetch_hand_depth(T,128,128,1) u16 mm
    fetch_head_seg  (T,128,128,1) u16     fetch_hand_seg  (T,128,128,1) u16
    state (T,42) f32   # the mshab 42-dim vector (see baseline/RL/adapter/sac_policy.py docstring)
    action (T,13) f32  # env-executed actions in [-1,1]
    attrs: model_id, target_seg_id (per-actor segmentation id of the target),
           episode label, success
- Segmentation: ManiSkill per-actor segmentation from BOTH cameras; store raw
  id maps + target id (binary target mask derived at train time).

## Language instruction (shared function — must match between train and eval)
def instruction(model_id: str) -> str:
    name = model_id.split("_", 1)[1] if "_" in model_id else model_id
    name = name.replace("_", " ").replace("-", " ")
    return f"pick up the {name}"
# e.g. "002_master_chef_can" -> "pick up the master chef can"
# "070-a_colored_wood_blocks" -> instruction uses the part after the first "_".

## Model (~1.3B, SG-VLA composition; all bases public)
- Vision: DINOv2 ViT-L/14 (`facebook/dinov2-large`) + SigLIP SO400M
  (`google/siglip-so400m-patch14-384`), Prismatic-style dual encoding: run
  both per image, spatially align token grids, channel-concat features.
- LLM: `Qwen/Qwen2.5-0.5B`.
- Inputs per step: 2 views x 4-timestep history, RGB resized to encoder-native
  res + depth normalized `1 - tanh(d_mm/1000)` (SG-VLA's formula) as an
  auxiliary channel/token stream (implementer's documented choice), language
  instruction, proprio state (42-d) embedded as tokens.
- New modules (from scratch): projector (vision->LLM space), 13-d continuous
  action head (L1 loss; SG-VLA's flow-matching expert is a documented
  stretch goal, not MVP), and 5 auxiliary decoders co-trained from the shared
  features: (1) target binary mask (head cam), (2) global robot position (3),
  (3) qpos (12), (4) is_grasped (1, BCE), (5) target pose wrt base (7).
- Training: single-stage (single subtask -> no progressive curriculum), bf16,
  AdamW lr 2e-5 cosine, warmup 2k steps, global batch 64, ~30 sampled frames
  per episode per epoch, 2-3 epochs, FSDP (or sharded DDP) across GPUs 4-7,
  gradient checkpointing, checkpoints + tensorboard to
  baseline/VLA/ckpts/. Unpublished SG-VLA hyperparameters
  follow Prismatic/OpenVLA conventions — document every such default.

## Evaluation
- Mirror baseline/RL/eval_rl_on_benchmark.py EXACTLY (read it first): same
  referee (grasp_anywhere.utils.monitor_core), same easy/hard protocols, same
  DynamicBenchmarkManager wiring, same success rule (2 s hold + zero
  collision), same output schema. Swap only the policy: VLA checkpoint, obs
  from 2-cam RGB+depth history + instruction + state, head action dims zeroed.
- Runs in the `vla` env (mani_skill 3.0.1 — the SAME simulator version as the
  main benchmark and RL results).
