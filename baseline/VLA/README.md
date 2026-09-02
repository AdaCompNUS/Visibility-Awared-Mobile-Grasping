# VLA Baseline (SG-VLA-style)

SG-VLA-style ~1.24B policy per [SPEC.md](SPEC.md): DINOv2-L + SigLIP-SO400M
dual vision encoding (Prismatic-style channel-concat on aligned 16x16 grids),
Qwen2.5-0.5B LLM, from-scratch projector, 13-d L1 action head, 5 auxiliary
decoders. Every hyperparameter the paper does not pin is documented in
[DECISIONS.md](DECISIONS.md). Eval contract: [model/POLICY_API.md](model/POLICY_API.md).

## Layout
- `model/` — config, dual vision backbone, SGVLA module, `VLAPolicy`
- `dataset.py` — lazy h5 loader (SPEC schema) + `--synthetic` fabricated data
- `train.py` — bf16 FSDP training, tensorboard + checkpoints under `ckpts/`
- `data/`, `ckpts/`, `hf_cache/` — repo-relative symlinks into storage (never
  use absolute storage paths)

## Commands (repo root, GPUs 4-7 ONLY)
```bash
# full training (FSDP full-shard, global batch 64)
CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vla torchrun --standalone \
    --nproc_per_node=4 baseline/VLA/train.py --epochs 3 --run-name sgvla_main

# single-GPU debug (pure bf16; fp32+AdamW does NOT fit a 24 GB A5000)
CUDA_VISIBLE_DEVICES=4 pixi run -e vla python baseline/VLA/train.py \
    --synthetic --debug-steps 3 --micro-batch 2 --global-batch 8

# throughput calibration on synthetic data
CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vla torchrun --standalone \
    --nproc_per_node=4 baseline/VLA/train.py --synthetic --calib-steps 100
```

## Measured calibration (2026-08-29, synthetic data, real configuration)
- Params 1237.8M = DINOv2-L 304.4M + SigLIP-SO400M 428.2M + Qwen2.5-0.5B
  494.0M + new modules 11.1M. 2083 tokens/sample.
- 4x A5000, FSDP full-shard, global batch 64 (micro 4 x accum 4): median
  **9.12 s/step** (mean 9.82 s with datagen co-tenants on the GPUs), 7.1 GiB
  max-allocated VRAM/rank, 13.8 GiB host PSS for the whole job (4 ranks + 16
  workers).
- 9000 episodes x 30 frames = 270,000 samples/epoch = 4,218 steps/epoch:
  **~21.4 h for 2 epochs, ~32 h for 3 epochs** (exclusive GPUs; add ~10-20%
  under co-tenancy).
- fp32 model + fp32 AdamW measured OOM on one 24 GB A5000 -> DDP fallback is
  ruled out; single-GPU debug runs pure bf16.
- Policy inference: ~116 ms/act, 2.45 GiB VRAM (bf16, one GPU).

## Inference
```python
import sys; sys.path.insert(0, "baseline/VLA")
from model.policy import VLAPolicy
policy = VLAPolicy.load("baseline/VLA/ckpts/<run>/latest", device="cuda:0")
policy.reset(model_id="002_master_chef_can")   # every episode start
action = policy.act(obs)                       # (13,) float32 in [-1,1]
```
Exact obs schema (keys, dtypes, RAW mm depth, history semantics):
[model/POLICY_API.md](model/POLICY_API.md). The eval wrapper zeroes head dims
(-5, -4) after the policy, like the RL baseline.
