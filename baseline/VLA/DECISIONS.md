# DECISIONS.md — every hyperparameter SG-VLA's paper does not specify

SG-VLA (arXiv 2603.22760) released no code, weights, or appendix-level
hyperparameters. SPEC.md pins: dual DINOv2-L + SigLIP-SO400M encoding,
Qwen2.5-0.5B, 13-d L1 action head, 5 aux decoders, depth = `1 - tanh(d_mm/1000)`,
2 views x 4-frame history, 42-d proprio, bf16, AdamW lr 2e-5 cosine, warmup 2k,
global batch 64, ~30 frames/episode/epoch, 2-3 epochs, FSDP + grad
checkpointing. Everything below is OUR choice, labeled by convention source.

## Architecture

| Decision | Value | Rationale / convention |
|---|---|---|
| Encoder input resolution | 224x224 for BOTH encoders (16x16=256 tokens each, patch 14) | Prismatic/OpenVLA-convention default ("DINOSigLIP 224px", the OpenVLA backbone). SigLIP-384's position embeddings are bicubically interpolated (`interpolate_pos_encoding=True`); DINOv2 interpolates natively. Source frames are 128x128, so 384px would triple LLM sequence cost with zero added information. |
| Resize + pixel norm | bilinear+antialias to 224 on-GPU inside the model; DINOv2 ImageNet mean/std, SigLIP 0.5/0.5 | HF processor values; done in-model so train and eval can never diverge. |
| Grid alignment / fusion | identical 16x16 grids, channel-concat per token (1024+1152=2176) | Prismatic dual-encoder convention (SPEC-pinned), no interpolation needed at equal grids. |
| DINOv2 CLS/register tokens | dropped (patch tokens only) | Prismatic convention. |
| Depth stream | per-image 128x128 normalized depth cut into the same 16x16 grid (8x8 px = 64 values/patch), `Linear(64->2176)`, ADDED to the fused vision token before the projector | SPEC leaves "channel/token stream" to the implementer. Additive embedding keeps sequence length at 2083 (vs +2048 tokens for a separate stream) and is spatially aligned token-for-token. |
| Projector | 2-layer MLP 2176 -> 2048 -> 896, GELU, from scratch | Prismatic uses an MLP projector; hidden size is ours. |
| Token order | `[instruction(32, padded)] [vision t-3..t, per frame head|hand, 256 each] [AUX] [proprio] [ACT]` = 2083 tokens | Causal LLM: instruction first so vision attends to language (needed for the language-conditioned target mask); AUX query placed BEFORE the proprio token so aux decoders cannot copy qpos/is_grasped/target-pose out of the proprio input; ACT last sees everything. |
| Frame/view identity | learned frame (4) + view (2) embeddings added to vision tokens | ours; standard practice for multi-image sequences. |
| Proprio embedding | `(state-mean)/std` -> MLP 42->896->896 -> 1 token | SPEC says "embedded as tokens"; 1 token is ours. |
| Instruction budget | 32 tokens, right-padded (Qwen2.5 tokenizer, no chat template, plain text) | ours; longest pick instruction is ~8 tokens. |
| Action head | MLP 896->512->13 on the [ACT] hidden state, LINEAR output, L1 loss on raw targets, clamp to [-1,1] at inference only | OpenVLA-style single-step action (no chunking — SPEC names a 13-d head). No tanh: dataset actions saturate at +-1 and tanh kills those gradients. |
| Action normalization | identity | env-executed actions are already in [-1,1] by construction; OpenVLA's q1/q99 rescale is a no-op here. Documented so the eval side knows there is NO unnormalization step. |
| Aux decoders | mask: LLM outputs at current-frame head-cam token positions -> 16x16x896 -> conv + 3x nearest-upsample -> 128x128 logits. Scalars (robot_pos 3, qpos 12, is_grasped 1, target_pose 7): separate MLPs 896->256->d on the [AUX] hidden state | "co-trained from shared features" (SPEC); the concrete readout is ours. |
| Aux targets | qpos = state[0:12], target_pose = state[31:38], is_grasped = state[41], each normalized with the SAME state stats; global robot position from an OPTIONAL per-episode h5 key (`robot_pos`/`base_pose`/`robot_base_pose`) — loss masked to 0 when the datagen file does not carry it (the SPEC h5 schema does not include global base pose; the 42-d state strips base x/y/theta) | deviation-by-necessity, reported. |
| Loss weights | action 1.0, mask 0.5, robot_pos / qpos / grasped / target_pose 0.25 each | unpublished in SG-VLA; ours (action-dominant, aux as regularizers). Configurable in `SGVLAConfig`. |
| Total params | 1237.8M = DINOv2 304.4M + SigLIP 428.2M + Qwen2.5-0.5B 494.0M + new modules 11.1M | within SPEC's ~1.3B. |

## Data pipeline

| Decision | Value | Rationale |
|---|---|---|
| Episode discovery | any h5 group containing an `action` dataset | SPEC does not pin group naming; robust to the datagen agent's layout. |
| 30 frames/episode/epoch | index = (episode, slot) pairs; each access draws a fresh uniform random frame (with replacement if T<30) | "~30 sampled frames" (SPEC); fresh randomness each epoch is ours. |
| History padding | [t-3..t] clamped at 0 (first frame repeated) | matches VLAPolicy exactly (POLICY_API.md). |
| Depth dtype | u16 mm kept RAW through the loader; normalized inside the model | one formula implementation, shared with eval. |
| State/aux normalization | mean/std over all `state` rows, std clamped >= 1e-2 (goal-pos dims are constant 0) | Prismatic/OpenVLA normalize proprio; clamp is ours. Stats stored as model BUFFERS -> inside every checkpoint. |
| success filter | off by default (`require_success=True` available) | datagen already keeps only pick-valid episodes. |
| Workers | 4/rank x 4 ranks = 16 total, no in-RAM caching, lazy per-worker h5 handles | SPEC RAM budget. |

## Optimization (unpinned parts)

| Decision | Value | Rationale |
|---|---|---|
| AdamW betas / eps / weight decay | (0.9, 0.95) / 1e-8 / 0.1 (no decay on ndim<=1 params + embeddings) | Prismatic-convention default. |
| LR schedule | linear warmup 2000 steps -> cosine to 0 over total steps; all modules one LR (2e-5), full fine-tune incl. both vision towers | SPEC pins lr/warmup/cosine; single param group + decay-to-0 is Prismatic convention. |
| Grad clip | global norm 1.0 | Prismatic/OpenVLA convention. |
| Global batch 64 | micro-batch 4/GPU x 4 GPUs x grad-accum 4 | measured to fit comfortably (<~16 GiB/GPU); accum derived automatically from `--micro-batch`. |
| Precision | FSDP MixedPrecision: fp32 sharded master params, bf16 compute/reduce, fp32 buffers; losses computed in fp32 | Prismatic trains pure-bf16; we keep fp32 masters (safer, fits). Single-GPU debug path is pure bf16 — MEASURED: fp32 params + fp32 AdamW do NOT fit on one 24 GB A5000, which also rules out the DDP fallback (SPEC said measure, not guess). |
| FSDP wrapping | FULL_SHARD, auto-wrap {Dinov2Layer, SiglipEncoderLayer, Qwen2DecoderLayer}, use_orig_params, limit_all_gathers | standard. |
| Grad checkpointing | non-reentrant, on all three backbones | SPEC-pinned on; flavor ours. |
| Epochs | default 3 (SPEC says 2-3) | more passes for a small model. |
| Checkpoints | every 1000 optimizer steps + end of every epoch: `model.pt` (full fp32 state dict incl. norm-stat buffers), `config.json`, `norm_stats.json`, `trainer_state.json`; keep last 3 + `latest` symlink | cadence ours. `--resume` warm-starts model weights (optimizer state intentionally not persisted — 10 GB/save; acceptable for a <1-day run). |
| Seeding | torch seed = `--seed`+rank; dataloader workers auto-seeded per worker | ours. |

## Known deviations from SG-VLA (reported)
1. **Flow-matching action expert**: SPEC marks it a stretch goal; we ship the
   L1 regression head (single-step, no chunking).
2. **Global-robot-position aux head** trains only if datagen adds a base-pose
   key to the h5 (schema gap noted above); otherwise its loss is masked and
   the head simply stays near init. All other 4 aux decoders always train.
3. **Single-stage training** (no progressive curriculum) — SPEC-pinned for the
   single Pick subtask.
