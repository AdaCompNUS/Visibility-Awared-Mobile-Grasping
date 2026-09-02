# VLAPolicy API (contract for the eval agent — exact)

```python
import sys; sys.path.insert(0, "<repo>/baseline/VLA")   # package root is baseline/VLA
from model.policy import VLAPolicy

policy = VLAPolicy.load(ckpt_dir, device="cuda:0")       # ckpt_dir: see below
policy.reset(model_id="002_master_chef_can")             # EVERY episode start
action = policy.act(obs)                                 # -> np.ndarray (13,) float32 in [-1, 1]
```

## `VLAPolicy.load(ckpt_dir, device="cuda:0", dtype=torch.bfloat16)`
`ckpt_dir` must contain `model.pt` (state dict, includes normalization-stat
buffers), `config.json`. Checkpoints are written by train.py under
`baseline/VLA/ckpts/<run>/step_XXXXXX/` (and `latest/`). Runs in the `vla`
pixi env; base HF configs/tokenizer load from `HF_HOME` (already set by env
activation). ~2.5 GB VRAM in bf16.

## `policy.reset(instruction_text=None, model_id=None)`
Call at the start of every episode (it clears the frame history and fixes the
instruction). Pass exactly one of:
- `model_id`: raw YCB id, e.g. `"024_bowl"` — the policy applies the SPEC
  `instruction()` function itself (recommended; guarantees train/eval match), or
- `instruction_text`: a pre-built string (must equal `instruction(model_id)`).

## `policy.act(obs) -> np.ndarray (13,) float32 in [-1, 1]`
One env step. `obs` is a plain dict for a SINGLE env (no batch dimension;
numpy arrays or torch tensors, any device — extra leading singleton dims are
tolerated via reshape):

| key               | shape        | dtype           | content |
|-------------------|--------------|-----------------|---------|
| `fetch_head_rgb`  | (128,128,3)  | uint8           | head camera RGB, 0-255 |
| `fetch_hand_rgb`  | (128,128,3)  | uint8           | hand camera RGB, 0-255 |
| `fetch_head_depth`| (128,128) or (128,128,1) | uint16 or float | **RAW depth in MILLIMETERS** (ManiSkill u16 mm maps as-is). Do NOT normalize — the model applies `1 - tanh(d_mm/1000)` internally. |
| `fetch_hand_depth`| same         | same            | same |
| `state`           | (42,)        | float32         | the RAW mshab 42-d vector, exactly as in `baseline/RL/adapter/sac_policy.py`: `qpos[3:](12) + qvel[3:](12) + tcp_pose_wrt_base(7) + obj_pose_wrt_base(7) + goal_pos_wrt_base(3) + is_grasped(1)`. No normalization — the model normalizes with checkpoint stats. |

### History semantics (first steps of an episode)
The policy keeps an internal 4-frame history (oldest -> newest). `act` pushes
the current obs BEFORE inference, so the action at step t uses frames
[t-3, t-2, t-1, t]. For t < 3 the missing slots are filled by repeating the
earliest available frame (step 0) — identical to training-time padding. Call
`reset(...)` between episodes or the history leaks across episodes.

### Output
13-d action for Fetch `pd_joint_delta_pos`, already clamped to [-1, 1]:
`[arm delta x7 | gripper abs x1 | head_pan, head_tilt, torso x3 | base fwd, ang x2]`.
The policy does **NOT** zero the head dims (indices -5, -4); the eval wrapper
must zero them after the policy, exactly like the RL baseline
(`HEAD_ACTION_DIMS = (-5, -4)` in `baseline/RL/adapter/sac_policy.py`).

### Instruction string (fixed per episode)
`instruction(model_id)`: strip through the first `_`, replace `_`/`-` with
spaces, prefix "pick up the " — e.g. `"002_master_chef_can"` ->
`"pick up the master chef can"`.

### Timing
One `act` call runs the full model (8 images through both vision encoders +
2083-token LLM forward): ~0.1-0.2 s on one A5000 in bf16. Batch size is 1;
run one env per policy instance (each instance holds its own history).
