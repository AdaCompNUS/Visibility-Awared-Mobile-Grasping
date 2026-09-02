"""SG-VLA-style model configuration (single source of truth, JSON-serializable).

Every value that SG-VLA's paper does not pin is documented in
baseline/VLA/DECISIONS.md.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Base weights always come from the repo-relative cache symlink (SPEC: no
# absolute storage paths). The pixi vla env sets HF_HOME the same way; this
# is the fallback when running the interpreter directly. Must run before the
# first `import transformers` (huggingface_hub reads HF_HOME at import time).
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_cache"))


@dataclass
class SGVLAConfig:
    # ---- base model identifiers (loaded from HF_HOME cache) ----
    dino_model: str = "facebook/dinov2-large"
    siglip_model: str = "google/siglip-so400m-patch14-384"
    llm_model: str = "Qwen/Qwen2.5-0.5B"

    # ---- input geometry ----
    image_size: int = 224          # encoder input res (OpenVLA/Prismatic DINOSigLIP-224 convention)
    source_image_size: int = 128   # raw camera res in the dataset
    vision_grid: int = 16          # 224 / 14 = 16 -> 16x16 = 256 tokens per image
    n_views: int = 2               # [head, hand] — fixed order everywhere
    history: int = 4               # 4-frame history, oldest -> newest
    instr_max_len: int = 32        # instruction token budget (padded/truncated)

    # ---- dims ----
    state_dim: int = 42
    action_dim: int = 13
    mask_size: int = 128           # target-mask decoder output res (= head cam res)

    # ---- new-module sizes ----
    projector_hidden: int = 2048   # fused-vision -> LLM MLP hidden
    action_head_hidden: int = 512
    aux_head_hidden: int = 256
    mask_decoder_base: int = 256   # channels at 16x16 before upsampling

    # ---- loss weights (unpublished in SG-VLA; see DECISIONS.md) ----
    w_action: float = 1.0
    w_mask: float = 0.5
    w_robot_pos: float = 0.25
    w_qpos: float = 0.25
    w_grasped: float = 0.25
    w_target_pose: float = 0.25

    # ---- state-vector slices (mshab 42-d layout, see RL adapter docstring) ----
    qpos_slice: tuple = (0, 12)          # qpos[3:] (12)
    target_pose_slice: tuple = (31, 38)  # obj_pose_wrt_base (7): p(3) + q(4, wxyz)
    grasped_index: int = 41              # is_grasped (1)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SGVLAConfig":
        d = json.loads(Path(path).read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known}
        return cls(**d)


def instruction(model_id: str) -> str:
    """SPEC-shared language instruction. Must match between train and eval."""
    name = model_id.split("_", 1)[1] if "_" in model_id else model_id
    name = name.replace("_", " ").replace("-", " ")
    return f"pick up the {name}"
