"""Inference-time policy wrapper. Contract: see POLICY_API.md (keep in sync)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import SGVLAConfig, instruction
from .sgvla import SGVLA


class VLAPolicy:
    """`VLAPolicy.load(ckpt_dir)` -> `.reset(instruction=...)` -> `.act(obs)`."""

    def __init__(self, model: SGVLA, tokenizer, device: torch.device, dtype=torch.bfloat16):
        self.model = model.to(device=device, dtype=dtype).eval()
        # normalization buffers must stay fp32-accurate; keep a fp32 copy
        self.model.state_mean.data = self.model.state_mean.data.float()
        self.model.state_std.data = self.model.state_std.data.float()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.cfg = model.cfg
        self._hist: Optional[deque] = None
        self._instr_ids = None
        self._instr_mask = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_dir: str | Path, device: str | torch.device = "cuda:0",
             dtype=torch.bfloat16) -> "VLAPolicy":
        ckpt_dir = Path(ckpt_dir)
        cfg = SGVLAConfig.from_json(ckpt_dir / "config.json")
        model = SGVLA(cfg)
        sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "position_ids" not in k]
        assert not real_missing and not unexpected, (real_missing, unexpected)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.llm_model)
        return cls(model, tok, torch.device(device), dtype)

    # ------------------------------------------------------------------
    def reset(self, instruction_text: Optional[str] = None, model_id: Optional[str] = None):
        """Call at every episode start. Provide instruction_text OR model_id."""
        assert (instruction_text is None) != (model_id is None), \
            "pass exactly one of instruction_text / model_id"
        if instruction_text is None:
            instruction_text = instruction(model_id)
        tok = self.tokenizer([instruction_text], padding="max_length", truncation=True,
                             max_length=self.cfg.instr_max_len, return_tensors="pt")
        self._instr_ids = tok["input_ids"].to(self.device)
        self._instr_mask = tok["attention_mask"].to(self.device)
        self._hist = deque(maxlen=self.cfg.history)

    # ------------------------------------------------------------------
    @staticmethod
    def _np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _frame(self, obs: Dict) -> Dict[str, np.ndarray]:
        rgb = np.stack([
            self._np(obs["fetch_head_rgb"]).reshape(128, 128, 3),
            self._np(obs["fetch_hand_rgb"]).reshape(128, 128, 3),
        ]).astype(np.uint8)                                    # (2,128,128,3)
        depth = np.stack([
            self._np(obs["fetch_head_depth"]).reshape(128, 128),
            self._np(obs["fetch_hand_depth"]).reshape(128, 128),
        ]).astype(np.float32)                                  # (2,128,128) raw mm
        return {"rgb": rgb, "depth": depth}

    @torch.no_grad()
    def act(self, obs: Dict) -> np.ndarray:
        """obs per POLICY_API.md -> action (13,) float32 in [-1, 1].

        Head-camera pan/tilt dims are NOT zeroed here — the eval env wrapper
        does that (mirrors the RL baseline).
        """
        assert self._hist is not None, "call reset(...) before act(...)"
        self._hist.append(self._frame(obs))
        frames = list(self._hist)
        while len(frames) < self.cfg.history:      # pad by repeating the oldest
            frames.insert(0, frames[0])

        rgb = torch.from_numpy(np.stack([f["rgb"] for f in frames]))[None].to(self.device)
        depth = torch.from_numpy(np.stack([f["depth"] for f in frames]))[None].to(self.device)
        state = torch.from_numpy(
            self._np(obs["state"]).astype(np.float32).reshape(1, self.cfg.state_dim)
        ).to(self.device)

        act = self.model.predict_action(rgb, depth, self._instr_ids, self._instr_mask, state)
        return act[0].float().cpu().numpy()
