"""Lazy H5 dataset for the SG-VLA baseline (SPEC schema).

Layout (produced by the datagen agent): baseline/VLA/data/pick/<object>.h5,
one HDF5 group per episode (group naming is NOT pinned by SPEC, so episodes
are discovered as any group that contains an `action` dataset). Per episode:

  fetch_head_rgb  (T,128,128,3) u8    fetch_hand_rgb  (T,128,128,3) u8
  fetch_head_depth(T,128,128,1) u16mm fetch_hand_depth(T,128,128,1) u16mm
  fetch_head_seg  (T,128,128,1) u16   fetch_hand_seg  (T,128,128,1) u16
  state (T,42) f32                    action (T,13) f32
  attrs: model_id, target_seg_id, label, success

Design:
- No in-RAM caching; h5 files are opened lazily per worker process.
- ~`frames_per_episode` (30) frames sampled per episode per epoch: the map
  index enumerates (episode, slot) pairs and each __getitem__ draws a fresh
  uniform random frame from that episode (fresh randomness every epoch).
- 4-frame history [t-3..t] clamped at episode start (frame 0 repeated) —
  identical semantics to VLAPolicy at eval (see model/POLICY_API.md).
- Binary target mask = (fetch_head_seg[t] == target_seg_id).
- Optional per-episode `robot_pos` / `base_pose` dataset (T,>=3) provides the
  global-robot-position auxiliary target; absent -> loss masked (valid=False).
- `SyntheticVLADataset` fabricates correctly-shaped samples for tests /
  throughput calibration before real data lands.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model.config import SGVLAConfig, instruction

DATA_DIR = Path(__file__).resolve().parent / "data" / "pick"  # repo-relative symlink
ROBOT_POS_KEYS = ("robot_pos", "base_pose", "robot_base_pose")


# ----------------------------------------------------------------------
def _find_episode_groups(f: h5py.File) -> List[str]:
    """All group paths that directly contain an `action` dataset."""
    out: List[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "action" in obj and isinstance(obj["action"], h5py.Dataset):
            out.append(name)

    f.visititems(visit)
    if not out and "action" in f:  # single-episode flat file (defensive)
        out.append("/")
    return sorted(out)


class VLAPickDataset(Dataset):
    """Lazy loader over every <object>.h5 under `data_dir`."""

    def __init__(
        self,
        cfg: SGVLAConfig,
        data_dir: str | Path = DATA_DIR,
        frames_per_episode: int = 30,
        files: Optional[Sequence[str | Path]] = None,
        require_success: bool = False,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.frames_per_episode = frames_per_episode
        self.data_dir = Path(data_dir)
        self.require_success = require_success
        self._seed = seed
        self._handles: Dict[int, h5py.File] = {}  # file_idx -> open handle (per worker)

        self.files = sorted(Path(p) for p in files) if files else sorted(self.data_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"no .h5 files under {self.data_dir}")

        # ---- episode index (metadata only, closed afterwards) ----
        self.episodes: List[tuple] = []  # (file_idx, group_path, T, model_id, target_seg_id, has_robot_pos)
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                for g in _find_episode_groups(f):
                    grp = f[g] if g != "/" else f
                    if self.require_success and not bool(grp.attrs.get("success", True)):
                        continue
                    T = int(grp["action"].shape[0])
                    if T < 1:
                        continue
                    model_id = grp.attrs.get("model_id", path.stem)
                    if isinstance(model_id, bytes):
                        model_id = model_id.decode()
                    seg_id = int(grp.attrs.get("target_seg_id", -1))
                    rp = next((k for k in ROBOT_POS_KEYS if k in grp), None)
                    self.episodes.append((fi, g, T, str(model_id), seg_id, rp))
        if not self.episodes:
            raise RuntimeError(f"no episodes found in {self.data_dir}")

    # -- lazy per-worker file handles ----------------------------------
    def _file(self, fi: int) -> h5py.File:
        h = self._handles.get(fi)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    def __getstate__(self):  # drop open handles when pickled into workers
        d = dict(self.__dict__)
        d["_handles"] = {}
        return d

    def __len__(self) -> int:
        return len(self.episodes) * self.frames_per_episode

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        fi, gpath, T, model_id, seg_id, rp_key = self.episodes[idx // self.frames_per_episode]
        t = int(np.random.randint(T))
        hist = [max(0, t - k) for k in range(cfg.history - 1, -1, -1)]  # oldest..newest
        uniq = sorted(set(hist))                        # h5 fancy index must be increasing
        pos = {u: i for i, u in enumerate(uniq)}
        sel = [pos[i] for i in hist]

        grp = self._file(fi)[gpath]
        rgb = np.stack(
            [grp["fetch_head_rgb"][uniq][sel], grp["fetch_hand_rgb"][uniq][sel]], axis=1
        )  # (4, 2, 128, 128, 3) u8
        depth = np.stack(
            [grp["fetch_head_depth"][uniq][sel], grp["fetch_hand_depth"][uniq][sel]], axis=1
        ).astype(np.float32).reshape(cfg.history, cfg.n_views, 128, 128)  # raw mm

        seg = np.asarray(grp["fetch_head_seg"][t]).reshape(128, 128)
        mask = (seg == seg_id).astype(np.float32)

        state = np.asarray(grp["state"][t], dtype=np.float32)
        action = np.asarray(grp["action"][t], dtype=np.float32)

        if rp_key is not None:
            robot_pos = np.asarray(grp[rp_key][t], dtype=np.float32).reshape(-1)[:3]
            rp_valid = True
        else:
            robot_pos, rp_valid = np.zeros(3, dtype=np.float32), False

        return {
            "rgb": torch.from_numpy(rgb),
            "depth_mm": torch.from_numpy(depth),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "target_mask": torch.from_numpy(mask),
            "robot_pos": torch.from_numpy(robot_pos),
            "robot_pos_valid": torch.tensor(rp_valid),
            "instruction": instruction(model_id),
        }

    # ------------------------------------------------------------------
    def compute_norm_stats(self, max_episodes_per_file: Optional[int] = None) -> Dict[str, list]:
        """Streamed mean/std of `state` (+ robot_pos when present). No RAM blowup."""
        n = 0
        s1 = np.zeros(self.cfg.state_dim, np.float64)
        s2 = np.zeros(self.cfg.state_dim, np.float64)
        rn = 0
        r1 = np.zeros(3, np.float64)
        r2 = np.zeros(3, np.float64)
        per_file: Dict[int, int] = {}
        for fi, gpath, T, _, _, rp_key in self.episodes:
            if max_episodes_per_file is not None:
                if per_file.get(fi, 0) >= max_episodes_per_file:
                    continue
                per_file[fi] = per_file.get(fi, 0) + 1
            st = np.asarray(self._file(fi)[gpath]["state"], dtype=np.float64)
            n += st.shape[0]
            s1 += st.sum(0)
            s2 += (st ** 2).sum(0)
            if rp_key is not None:
                rp = np.asarray(self._file(fi)[gpath][rp_key], dtype=np.float64)[:, :3]
                rn += rp.shape[0]
                r1 += rp.sum(0)
                r2 += (rp ** 2).sum(0)
        mean = s1 / max(n, 1)
        std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 1e-12))
        out = {"state_mean": mean.tolist(), "state_std": std.tolist()}
        if rn:
            rm = r1 / rn
            out["robot_pos_mean"] = rm.tolist()
            out["robot_pos_std"] = np.sqrt(np.maximum(r2 / rn - rm ** 2, 1e-12)).tolist()
        return out


# ----------------------------------------------------------------------
class SyntheticVLADataset(Dataset):
    """Correctly-shaped fabricated samples (schema identical to VLAPickDataset)."""

    def __init__(self, cfg: SGVLAConfig, n_episodes: int = 64, frames_per_episode: int = 30):
        self.cfg = cfg
        self.n = n_episodes * frames_per_episode

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        cfg = self.cfg
        g = torch.Generator().manual_seed(idx)
        mask = torch.zeros(128, 128)
        cx, cy = torch.randint(16, 112, (2,), generator=g)
        mask[cy - 8: cy + 8, cx - 8: cx + 8] = 1.0
        return {
            "rgb": torch.randint(0, 256, (cfg.history, cfg.n_views, 128, 128, 3),
                                 generator=g, dtype=torch.uint8),
            "depth_mm": torch.rand(cfg.history, cfg.n_views, 128, 128, generator=g) * 3000.0,
            "state": torch.randn(cfg.state_dim, generator=g),
            "action": (torch.rand(cfg.action_dim, generator=g) * 2 - 1),
            "target_mask": mask,
            "robot_pos": torch.randn(3, generator=g),
            "robot_pos_valid": torch.tensor(True),
            "instruction": instruction("002_master_chef_can"),
        }

    def compute_norm_stats(self, **_):
        return {"state_mean": [0.0] * self.cfg.state_dim, "state_std": [1.0] * self.cfg.state_dim}


# ----------------------------------------------------------------------
def make_collate(tokenizer, instr_max_len: int = 32):
    """Batch dict collation + instruction tokenization (runs in the worker)."""

    def collate(samples: List[dict]) -> Dict[str, torch.Tensor]:
        batch = {
            k: torch.stack([s[k] for s in samples])
            for k in ("rgb", "depth_mm", "state", "action", "target_mask",
                      "robot_pos", "robot_pos_valid")
        }
        tok = tokenizer(
            [s["instruction"] for s in samples],
            padding=True, truncation=True, max_length=instr_max_len, return_tensors="pt",
        )
        batch["instr_ids"] = tok["input_ids"]
        batch["instr_mask"] = tok["attention_mask"]
        return batch

    return collate


def build_dataset(cfg: SGVLAConfig, synthetic: bool = False, data_dir=DATA_DIR,
                  frames_per_episode: int = 30, n_synthetic_episodes: int = 64, **kw):
    if synthetic:
        return SyntheticVLADataset(cfg, n_synthetic_episodes, frames_per_episode)
    return VLAPickDataset(cfg, data_dir=data_dir, frames_per_episode=frames_per_episode, **kw)


if __name__ == "__main__":
    # smoke test: python dataset.py [--synthetic]
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    cfg = SGVLAConfig()
    ds = build_dataset(cfg, synthetic=args.synthetic, data_dir=args.data_dir)
    print(f"dataset: {len(ds)} samples")
    s = ds[0]
    for k, v in s.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else v} "
              f"{v.dtype if hasattr(v, 'dtype') else ''}")
