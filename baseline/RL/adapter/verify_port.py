"""Verify the plain-torch actor port against mshab-generated reference pairs.

Runs in the MAIN pixi environment (no mshab import). Loads the released
checkpoint into adapter.sac_policy.SACActor, replays the observations dumped by
dump_reference.py, and compares actions.

Usage (from baseline/RL):  pixi run python adapter/verify_port.py
"""

import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sac_policy import PIXEL_KEYS, load_actor  # noqa: E402


def main():
    ref = np.load(Path(__file__).parent / "reference_pairs.npz", allow_pickle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = load_actor(
        str(Path(__file__).parent.parent / "mshab_checkpoints/rl/tidy_house/pick/all/policy.pt"),
        device=device,
    )
    pixels = {
        k: torch.as_tensor(ref[k], dtype=torch.float32, device=device)
        for k in PIXEL_KEYS
    }
    state = torch.as_tensor(ref["state"], dtype=torch.float32, device=device)
    want = torch.as_tensor(ref["action"], dtype=torch.float32, device=device)

    got = []
    B = 64
    for i in range(0, state.shape[0], B):
        got.append(actor({k: v[i : i + B] for k, v in pixels.items()}, state[i : i + B]))
    got = torch.cat(got)

    diff = (got - want).abs()
    print(f"pairs={state.shape[0]}  max|diff|={diff.max().item():.3e}  mean|diff|={diff.mean().item():.3e}")
    print("metadata:", {k: ref[k].tolist() if ref[k].ndim else ref[k].item() for k in ("base_link", "tcp_link", "control_freq", "sim_freq", "action_dim", "state_dim")})
    print("active_joints:", list(ref["active_joints"]))
    assert diff.max().item() < 1e-4, "PORT MISMATCH"
    print("PORT VERIFIED")


if __name__ == "__main__":
    main()
