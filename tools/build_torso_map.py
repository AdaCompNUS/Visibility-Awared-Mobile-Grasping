#!/usr/bin/env python3
"""
Build a POSITION-ONLY torso policy map by FK sampling (no capability-map dependency).

Why this exists
---------------
Your existing saved reachability/capability artifacts (as used in this repo) store:
  [x, y, z, roll, pitch, yaw, visitation_count, manipulability]
They do NOT store torso explicitly. So we cannot recover "best torso" from them.

Instead, we generate a new map using sampling + FK, using your existing IKFast FK interface.

Offline objective (what we maximize)
-----------------------------------
We want a torso policy depending only on target position in the robot/base frame:
  policy(x,y,z) -> torso_lift

We define a simple, non-heuristic reachability metric by sampling joint configurations
uniformly within joint limits (including torso), computing FK, and counting how often the
end-effector position lands in each (x,y,z) bin for each torso bin.

For each discretized position bin p = (x,y,z), we choose:

  t*(p) = argmax_t  Count[ FK(q) lands in bin p AND torso(q) lands in torso-bin t ]

Interpretation:
- Count is proportional to the "volume" of joint space mapping to that position at that torso,
  under the chosen sampling distribution (uniform within joint limits).
- We marginalize out orientation automatically (FK produces whatever orientation comes out).

This produces a compact map you can load later during planning to pick torso from target height.

Output
------
Pickle dict with:
  - xyz_step, torso_step
  - xyz_ijk: (M,3) int bins
  - xyz: (M,3) float bin centers
  - best_torso: (M,) float (meters)
  - best_count: (M,) int counts for the winning torso bin
  - total_count: (M,) int total counts across all torso bins for that xyz
  - z_ref_median: robust implied offset median(z - best_torso) over all bins with enough support

Usage
-----
python3 tools/build_torso_policy_from_fk_sampling.py \
  --out resources/torso_policy_fk_pos_only.pkl \
  --n-samples 300000 --batch 2000 --xyz-step 0.02 --torso-step 0.01
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_anywhere.robot.ik.ikfast_api import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    compute_fk,
)


@dataclass(frozen=True)
class _BinKey:
    ix: int
    iy: int
    iz: int
    it: int


def _bin_index(v: np.ndarray, step: float) -> np.ndarray:
    # Round to nearest bin index, robust for negatives.
    return np.rint(v / step).astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="resources/torso_map.pkl",
        help="Output pickle file path.",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=300_000_000,
        help="Number of joint samples (uniform within joint limits).",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=30000,
        help="Batch size for progress printing (FK is still per-sample).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    ap.add_argument(
        "--xyz-step",
        type=float,
        default=0.02,
        help="Position bin size in meters.",
    )
    ap.add_argument(
        "--torso-step",
        type=float,
        default=0.05,
        help="Torso-lift bin size in meters.",
    )
    ap.add_argument(
        "--min-total-count",
        type=int,
        default=10,
        help="Only use xyz bins with at least this many total samples when computing z_ref.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    lower = np.asarray(JOINT_LIMITS_LOWER, dtype=np.float64)
    upper = np.asarray(JOINT_LIMITS_UPPER, dtype=np.float64)

    if lower.shape != (8,) or upper.shape != (8,):
        raise SystemExit(f"Expected 8 joint limits. Got {lower.shape} / {upper.shape}")

    xyz_step = float(args.xyz_step)
    torso_step = float(args.torso_step)

    # Counts keyed by (ix,iy,iz,it)
    counts_4d: dict[_BinKey, int] = defaultdict(int)
    # Total counts per xyz (for z_ref stats + confidence)
    total_counts_xyz: dict[tuple[int, int, int], int] = defaultdict(int)

    n = int(args.n_samples)
    batch = int(args.batch)

    print(
        "Sampling joints uniformly within limits and accumulating FK position counts..."
    )
    print(f"  n_samples: {n}")
    print(f"  xyz_step:  {xyz_step} m")
    print(f"  torso_step:{torso_step} m")
    print(f"  torso limits (m): [{lower[0]:.4f}, {upper[0]:.4f}]")

    # Main loop (FK via IKFast is per-sample; keep it simple and predictable)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        q = rng.uniform(lower, upper, size=(end - start, 8)).astype(np.float64)

        # Bin torso directly from joint[0]
        torso = q[:, 0]
        it = _bin_index(torso, torso_step)

        for i in range(q.shape[0]):
            pos, _rot = compute_fk(q[i].tolist())
            pos = np.asarray(pos, dtype=np.float64)
            ix, iy, iz = _bin_index(pos, xyz_step).tolist()
            key = _BinKey(ix=ix, iy=iy, iz=iz, it=int(it[i]))
            counts_4d[key] += 1
            total_counts_xyz[(ix, iy, iz)] += 1

        if (start // batch) % 10 == 0:
            print(
                f"  progress: {end}/{n} samples, unique (xyz,torso) bins: {len(counts_4d)}"
            )

    # Reduce to best torso per xyz
    best_for_xyz: dict[tuple[int, int, int], tuple[int, int]] = {}  # (it, best_count)
    for k, c in counts_4d.items():
        xyz = (k.ix, k.iy, k.iz)
        prev = best_for_xyz.get(xyz)
        if prev is None or c > prev[1]:
            best_for_xyz[xyz] = (k.it, c)

    xyz_ijk = np.array(list(best_for_xyz.keys()), dtype=np.int32)
    best_it = np.array([best_for_xyz[tuple(x)][0] for x in xyz_ijk], dtype=np.int32)
    best_count = np.array([best_for_xyz[tuple(x)][1] for x in xyz_ijk], dtype=np.int32)
    total_count = np.array(
        [total_counts_xyz[tuple(x)] for x in xyz_ijk], dtype=np.int32
    )

    xyz = xyz_ijk.astype(np.float32) * xyz_step
    best_torso = best_it.astype(np.float32) * torso_step

    # Implied z_ref stats (optional convenience)
    z = xyz[:, 2].astype(np.float64)
    torso_m = best_torso.astype(np.float64)
    mask = total_count >= int(args.min_total_count)
    z_ref_samples = (
        (z[mask] - torso_m[mask]) if np.any(mask) else np.array([], dtype=np.float64)
    )
    z_ref_median = (
        float(np.median(z_ref_samples)) if z_ref_samples.size else float("nan")
    )

    out = {
        "objective": "best_torso(xyz) = argmax_t Count[FK(q)->xyz AND torso(q)->t], with uniform joint sampling",
        "n_samples": int(n),
        "xyz_step": float(xyz_step),
        "torso_step": float(torso_step),
        "joint_limits_lower": lower.astype(np.float32),
        "joint_limits_upper": upper.astype(np.float32),
        "xyz_ijk": xyz_ijk,
        "xyz": xyz,
        "best_torso": best_torso,
        "best_count": best_count,
        "total_count": total_count,
        "min_total_count_for_z_ref": int(args.min_total_count),
        "z_ref_median": z_ref_median,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        pickle.dump(out, f)

    print("\nSaved position-only torso policy map:")
    print(f"  path: {outp}")
    print(f"  xyz bins: {xyz.shape[0]}")
    print(
        f"  best_torso range (m): [{float(np.min(best_torso)):.3f}, {float(np.max(best_torso)):.3f}]"
    )
    print(
        f"  z_ref_median (from bins with total_count>={args.min_total_count}): {z_ref_median:.4f}"
    )


if __name__ == "__main__":
    main()
