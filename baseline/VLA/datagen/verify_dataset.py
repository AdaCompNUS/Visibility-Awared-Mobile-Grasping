"""Verify the generated TidyHouse-Pick dataset against baseline/VLA/SPEC.md.

For each <object>.h5 under the data dir:
  - kept-episode count, mean episode length, label distribution (from stats json)
  - schema check on every episode group (dataset names/shapes/dtypes/attrs)
  - sampled frames decode: rgb non-black, depth in plausible mm range,
    seg contains target_seg_id in >= one sampled head frame
  - file size

Usage (any python with h5py+numpy, e.g.):
  pixi run -e rl python baseline/VLA/datagen/verify_dataset.py [--data-dir baseline/VLA/data/pick]
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np

OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "024_bowl",
]

IMG_KEYS = {
    "fetch_head_rgb": ((128, 128, 3), np.uint8),
    "fetch_hand_rgb": ((128, 128, 3), np.uint8),
    "fetch_head_depth": ((128, 128, 1), np.uint16),
    "fetch_hand_depth": ((128, 128, 1), np.uint16),
    "fetch_head_seg": ((128, 128, 1), np.uint16),
    "fetch_hand_seg": ((128, 128, 1), np.uint16),
}
REQ_ATTRS = ["model_id", "target_seg_id", "label", "success"]


def verify_object(h5_path: Path, json_path: Path, n_sample_eps=25, seed=0):
    rng = random.Random(seed)
    problems = []
    with h5py.File(h5_path, "r") as f:
        eps = list(f.keys())
        n = len(eps)
        lengths = []
        seg_hit_eps = 0
        rgb_ok_eps = 0
        depth_ok_eps = 0
        sampled = rng.sample(eps, min(n_sample_eps, n))
        for ep in eps:
            g = f[ep]
            T = g["action"].shape[0]
            lengths.append(T)
            for k, (shape, dtype) in IMG_KEYS.items():
                if k not in g:
                    problems.append(f"{ep}: missing {k}")
                    continue
                if tuple(g[k].shape) != (T, *shape) or g[k].dtype != dtype:
                    problems.append(
                        f"{ep}: {k} shape/dtype {g[k].shape} {g[k].dtype}"
                    )
            if tuple(g["state"].shape) != (T, 42) or g["state"].dtype != np.float32:
                problems.append(f"{ep}: state {g['state'].shape} {g['state'].dtype}")
            if "base_pose" not in g:
                problems.append(f"{ep}: missing base_pose")
            elif tuple(g["base_pose"].shape) != (T, 3) or g["base_pose"].dtype != np.float32:
                problems.append(f"{ep}: base_pose {g['base_pose'].shape} {g['base_pose'].dtype}")
            if tuple(g["action"].shape) != (T, 13):
                problems.append(f"{ep}: action {g['action'].shape}")
            for a in REQ_ATTRS:
                if a not in g.attrs:
                    problems.append(f"{ep}: missing attr {a}")
        for ep in sampled:
            g = f[ep]
            T = g["action"].shape[0]
            t = rng.randrange(T)
            rgb = g["fetch_head_rgb"][t]
            depth = g["fetch_head_depth"][t]
            seg_head = g["fetch_head_seg"][:]  # all frames, check target visibility
            tid = int(g.attrs["target_seg_id"])
            if rgb.mean() > 5:
                rgb_ok_eps += 1
            if 100 < np.median(depth[depth > 0]) < 20000:
                depth_ok_eps += 1
            if (seg_head == tid).any():
                seg_hit_eps += 1
            act = g["action"][:]
            if np.abs(act).max() > 1.0 + 1e-5:
                problems.append(f"{ep}: action out of [-1,1]")
            if np.abs(act[:, -5:-3]).max() > 0:
                problems.append(f"{ep}: head action dims nonzero")
        report = dict(
            episodes=n,
            mean_len=float(np.mean(lengths)),
            min_len=int(np.min(lengths)),
            max_len=int(np.max(lengths)),
            sampled=len(sampled),
            rgb_nonblack=f"{rgb_ok_eps}/{len(sampled)}",
            depth_plausible=f"{depth_ok_eps}/{len(sampled)}",
            target_in_head_seg=f"{seg_hit_eps}/{len(sampled)}",
            size_gb=round(h5_path.stat().st_size / 1e9, 2),
            problems=problems[:10],
            n_problems=len(problems),
        )
    if json_path.exists():
        stats = json.loads(json_path.read_text())
        report["label_histogram"] = stats.get("label_histogram")
        report["success_once_rate"] = round(stats.get("success_once_rate", -1), 4)
        report["completed_episodes"] = stats.get("completed_episodes")
        report["wall_time_sec"] = stats.get("wall_time_sec")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="baseline/VLA/data/pick")
    ap.add_argument("--objects", nargs="*", default=OBJECTS)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    total = 0
    for obj in args.objects:
        h5p = data_dir / f"{obj}.h5"
        if not h5p.exists():
            print(f"{obj}: MISSING")
            continue
        rep = verify_object(h5p, data_dir / f"{obj}.json")
        total += rep["size_gb"]
        print(f"=== {obj} ===")
        print(json.dumps(rep, indent=2))
    print(f"TOTAL SIZE: {total:.2f} GB")


if __name__ == "__main__":
    main()
