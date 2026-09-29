"""Replay dumped benchmark planning queries through VampBackend and FetchPlanningBackend.

usage: pixi run python experiments/planner_ab/replay_queries.py OUT.json QUERY_DIR [QUERY_DIR ...]

Every path either planner returns is re-checked with the fetch_planning model
(full sphere-radius range), densified along its edges.
"""

import glob
import json
import sys
import time

import numpy as np

from grasp_anywhere.robot.utils.fetch_planning_backend import FetchPlanningBackend
from grasp_anywhere.robot.utils.vamp_backend import VampBackend

out_path, dirs = sys.argv[1], sys.argv[2:]
files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz"))
vb, fb = VampBackend(), FetchPlanningBackend()
checker = FetchPlanningBackend()  # full-radius ground truth for returned paths
rows = []
for i, f in enumerate(files):
    d = np.load(f)
    meta = json.loads(str(d["meta"]))
    if meta.get("attached"):
        continue
    clouds = [(d[f"cloud_{j}"].tolist(), float(r)) for j, r in enumerate(d["cloud_radii"])]
    for b in (vb, fb, checker):
        b.clear_pointclouds()
        for pts, r in clouds:
            b.add_pointcloud(pts, r)
    start, goal = d["start"], d["goal"]
    row = {"file": f, "kind": meta["kind"], "source": meta["motion_planner"]}
    for name, b in (("vamp", vb), ("fetch_planning", fb)):
        if meta["kind"] == "whole_body":
            sb, sa, gb, ga = start[:3], start[3:], goal[:3], goal[3:]
            if not (b.validate(sa, sb) and b.validate(ga, gb)):
                row[name] = {"status": "invalid"}
                continue
            t0 = time.perf_counter()
            r = b.plan_whole_body(sa.tolist(), ga.tolist(), sb.tolist(), gb.tolist())
            wall = (time.perf_counter() - t0) * 1e3
            ok = r["success"]
            arm, base = (r["arm_path"], r["base_configs"]) if ok else (None, None)
        else:
            base_pose = meta["base"]
            b.set_base_params(base_pose[2], base_pose[0], base_pose[1])
            t0 = time.perf_counter()
            w, _ = b.plan_arm(base_pose, np.asarray(start), np.asarray(goal))
            wall = (time.perf_counter() - t0) * 1e3
            ok = w is not None
            arm, base = (w, [base_pose] * len(w)) if ok else (None, None)
        res = {"status": "success" if ok else "failed", "wall_ms": wall}
        if ok:
            res["n_waypoints"] = len(arm)
            res["path_collides_full_model"] = bool(checker.path_in_collision(arm, base, -1))
        row[name] = res
    rows.append(row)
    if i % 20 == 0:
        print(i, len(files), row, flush=True)

json.dump(rows, open(out_path, "w"))

for kind in ("whole_body", "arm"):
    rs = [r for r in rows if r["kind"] == kind]
    print(f"\n== {kind}: {len(rs)} queries ==")
    for name in ("vamp", "fetch_planning"):
        st = [r[name]["status"] for r in rs]
        ok = [r[name] for r in rs if r[name]["status"] == "success"]
        w = np.array([r[name]["wall_ms"] for r in rs if "wall_ms" in r[name]])
        wok = np.array([o["wall_ms"] for o in ok])
        coll = sum(o["path_collides_full_model"] for o in ok)
        print(
            f"{name:>15}: solved {len(ok)}/{len(rs) - st.count('invalid')} valid "
            f"(invalid start/goal {st.count('invalid')}) | wall ms solved median "
            f"{np.median(wok) if len(wok) else float('nan'):.1f} p90 {np.percentile(wok, 90) if len(wok) else float('nan'):.1f} "
            f"| mean all {w.mean() if len(w) else float('nan'):.1f} | paths colliding (full model) {coll}"
        )
    both = [r for r in rs if r["vamp"]["status"] != "invalid" and r["fetch_planning"]["status"] != "invalid"]
    ov = sum(r["vamp"]["status"] == "success" and r["fetch_planning"]["status"] != "success" for r in both)
    of = sum(r["fetch_planning"]["status"] == "success" and r["vamp"]["status"] != "success" for r in both)
    print(f"  valid for both: {len(both)} | solved only by vamp {ov} | only by fetch_planning {of}")
