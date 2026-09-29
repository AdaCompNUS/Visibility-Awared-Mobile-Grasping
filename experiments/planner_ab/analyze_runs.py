"""Compare benchmark runs: success, failure reasons, paired test, planner-call stats.

usage: pixi run python experiments/planner_ab/analyze_runs.py NAME=results_dir [NAME=results_dir ...]
"""

import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load(path):
    with open(f"{path}/benchmark_results.json") as f:
        r = json.load(f)
    tasks = {}
    for scene, s in r["scenes"].items():
        for t in s["tasks"]:
            tasks[(scene, t["task_id"])] = t
    return tasks


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main(args):
    runs = {}
    for a in args:
        name, path = a.split("=", 1)
        runs[name] = load(path)
    names = list(runs)

    print("== Task outcomes ==")
    for name, tasks in runs.items():
        n = len(tasks)
        k = sum(t["success"] for t in tasks.values())
        lo, hi = wilson(k, n)
        reasons = Counter(t["failure_reason"] for t in tasks.values() if not t["success"])
        coll = sum(bool(t.get("collision_detected")) for t in tasks.values())
        print(
            f"{name:>10}: {k}/{n} = {100*k/n:.1f}% (95% CI {100*lo:.1f}-{100*hi:.1f})"
            f" | collision_detected in {coll} tasks | failures: {dict(reasons.most_common())}"
        )

    if len(names) >= 2:
        a, b = names[0], names[1]
        common = sorted(set(runs[a]) & set(runs[b]))
        only_a = sum(runs[a][k]["success"] and not runs[b][k]["success"] for k in common)
        only_b = sum(runs[b][k]["success"] and not runs[a][k]["success"] for k in common)
        print(
            f"\n== Paired ({len(common)} common tasks) == {a} only: {only_a}, "
            f"{b} only: {only_b}, exact McNemar p = {mcnemar_exact(only_a, only_b):.3g}"
        )
        for reason in ("collision", "planning_failure"):
            ra = sum(runs[a][k]["failure_reason"] == reason for k in common)
            rb = sum(runs[b][k]["failure_reason"] == reason for k in common)
            print(f"   {reason}: {a}={ra} {b}={rb}")

    print("\n== Planner calls ==")
    for name, tasks in runs.items():
        calls = defaultdict(list)
        for t in tasks.values():
            for c in t.get("planning_calls", []):
                calls[c["kind"]].append(c)
        for kind, cs in sorted(calls.items()):
            ok = [c for c in cs if c["success"]]
            invalid = Counter(c.get("status") for c in cs if not c["success"])
            w_ok = [c["wall_time_ms"] for c in ok]
            w_all = [c["wall_time_ms"] for c in cs if c.get("status") not in ("invalid_start", "invalid_goal")]
            print(
                f"{name:>10} {kind:>10}: {len(ok)}/{len(cs)} solved ({100*len(ok)/max(1,len(cs)):.1f}%)"
                f" | wall ms solved: median {pct(w_ok,50):.1f} p90 {pct(w_ok,90):.1f} p99 {pct(w_ok,99):.1f}"
                f" | wall ms all attempts: mean {np.mean(w_all) if w_all else float('nan'):.1f}"
                f" | failures {dict(invalid)}"
            )
        per_task = [sum(c["wall_time_ms"] for c in t.get("planning_calls", [])) for t in tasks.values()]
        print(f"{name:>10} total planning wall time per task: median {pct(per_task,50):.0f} ms, mean {np.mean(per_task):.0f} ms")


if __name__ == "__main__":
    main(sys.argv[1:])
