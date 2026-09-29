#!/usr/bin/env python3
"""Aggregate repeated paper benchmark runs with confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any


FAILURE_COUNTERS = (
    "collision_failures",
    "grasping_failures",
    "ik_failures",
    "out_of_reachability",
    "perception_failure",
    "planning_failures",
    "dynamic_interaction_failures",
)


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return [None, None]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _result_tasks(result: dict[str, Any]):
    for scene_id, scene in result["scenes"].items():
        for task in scene["tasks"]:
            yield scene_id, task


def validate_result(result: dict[str, Any], expected_tasks: int = 400) -> None:
    """Reject partial or structurally inconsistent benchmark artifacts."""
    tasks = list(_result_tasks(result))
    summary = result["summary"]
    if len(tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} task records, found {len(tasks)}")
    if int(summary["total_tasks"]) != expected_tasks:
        raise ValueError("summary total_tasks does not match task records")
    successes = sum(bool(task["success"]) for _, task in tasks)
    if successes != int(summary["successful_tasks"]):
        raise ValueError("summary successful_tasks does not match task records")
    failures = sum(not bool(task["success"]) for _, task in tasks)
    if failures != int(summary["failed_tasks"]):
        raise ValueError("summary failed_tasks does not match task records")
    if successes + failures != expected_tasks:
        raise ValueError("success and failure counts do not partition all tasks")


def summarize_runs(
    runs: list[dict[str, Any]], confidence: float = 0.95
) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot summarize an empty run group")
    totals = [int(run["summary"]["total_tasks"]) for run in runs]
    successes = [int(run["summary"]["successful_tasks"]) for run in runs]
    run_rates = [success / total for success, total in zip(successes, totals)]
    pooled_total = sum(totals)
    pooled_successes = sum(successes)
    summary: dict[str, Any] = {
        "runs": len(runs),
        "tasks_per_run": totals,
        "successes_per_run": successes,
        "success_rates_per_run": run_rates,
        "run_mean_success_rate": statistics.mean(run_rates),
        "run_std_success_rate": statistics.stdev(run_rates) if len(runs) > 1 else 0.0,
        "pooled_total": pooled_total,
        "pooled_successes": pooled_successes,
        "pooled_success_rate": pooled_successes / pooled_total,
        "pooled_success_wilson": wilson_interval(
            pooled_successes, pooled_total, confidence
        ),
    }
    failure_metrics = {}
    for key in FAILURE_COUNTERS:
        counts = [int(run["summary"].get(key, 0)) for run in runs]
        pooled_count = sum(counts)
        failure_metrics[key] = {
            "counts_per_run": counts,
            "pooled_count": pooled_count,
            "pooled_rate": pooled_count / pooled_total,
            "pooled_wilson": wilson_interval(pooled_count, pooled_total, confidence),
        }
    summary["failures"] = failure_metrics
    return summary


def aggregate_campaign(campaign_dir: Path, confidence: float = 0.95):
    manifest_path = campaign_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_tasks = int(manifest.get("expected_tasks_per_run", 400))
    expected_repeats = int(manifest["repeats"])
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any], str]]] = defaultdict(
        list
    )
    errors = []

    for entry in manifest["runs"]:
        if entry.get("status") != "completed":
            continue
        result_path = campaign_dir / entry["result_path"]
        try:
            result = json.loads(result_path.read_text())
            validate_result(result, expected_tasks)
            expected_hash = entry.get("config_sha256")
            actual_hash = result.get("config", {}).get("config_sha256")
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"config hash mismatch: expected {expected_hash}, got {actual_hash}"
                )
            groups[(entry["method"], entry["condition"])].append(
                (int(entry["repeat"]), result, entry["result_path"])
            )
        except Exception as exc:
            errors.append({"result_path": entry.get("result_path"), "error": str(exc)})

    aggregated: dict[str, Any] = {
        "campaign_id": manifest["campaign_id"],
        "confidence": confidence,
        "interval": "pooled Wilson score interval",
        "expected_repeats": expected_repeats,
        "expected_tasks_per_run": expected_tasks,
        "groups": {},
        "validation_errors": errors,
    }
    for (method, condition), items in sorted(groups.items()):
        items.sort(key=lambda item: item[0])
        group = summarize_runs([item[1] for item in items], confidence)
        group["repeats"] = [item[0] for item in items]
        group["result_paths"] = [item[2] for item in items]
        group["complete"] = len(items) == expected_repeats
        aggregated["groups"][f"{method}/{condition}"] = group

    aggregated["acceptance"] = acceptance_checks(aggregated)
    return aggregated


def acceptance_checks(aggregated: dict[str, Any]):
    groups = aggregated["groups"]
    checks: dict[str, Any] = {}
    ours_static = groups.get("ours/static")
    ours_dynamic = groups.get("ours/dynamic")
    checks["ours_static_mean_above_70"] = bool(
        ours_static and ours_static["run_mean_success_rate"] > 0.70
    )
    checks["ours_dynamic_mean_above_60"] = bool(
        ours_dynamic and ours_dynamic["run_mean_success_rate"] > 0.60
    )
    for condition, ours in (("static", ours_static), ("dynamic", ours_dynamic)):
        peers = [
            (name, group)
            for name, group in groups.items()
            if name.endswith(f"/{condition}") and not name.startswith("ours/")
        ]
        checks[f"ours_highest_{condition}"] = bool(
            ours
            and peers
            and all(
                ours["run_mean_success_rate"] > peer["run_mean_success_rate"]
                for _, peer in peers
            )
        )
    checks["all_groups_complete"] = bool(groups) and all(
        group["complete"] for group in groups.values()
    )
    checks["all_passed"] = all(checks.values())
    return checks


def write_outputs(campaign_dir: Path, aggregated: dict[str, Any]) -> None:
    aggregate_dir = campaign_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "summary.json").write_text(
        json.dumps(aggregated, indent=2) + "\n"
    )

    fieldnames = [
        "method",
        "condition",
        "runs",
        "pooled_successes",
        "pooled_total",
        "pooled_success_rate",
        "wilson_low",
        "wilson_high",
        "run_mean_success_rate",
        "run_std_success_rate",
        "successes_per_run",
    ]
    with (aggregate_dir / "success_rates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, group in sorted(aggregated["groups"].items()):
            method, condition = name.split("/", 1)
            writer.writerow(
                {
                    "method": method,
                    "condition": condition,
                    "runs": group["runs"],
                    "pooled_successes": group["pooled_successes"],
                    "pooled_total": group["pooled_total"],
                    "pooled_success_rate": group["pooled_success_rate"],
                    "wilson_low": group["pooled_success_wilson"][0],
                    "wilson_high": group["pooled_success_wilson"][1],
                    "run_mean_success_rate": group["run_mean_success_rate"],
                    "run_std_success_rate": group["run_std_success_rate"],
                    "successes_per_run": ";".join(
                        str(value) for value in group["successes_per_run"]
                    ),
                }
            )

    lines = [
        f"# Paper rerun: {aggregated['campaign_id']}",
        "",
        "Rates are mean ± sample standard deviation across independent runs; "
        "brackets are pooled 95% Wilson confidence intervals.",
        "",
        "| Method | Condition | Runs | Success rate | 95% CI | Per-run successes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, group in sorted(aggregated["groups"].items()):
        method, condition = name.split("/", 1)
        low, high = group["pooled_success_wilson"]
        lines.append(
            f"| {method} | {condition} | {group['runs']} | "
            f"{100 * group['run_mean_success_rate']:.2f}% ± "
            f"{100 * group['run_std_success_rate']:.2f}% | "
            f"[{100 * low:.2f}%, {100 * high:.2f}%] | "
            f"{group['successes_per_run']} |"
        )
    lines.extend(["", "## Acceptance", ""])
    for name, passed in aggregated["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    (aggregate_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    aggregated = aggregate_campaign(args.campaign_dir, args.confidence)
    write_outputs(args.campaign_dir, aggregated)
    print((args.campaign_dir / "aggregate" / "summary.md").read_text())


if __name__ == "__main__":
    main()
