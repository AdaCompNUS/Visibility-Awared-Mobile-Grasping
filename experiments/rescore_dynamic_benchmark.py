#!/usr/bin/env python3
"""Rescore a completed dynamic benchmark at a stricter global path threshold.

The dynamic benchmark records the continuous spawn-to-route distance and maximum
route deviation for every task. Those metrics are sufficient to apply a stricter
``nav_path_change_threshold`` without rerunning the robot: the threshold is only
used to classify whether a completed replan measurably changed the route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def path_changed_at_threshold(metrics: dict[str, Any], threshold: float) -> bool:
    """Apply the benchmark's global route-change predicate to recorded metrics."""
    spawn_distance = metrics.get("spawn_route_distance_m")
    maximum_deviation = metrics.get("maximum_route_deviation_m")
    return bool(
        metrics.get("obstacle_spawned", False)
        and spawn_distance is not None
        and maximum_deviation is not None
        and float(spawn_distance) <= threshold
        and float(maximum_deviation) >= threshold
    )


def _iter_tasks(results: dict[str, Any]):
    for scene_id, scene in results["scenes"].items():
        for task in scene["tasks"]:
            yield scene_id, scene, task


def _rebuild_summary(results: dict[str, Any]) -> dict[str, int]:
    tasks = [task for _, _, task in _iter_tasks(results)]
    summary = {
        "total_tasks": len(tasks),
        "successful_tasks": sum(bool(task["success"]) for task in tasks),
        "failed_tasks": sum(not bool(task["success"]) for task in tasks),
        "collision_failures": 0,
        "grasping_failures": 0,
        "out_of_reachability": 0,
        "perception_failure": 0,
        "planning_failures": 0,
        "ik_failures": 0,
        "dynamic_tasks": len(tasks),
        "dynamic_obstacle_spawned_tasks": sum(
            bool(task["dynamic_metrics"]["obstacle_spawned"]) for task in tasks
        ),
        "dynamic_path_affected_tasks": sum(
            bool(task["dynamic_metrics"]["path_changed"]) for task in tasks
        ),
        "dynamic_replans": sum(
            int(task["dynamic_metrics"]["replan_count"]) for task in tasks
        ),
        "dynamic_interaction_failures": sum(
            task.get("failure_reason") == "dynamic_path_unaffected" for task in tasks
        ),
    }

    reason_to_counter = {
        "collision": "collision_failures",
        "grasping_failure": "grasping_failures",
        "out_of_reachability": "out_of_reachability",
        "perception_failure": "perception_failure",
        "ik_failure": "ik_failures",
        "planning_failure": "planning_failures",
        "navigation_failure": "planning_failures",
    }
    for task in tasks:
        counter = reason_to_counter.get(task.get("failure_reason"))
        if counter is not None:
            summary[counter] += 1
    return summary


def rescore_results(
    source: dict[str, Any], threshold: float, *, source_path: str, source_sha256: str
) -> dict[str, Any]:
    """Return a provenance-preserving rescore at a stricter global threshold."""
    results = copy.deepcopy(source)
    benchmark = results.get("config", {}).get("benchmark", {})
    if not benchmark.get("enable_dynamic_challenges", False):
        raise ValueError("source is not a dynamic benchmark result")
    if not benchmark.get("require_dynamic_path_change", True):
        raise ValueError("source did not require a dynamic path change")

    original_threshold = float(benchmark["nav_path_change_threshold"])
    threshold = float(threshold)
    if threshold < original_threshold:
        raise ValueError(
            "only stricter thresholds can be reconstructed from completed results"
        )

    for _, scene, task in _iter_tasks(results):
        metrics = task.get("dynamic_metrics")
        if metrics is None:
            raise ValueError("every task must contain dynamic_metrics")
        changed = path_changed_at_threshold(metrics, threshold)
        metrics["path_changed"] = changed
        task["dynamic_path_affected"] = changed

        if not changed:
            task["success"] = False
            task["failure_reason"] = "dynamic_path_unaffected"

        scene_tasks = scene["tasks"]
        scene["scene_success_rate"] = sum(
            bool(scene_task["success"]) for scene_task in scene_tasks
        ) / len(scene_tasks)

    benchmark["nav_path_change_threshold"] = threshold
    results["summary"] = _rebuild_summary(results)
    results["rescore"] = {
        "method": "global_dynamic_path_threshold",
        "source_results": source_path,
        "source_sha256": source_sha256,
        "original_threshold_m": original_threshold,
        "threshold_m": threshold,
        "rescored_at": datetime.now(timezone.utc).isoformat(),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore a completed dynamic benchmark at a stricter path threshold"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    rescored = rescore_results(
        source,
        args.threshold,
        source_path=str(args.source),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2) + "\n")

    summary = rescored["summary"]
    print(
        f"Rescored {summary['total_tasks']} tasks: "
        f"{summary['successful_tasks']}/{summary['total_tasks']} successful"
    )


if __name__ == "__main__":
    main()
