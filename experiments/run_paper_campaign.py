#!/usr/bin/env python3
"""Run the paper's simulation experiments as a resumable repeated campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aggregate_paper_campaign import (
    aggregate_campaign,
    validate_result,
    write_outputs,
)


DEFAULT_BENCHMARK = PROJECT_ROOT / "resources" / "grasp_benchmark.json"
DEFAULT_DYNAMIC = (
    PROJECT_ROOT / "grasp_anywhere" / "configs" / "maniskill_fetch_dynamic_easy.yaml"
)
METHOD_CONFIGS = {
    "ours": "maniskill_fetch.yaml",
    "navigation_and_manipulation": "maniskill_fetch_baseline_nav_manip.yaml",
    "capmap_placement": "maniskill_fetch_nav_prepose.yaml",
    "direct_grasping": "maniskill_fetch_baseline_sequential_scheduler.yaml",
    "closed_loop_replanning": "maniskill_fetch_closed_loop.yaml",
    "velocity_agnostic": "maniskill_fetch_baseline_no_velocity.yaml",
}
CORE_METHOD_ORDER = (
    "ours",
    "navigation_and_manipulation",
    "capmap_placement",
    "direct_grasping",
    "closed_loop_replanning",
    "velocity_agnostic",
)
TRIGGER_DISTANCES = (0.5, 1.0, 1.5, 2.0, 2.5)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def pipeline_fingerprint() -> str:
    """Hash executable Python/YAML state so a resumed campaign cannot drift."""
    paths = [PROJECT_ROOT / "pixi.toml"]
    for root in (
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "grasp_anywhere",
    ):
        paths.extend(root.rglob("*.py"))
        paths.extend(root.rglob("*.yaml"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def config_without_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.pop("benchmark", None)
    return value


def prepare_configs(campaign_dir: Path, scope: str):
    config_root = PROJECT_ROOT / "grasp_anywhere" / "configs"
    dynamic_template = load_yaml(DEFAULT_DYNAMIC)
    dynamic_benchmark = copy.deepcopy(dynamic_template["benchmark"])
    output_dir = campaign_dir / "configs"
    configs: dict[tuple[str, str], Path] = {}

    for method in CORE_METHOD_ORDER:
        source = config_root / METHOD_CONFIGS[method]
        static_config = load_yaml(source)
        static_path = output_dir / f"{method}__static.yaml"
        write_yaml(static_path, static_config)
        configs[(method, "static")] = static_path

        dynamic_config = copy.deepcopy(static_config)
        dynamic_config["benchmark"] = copy.deepcopy(dynamic_benchmark)
        if config_without_benchmark(dynamic_config) != config_without_benchmark(
            static_config
        ):
            raise AssertionError(f"dynamic merge changed non-benchmark fields for {method}")
        dynamic_path = output_dir / f"{method}__dynamic.yaml"
        write_yaml(dynamic_path, dynamic_config)
        configs[(method, "dynamic")] = dynamic_path

    ours_dynamic = load_yaml(configs[("ours", "dynamic")])
    if ours_dynamic != dynamic_template:
        raise ValueError(
            "generated ours/dynamic config does not exactly match the new-pipeline template"
        )

    if scope in {"appendix", "all"}:
        ours_static = load_yaml(configs[("ours", "static")])
        for distance in TRIGGER_DISTANCES:
            trigger = copy.deepcopy(ours_static)
            trigger["benchmark"] = copy.deepcopy(dynamic_benchmark)
            trigger["benchmark"]["nav_trigger_distance"] = distance
            # The new pipeline places a route-intersecting obstacle at controlled
            # lookahead. These bounds make the paper's appearance-distance sweep
            # effective instead of relying on the legacy, unused trigger field.
            trigger["benchmark"]["nav_spawn_preferred_distance"] = distance
            trigger["benchmark"]["nav_spawn_distance_min"] = max(0.30, distance - 0.15)
            trigger["benchmark"]["nav_spawn_distance_max"] = distance + 0.15
            condition = f"trigger_{distance:.1f}"
            path = output_dir / f"ours__{condition}.yaml"
            write_yaml(path, trigger)
            configs[("ours", condition)] = path

        moving = copy.deepcopy(ours_static)
        moving["benchmark"] = copy.deepcopy(dynamic_benchmark)
        moving["benchmark"]["nav_obstacle_speed"] = 0.2
        moving_path = output_dir / "ours__moving_0.2.yaml"
        write_yaml(moving_path, moving)
        configs[("ours", "moving_0.2")] = moving_path
    return configs


def build_run_matrix(configs, repeats: int, scope: str, base_seed: int):
    pairs: list[tuple[str, str]] = []
    if scope in {"core", "all"}:
        # Ours runs first so threshold acceptance is known before spending days
        # evaluating baselines against an unacceptable pipeline revision.
        pairs.extend([("ours", "static"), ("ours", "dynamic")])
        for method in CORE_METHOD_ORDER[1:]:
            pairs.extend([(method, "static"), (method, "dynamic")])
    if scope in {"appendix", "all"}:
        pairs.extend(("ours", f"trigger_{distance:.1f}") for distance in TRIGGER_DISTANCES)
        pairs.append(("ours", "moving_0.2"))

    entries = []
    for method, condition in pairs:
        config_path = configs[(method, condition)]
        for repeat in range(1, repeats + 1):
            entries.append(
                {
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    "run_seed": base_seed + repeat - 1,
                    "config_path": config_path.relative_to(config_path.parents[1]).as_posix(),
                    "config_sha256": sha256_path(config_path),
                    "status": "pending",
                    "attempts": [],
                }
            )
    return entries


def service_healthy(url: str, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    health_url = url.rstrip("/") + "/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2.0)
    return False


def initialize_campaign(args, campaign_dir: Path):
    campaign_dir.mkdir(parents=True, exist_ok=False)
    configs = prepare_configs(campaign_dir, args.scope)
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_dir.name,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "scope": args.scope,
        "repeats": args.repeats,
        "expected_tasks_per_run": 400,
        "confidence": 0.95,
        "interval": "pooled Wilson score interval",
        "base_seed": args.base_seed,
        "benchmark_path": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_path(args.benchmark),
        "pipeline_fingerprint": pipeline_fingerprint(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "paper_commit": git_value(
            "-C",
            "/media/run/Work/paper/visibility_awared_mobile_grasping",
            "rev-parse",
            "HEAD",
        ),
        "gpus": args.gpus,
        "num_processes": args.num_processes,
        "save_trajectories": args.save_trajectories,
        "runs": build_run_matrix(configs, args.repeats, args.scope, args.base_seed),
    }
    atomic_json(campaign_dir / "manifest.json", manifest)
    return manifest


def load_campaign(args, campaign_dir: Path):
    manifest = json.loads((campaign_dir / "manifest.json").read_text())
    if manifest["pipeline_fingerprint"] != pipeline_fingerprint():
        raise RuntimeError(
            "pipeline source changed since campaign creation; refusing a mixed-revision resume"
        )
    if manifest["benchmark_sha256"] != sha256_path(args.benchmark):
        raise RuntimeError("benchmark dataset changed since campaign creation")
    for entry in manifest["runs"]:
        if entry["status"] == "running":
            entry["status"] = "interrupted"
    return manifest


def run_entry(args, campaign_dir: Path, manifest, entry):
    if not service_healthy(args.grasp_service_url):
        raise RuntimeError(
            f"Contact-GraspNet is not healthy at {args.grasp_service_url}/healthz"
        )
    attempt_number = len(entry["attempts"]) + 1
    run_root = (
        campaign_dir
        / "runs"
        / entry["method"]
        / entry["condition"]
        / f"repeat_{entry['repeat']:02d}"
        / f"attempt_{attempt_number:02d}"
    )
    log_path = (
        campaign_dir
        / "logs"
        / entry["method"]
        / entry["condition"]
        / f"repeat_{entry['repeat']:02d}__attempt_{attempt_number:02d}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = campaign_dir / entry["config_path"]
    label = (
        f"{manifest['campaign_id']}/{entry['method']}/{entry['condition']}/"
        f"repeat_{entry['repeat']:02d}"
    )
    command = [
        sys.executable,
        "experiments/run_maniskill_benchmark.py",
        "--config",
        str(config_path),
        "--benchmark",
        str(args.benchmark),
        "--gpus",
        args.gpus,
        "--parallel",
        "--num-processes",
        str(args.num_processes),
        "--run-seed",
        str(entry["run_seed"]),
        "--run-label",
        label,
        "--output-dir",
        str(run_root),
    ]
    if args.save_trajectories:
        command.append("--save-trajectory")

    attempt = {
        "attempt": attempt_number,
        "started_at": utc_now(),
        "command": command,
        "log_path": log_path.relative_to(campaign_dir).as_posix(),
        "output_dir": run_root.relative_to(campaign_dir).as_posix(),
        "status": "running",
    }
    entry["attempts"].append(attempt)
    entry["status"] = "running"
    manifest["status"] = "running"
    manifest["updated_at"] = utc_now()
    atomic_json(campaign_dir / "manifest.json", manifest)

    print(f"\n=== START {label} attempt {attempt_number} ===", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise

    attempt["finished_at"] = utc_now()
    attempt["duration_seconds"] = time.monotonic() - started
    attempt["return_code"] = return_code
    result_path = run_root / "benchmark_results.json"
    try:
        if return_code != 0:
            raise RuntimeError(f"benchmark exited with status {return_code}")
        result = json.loads(result_path.read_text())
        validate_result(result, manifest["expected_tasks_per_run"])
        if result["config"]["config_sha256"] != entry["config_sha256"]:
            raise RuntimeError("result config hash does not match campaign snapshot")
        if result["config"]["benchmark_sha256"] != manifest["benchmark_sha256"]:
            raise RuntimeError("result benchmark hash does not match campaign manifest")
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["error"] = str(exc)
        entry["status"] = "failed"
        print(f"=== FAIL {label}: {exc} ===", flush=True)
        return False

    attempt["status"] = "completed"
    entry["status"] = "completed"
    entry["result_path"] = result_path.relative_to(campaign_dir).as_posix()
    entry["successful_tasks"] = result["summary"]["successful_tasks"]
    entry["total_tasks"] = result["summary"]["total_tasks"]
    print(
        f"=== DONE {label}: {entry['successful_tasks']}/{entry['total_tasks']} ===",
        flush=True,
    )
    return True


def ours_thresholds_complete_and_passing(campaign_dir: Path):
    aggregate = aggregate_campaign(campaign_dir)
    write_outputs(campaign_dir, aggregate)
    static = aggregate["groups"].get("ours/static")
    dynamic = aggregate["groups"].get("ours/dynamic")
    complete = bool(static and dynamic and static["complete"] and dynamic["complete"])
    passing = bool(
        complete
        and static["run_mean_success_rate"] > 0.70
        and dynamic["run_mean_success_rate"] > 0.60
    )
    return complete, passing, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scope", choices=("core", "appendix", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-processes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026081300)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--grasp-service-url", default="http://localhost:4003")
    parser.add_argument(
        "--save-trajectories", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--continue-on-threshold-failure", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("confidence intervals require at least two independent runs")
    if args.num_processes < 1:
        parser.error("num-processes must be positive")
    args.benchmark = args.benchmark.resolve()

    if args.campaign_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_dir = PROJECT_ROOT / "results" / "paper_campaigns" / f"rerun_{timestamp}"
    else:
        campaign_dir = args.campaign_dir.resolve()

    if args.resume:
        manifest = load_campaign(args, campaign_dir)
    else:
        if campaign_dir.exists():
            parser.error(f"campaign directory already exists: {campaign_dir}")
        manifest = initialize_campaign(args, campaign_dir)
    print(f"Campaign: {campaign_dir}", flush=True)
    print(f"Runs: {len(manifest['runs'])} × 400 tasks", flush=True)

    for entry in manifest["runs"]:
        if entry["status"] == "completed":
            continue
        if len(entry["attempts"]) >= args.max_attempts:
            manifest["status"] = "failed"
            manifest["updated_at"] = utc_now()
            atomic_json(campaign_dir / "manifest.json", manifest)
            raise RuntimeError(
                f"attempt limit reached for {entry['method']}/{entry['condition']}/"
                f"repeat {entry['repeat']}"
            )

        if entry["method"] != "ours" and manifest["scope"] in {"core", "all"}:
            complete, passing, aggregate = ours_thresholds_complete_and_passing(
                campaign_dir
            )
            if complete and not passing and not args.continue_on_threshold_failure:
                manifest["status"] = "threshold_check_failed"
                manifest["updated_at"] = utc_now()
                atomic_json(campaign_dir / "manifest.json", manifest)
                static = aggregate["groups"]["ours/static"]["run_mean_success_rate"]
                dynamic = aggregate["groups"]["ours/dynamic"]["run_mean_success_rate"]
                raise RuntimeError(
                    "ours acceptance failed before baseline launch: "
                    f"static={static:.2%}, dynamic={dynamic:.2%}"
                )

        completed = run_entry(args, campaign_dir, manifest, entry)
        manifest["updated_at"] = utc_now()
        atomic_json(campaign_dir / "manifest.json", manifest)
        if not completed:
            # Retry the same entry on the next process invocation; this keeps
            # failure artifacts intact and avoids silently mixing partial runs.
            manifest["status"] = "failed"
            atomic_json(campaign_dir / "manifest.json", manifest)
            raise RuntimeError(
                f"run failed; resume campaign to retry: {campaign_dir}"
            )

    aggregate = aggregate_campaign(campaign_dir)
    write_outputs(campaign_dir, aggregate)
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    manifest["acceptance"] = aggregate["acceptance"]
    atomic_json(campaign_dir / "manifest.json", manifest)
    print((campaign_dir / "aggregate" / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
