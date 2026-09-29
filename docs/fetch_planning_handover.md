# Handover: fetch-planning (FLASK) as the motion planner

Status as of 2026-09-29. Goal: replace the vendored VAMP planner with the
[fetch-planning](https://github.com/H-tr/Fetch-Planning) package (PyPI
`fetch-planning==0.3.0`, kinodynamic planner FLASK) and check on the 400-task
benchmark (`resources/grasp_benchmark.json`) whether it is faster and has
fewer collision / planning failures.

**The full 400-task comparison has not been run yet** — it was stopped to
move to the server. Everything below the "Results so far" heading is partial.

## What changed

| File | Change |
|---|---|
| `pixi.toml`, `pixi.lock` | `fetch-planning = "==0.3.0"` (prebuilt cp311 wheel; pulls `pin`, `pin-pink`, cmeel libs; numpy stays 1.26.4). |
| `grasp_anywhere/robot/utils/vamp_backend.py` | New. The previous VAMP code from `fetch.py`, moved unchanged. |
| `grasp_anywhere/robot/utils/fetch_planning_backend.py` | New. Same interface, everything through fetch_planning. |
| `grasp_anywhere/robot/fetch.py` | Delegates planning, validation, the replanning monitor's path check, robot self-filtering, EE FK, obstacles and attachments to `self.planner_backend`. Records every planner call in `self.planning_log`. |
| `grasp_anywhere/stage_planners/grasp_stage.py` | Two direct VAMP calls now go through `Fetch` (`validate_whole_body_config`, `eefk`); checked equivalent on 20k configs. |
| `experiments/run_maniskill_benchmark.py` | Stores `motion_planner` and `planning_calls` (per call: kind, success, wall time, planner stats) in every task result. |
| `grasp_anywhere/configs/maniskill_fetch_fetch_planning.yaml` | `maniskill_fetch.yaml` + `motion_planner: fetch_planning`, `fetch_planning.velocity_scale: 2.0`. |
| `experiments/planner_ab/` | A/B tooling: query-dumping benchmark wrapper, identical-query replay, run comparison. |
| `tests/test_fetch_planning_backend.py` | Backend tests (path check between waypoints, planning around a wall forward-only, self-filter, arm planning). |

Switch with `planning.motion_planner` in the YAML: `vamp` (default, unchanged
behaviour) or `fetch_planning`. In `fetch_planning` mode `vamp` is never
imported. Optional `planning.fetch_planning:` holds `KinodynamicConfig`
overrides (`time_limit`, `velocity_scale`, `acceleration_scale`,
`allow_reverse`, ...).

### fetch_planning backend details

- **Planner**: `MotionPlanner.plan_kinodynamic` (FLASK) only — whole body
  (`fetch_whole_body`), arm (`fetch_arm_with_torso`, base pinned at the current
  pose) and base-only (`fetch_base`). QRRT / geometric `plan()` is not used.
- **No reversing**: `allow_reverse=False` by default (the head camera faces
  forward). Rotate-in-place is allowed.
- **Base sampling bounds** (important, see results): FLASK samples base x, y
  uniformly inside the planner bounds, which default to ±10 m. The backend
  first tries a box around start and goal (+2 m, clipped to the obstacle
  cloud's bounding box + 0.5 m); on a search failure it retries once over the
  whole cloud bounding box. `time_limit` (default 1 s) applies per attempt.
- **Output**: the trajectory is sampled at 5 ms and resampled to the waypoint
  spacing the VAMP planners produced (0.03 of `|dxy| + 0.3|dθ| + 0.2|darm|`
  whole-body, 1/16 rad arm). The ManiSkill executor is a path follower
  (nearest waypoint + look-ahead); it ignores FLASK's timing.
- **Start/goal** are clamped into joint bounds (sim readings and the 3-decimal
  rounding in `plan_whole_body_motion` can sit just outside a limit, and FLASK
  rejects edges that leave the bounds).
- **Replanning monitor** (`path_in_collision`): densifies the straight edges
  between the remaining waypoints and calls `validate_batch`, like VAMP's
  `check_whole_body_collisions`.
- **Self-filtering** uses a second, obstacle-free planner:
  `filter_self_from_pointcloud` also drops points touching registered
  obstacles, which would erase mapped geometry from every new observation.
- **Deterministic**: per-query FLASK seed from a CRC of start/goal and a query
  counter; the global numpy RNG (seeded per task by the benchmark) is not used.

### Known gaps

- fetch-planning 0.3.0 has **no Python API for attached objects** (the C++
  side already has `env_.attachments`). `attach_objects_to_eef` returns False
  with a warning. The benchmark pipeline never plans with an object attached
  (attach is the last step of the grasp stage; `PlacePlanner` is constructed
  but never called), so results are unaffected.
- No sphere-only clear in fetch-planning: `clear_spheres` rebuilds the
  environment from the tracked point clouds.
- `Fetch.vamp_module` / `Fetch.planning_env` exist only in VAMP mode;
  `examples/visualize_*.py` and `experiments/benchmark_prepose_completeness.py`
  still use VAMP directly.
- The point-cloud structure for the full radius range costs ~136 ms per scene
  refresh (52k points) vs ~16 ms for VAMP's (too small, see below) range, so
  the replanning monitor loop runs a bit slower.
- `pytest` is not in the pixi env; the tests were run by calling the test
  functions directly.

## Results so far

### Collision models are identical — but VAMP's point-cloud radius range is wrong

VAMP builds every point-cloud collision structure with
`vamp.ROBOT_RADII_RANGES["fetch"] = (0.012, 0.055)`, but Fetch's whole-body
model (111 spheres, identical in both packages) goes up to **0.24 m** at the
base. VAMP therefore misses base collisions:

- 20k random configs on the scene_0 cloud: 298 "free" in VAMP but colliding in
  fetch_planning, never the reverse. With `r_max = 0.24` in VAMP: 20000/20000
  agree. Self-collision: 100 % agree.
- On 139 real benchmark queries that FLASK had failed, VAMP's (current
  pipeline) path collides under the full model in **75**.

The VAMP backend is intentionally left unchanged so the baseline matches the
earlier campaign. Fix for VAMP: pass `min/max` of `[s.r for s in
vamp_module.fk(q)]` instead of `ROBOT_RADII_RANGES`.

### Planner speed

- Synthetic (30 random whole-body queries on scene_0): same 25/30 solved by
  both; median wall time **29 ms** FLASK vs 63 ms VAMP (142 ms VAMP with the
  correct radius range).
- Live benchmark, whole-body calls (partial runs): median successful plan
  **58 ms** FLASK vs 92 ms VAMP; failed attempts give up after 1 s (FLASK) vs
  ~2 s (VAMP, iteration limit). Wall time covers planning + simplification +
  resampling / interpolation.

### Base bounds were the main problem (fixed)

First fetch_planning run used the default ±10 m bounds: FLASK failed **31 %**
of whole-body queries with valid start/goal (VAMP: 13 %). The failed queries
are feasible — VAMP with the correct radius range solved 131/139 of them.
Replaying real queries at 1 s:

| Base bounds | Solved |
|---|---|
| ±10 m (default) | 75/217 previously-failed queries |
| cloud bounding box + 0.5 m | 130/217 |
| start/goal box + 2 m | 176/217 |
| **current (local, then scene)**, 250 random valid queries | **237/250 (95 %)** vs 174/250 with ±10 m; median 52 ms |

### Speed limits (preliminary)

`velocity_scale` / `acceleration_scale` sweep with the bounds fix, stopped
after 51 previously-failed queries: v×1 a×1 → 42 solved, **v×2 a×1 → 46**,
v×2 a×2 → 36, v×3 a×3 → 34. Hence `velocity_scale: 2.0` in the new config;
confirm on more queries. The torso limit (0.1 m/s) is a suspect: a full
0.386 m stroke takes ~4 s, longer than FLASK's `max_extension_time` (3 s).

### Benchmark (partial, run seed 2026081300, 3 workers each)

| Run | Tasks | Success | Collision | Planning failure |
|---|---|---|---|---|
| VAMP (current pipeline) | 180 | 119 (66 %) | 15 | 18 |
| fetch_planning, ±10 m bounds | 60 | 36 (60 %) | 1 | 15 |
| same 60 tasks, VAMP | 60 | 39 (65 %) | 7 | 5 |
| fetch_planning, bounds fix | 12 | 7 | — | — |

Reading: collisions drop sharply with fetch_planning; the extra planning
failures in the first run came from the bounds issue that is now fixed. The
fixed configuration still needs the full 400-task run.

Local artifacts on the workstation (not in git; `results/` is ignored):
`results/planner_ab_20260929/` (configs, logs, partial `benchmark_results.json`
with `planning_calls`, the exact backend files each run used) and
`results/planner_ab_20260929/queries/` (~2 GB of dumped planner queries per
run, replayable with `replay_queries.py`; `B_fetch_planning_default_bounds`
holds the failed FLASK queries used above).

## Running on the server

1. `pixi install` (fetch-planning wheel needs glibc ≥ 2.28), then
   `bash scripts/download_resources.sh` and `pixi run download-assets`.
2. **Submodule**: the workstation's `third_party/perception_services` is at
   commit `8eba545` ("fix: repair Contact-GraspNet runtime"), which is **not
   pushed** to its remote (`threefruits/perception_services`, at `647d474`).
   This branch keeps the old pointer. Push that commit (or a fork) and bump the
   pointer, or re-apply the Contact-GraspNet fixes on the server
   (`service/setup_all.sh`).
3. Grasp service: `cd third_party/perception_services && pixi run -e grasp grasp-server`
   (port 4003, `GET /healthz`).
4. GPU memory on the workstation: ~1.6 GB per sim worker; the grasp server grew
   to ~7.4 GB. Five workers + server ran out of memory in the August campaign.

Benchmark runs (same seed for both):

```bash
pixi run python experiments/run_maniskill_benchmark.py \
  --config grasp_anywhere/configs/maniskill_fetch.yaml \
  --benchmark resources/grasp_benchmark.json --parallel --num-processes 3 --gpus 0 \
  --run-seed 2026081300 --output-dir results/planner_ab/A_vamp --save-trajectory

pixi run python experiments/run_maniskill_benchmark.py \
  --config grasp_anywhere/configs/maniskill_fetch_fetch_planning.yaml \
  --benchmark resources/grasp_benchmark.json --parallel --num-processes 3 --gpus 0 \
  --run-seed 2026081300 --output-dir results/planner_ab/B_fetch_planning --save-trajectory
```

To also dump every planning query (start, goal, obstacle clouds, result) for
offline replay, run the same arguments through the wrapper:

```bash
GA_PLAN_QUERY_DIR=results/planner_ab/B_queries \
  pixi run python experiments/planner_ab/run_benchmark_with_query_dump.py <same args>
```

Compare:

```bash
# task success, failure reasons, paired McNemar test, planner-call statistics
pixi run python experiments/planner_ab/analyze_runs.py \
  A=results/planner_ab/A_vamp B=results/planner_ab/B_fetch_planning

# identical queries through both backends; re-checks every returned path
# with the full-radius model
pixi run python experiments/planner_ab/replay_queries.py replay.json \
  results/planner_ab/A_queries results/planner_ab/B_queries
```

`experiments/run_paper_campaign.py` maps method names to configs in
`METHOD_CONFIGS`; add an entry for `maniskill_fetch_fetch_planning.yaml` to run
it as a campaign method. (Its `paper_commit` provenance field reads a
workstation path and will be empty on the server.)

## Next steps

1. Full 400-task A/B with the fixed configuration (above).
2. Confirm `velocity_scale` on more replayed queries; try raising only the
   torso limit.
3. Optional third arm: VAMP with the correct radius range, to separate the
   collision-model effect from the planner effect.
4. Remaining FLASK failures are mostly whole-body moves with the arm extended
   next to a table and the base near furniture — a useful regression set for
   Fetch-Planning.
5. Add an attach/detach binding to fetch-planning before using the place stage.
