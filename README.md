# Visibility-Awared Mobile Grasping in Dynamic Environments

### 📰 News — 🎉 Our live demo is now on [RoboMesh](https://robomesh.ssilabs.org/webapp/)!

Drive the mobile-grasping robot **right in your browser** — click an object, say _"grasp it,"_ and watch it navigate, perceive, and pick in real time. No install required. **→ [Try it online](https://robomesh.ssilabs.org/webapp/)**

> **Paper:** *Visibility-Aware Mobile Grasping in Dynamic Environments*
>
> This repository contains the code for our visibility-aware mobile grasping framework that enables a Fetch robot to grasp objects in cluttered, dynamic environments by jointly reasoning about base placement, arm planning, and active perception.

The public project/environment name is `mobile_grasping_in_dynamic`. The Python
import package is currently `grasp_anywhere`; keep that import path when running
the scripts in this repository.

## Prerequisites

- **OS:** Ubuntu 20.04
- **ROS:** Noetic (for Gazebo / real robot). The Noetic python bindings are also
  provided by the pixi environment for the simulation path (the robot code imports
  them at module load).
- **CUDA:** 12.4
- **Python:** 3.11 (provided by the pixi environment)
- **[pixi](https://pixi.sh):** the only tool you need — it creates the environment
  and compiles the native modules. (The legacy `env.yml` Conda flow is kept for
  reference but is no longer required.)
- **VAMP:** the motion planner source is vendored under `third_party/vamp` and is
  built automatically by `pixi install`.
- **TRAC-IK:** no ROS `trac_ik_python` needed — a self-contained pybind11 build is
  vendored under `third_party/pytracik` and compiled by `pixi install`.

ROS-side dependencies used by the real-robot code include `rospy`, `tf`,
`tf2_ros`, `cv_bridge`, `actionlib`, `sensor_msgs`, `geometry_msgs`,
`trajectory_msgs`, `move_base_msgs`, `control_msgs`, `nav_msgs`, and `std_msgs`.
Source the ROS workspace before running ROS/Gazebo or real-robot scripts.

## Installation

The environment is managed entirely by [pixi](https://pixi.sh). A single
`pixi install` creates the environment, installs the Python/ML stack and the ROS
Noetic bindings, and compiles the three native modules — `pytracik`
(third_party/pytracik), `vamp` (third_party/vamp), and `ikfast_fetch`. The two
CMake modules are built by scikit-build-core, which pulls `cmake`+`ninja` from
PyPI; their C/C++ dependencies (orocos-kdl, nlopt, eigen) come from the pixi
environment.

### 1. Create the environment (installs deps + builds the native modules)

```bash
pixi install
```

### 2. Download pre-computed resources

Large resource files (capability map, reachability map, etc.) are not stored in the
repository:

```bash
bash scripts/download_resources.sh
```

The downloader uses this Dropbox folder by default:

```text
https://www.dropbox.com/scl/fo/9gxri23a1fn4lmudmhat0/AJ3HfBHsj3XLkFANqXZsbb8?rlkey=5co0nluo78otf3103w5g9vwx9&st=yflfjy6u&dl=1
```

To mirror the resources elsewhere, set `MOBILE_GRASPING_RESOURCE_URL` to a
compatible archive download URL.

### 3. Download ManiSkill assets (for simulation)

```bash
pixi run download-assets
```

### 4. Validate the environment

```bash
pixi run smoke     # imports pytracik/vamp/ikfast, solves TRAC-IK, builds a ManiSkill scene
```

Run any command inside the environment with `pixi run <cmd>`, or open a shell with
`pixi shell`. (The legacy Conda flow — `conda env create -f env.yml` + `pip install -e .`
+ a manual IKFast build — is still described by `env.yml` but is no longer required.)

## Third-Party Planner

VAMP is required for whole-body motion planning and is vendored in this
repository under `third_party/vamp`. Build or install that local copy following
its README, and ensure the Python module is importable as `vamp` in the active
environment.

We thank the original VAMP project and its authors for the motion planning
library used by this codebase.

## External Services

The grasp generation client expects a local HTTP service at
`http://localhost:4003`. It must expose `/sample_grasp`.

Perception services (GraspNet, OWL-ViT, SAM) are provided as a git submodule
under [`third_party/perception_services`](third_party/perception_services/README.md).
See its README for setup and launch instructions.

| Service   | Default URL            | Description                    |
|-----------|------------------------|--------------------------------|
| GraspNet  | `http://localhost:4003` | 6-DOF grasp pose prediction   |
| OWL-ViT   | `http://localhost:4000` | Open-vocabulary object detection |
| SAM       | `http://localhost:4001` | Segment Anything mask prediction |

Service URLs are configured in the YAML config files under the `services` section:

```yaml
services:
  graspnet_url: "http://localhost:4003"
  owl_url: "http://localhost:4000"
  sam_url: "http://localhost:4001"
```

## Gazebo / ROS Setup

For the Gazebo digital twin environment, set up the [rls-digital-twin](https://github.com/AdaCompNUS/rls-digital-twin) project following its [installation guide](https://github.com/AdaCompNUS/rls-digital-twin/blob/main/INSTALL.md).

Then, in separate terminals (source your ROS workspace first):

```bash
# Terminal 1: Launch simulation
roslaunch low_level_planning rls_env.launch

# Terminal 2: Start whole-body controller
roslaunch fetch_drivers whole_body_controller.launch controller_type:=mpc
```

## Quick Start

### ManiSkill (Simulation)

```bash
pixi run run-benchmark
# equivalently:
pixi run python experiments/run_maniskill_benchmark.py \
    --config grasp_anywhere/configs/maniskill_fetch.yaml \
    --benchmark resources/grasp_benchmark.json
```

### Gazebo (ROS)

After launching the simulation and controller (see above):

```bash
python examples/grasp_anywhere_real_demo.py
```

### Real Robot

```bash
python experiments/run_real_robot.py \
    --config grasp_anywhere/configs/real_fetch.yaml
```

## Benchmark

### Run the Benchmark

Single run with GPU selection and optional trajectory saving:

```bash
python experiments/run_maniskill_benchmark.py \
    --config grasp_anywhere/configs/maniskill_fetch.yaml \
    --benchmark resources/grasp_benchmark.json \
    -g 0,1 -p -t
```

Run a specific scheduler via the helper script:

```bash
# Usage: ./scripts/run_benchmark.sh <scheduler_type> <gpus> [extra_args]
./scripts/run_benchmark.sh nav_manip 0,1 "-p -t"
./scripts/run_benchmark.sh closed_loop 2,3 "-p -t"
```

### Parallel / Continuous Benchmarking

Automatically allocate GPUs and run multiple experiments in parallel:

```bash
./scripts/run_continuous_benchmark.sh
```

### Trigger Distance Experiment

Sweep dynamic obstacle trigger distances (0.5 m -- 3.0 m) with repeated trials:

```bash
# Uses GPUs 0,1,2,3 by default
./scripts/run_trigger_distance_experiment.sh

# Or specify GPUs
./scripts/run_trigger_distance_experiment.sh 2,3,4,5
```

## Configuration

All configuration is done through YAML files in `grasp_anywhere/configs/`:

| Config | Description |
|--------|-------------|
| `maniskill_fetch.yaml` | ManiSkill simulation — our method (default scheduler) |
| `real_fetch.yaml` | Real Fetch robot |
| `maniskill_fetch_closed_loop.yaml` | Closed-loop baseline (prepose-grasp loop only) |
| `maniskill_fetch_nav_prepose.yaml` | Nav-Prepose baseline (decoupled nav + prepose sampler) |
| `maniskill_fetch_baseline_sequential_scheduler.yaml` | Sequential baseline |
| `maniskill_fetch_baseline_nav_manip.yaml` | Nav-Manip baseline |
| `maniskill_fetch_baseline_no_velocity.yaml` | No-velocity-awareness ablation |
| `maniskill_fetch_trigger_*.yaml` | Trigger distance sweep (0.5 m -- 3.0 m) |

Key configuration sections:

- **`planning`** — scheduler type, manipulation radius, replanning, ICP refinement, map paths
- **`services`** — GraspNet / OWL / SAM server URLs
- **`gaze`** — gaze optimizer parameters (lookahead, decay, joint priorities)
- **`monitor`** — contact force threshold, hold duration, slip tolerance
- **`benchmark`** — dynamic challenge flags, trigger distance
- **`debug`** — visualization and debug flags

## Project Structure

```
grasp_anywhere/
├── grasp_anywhere/          # Main package
│   ├── benchmark/           # Benchmark runners and critics
│   ├── checker/             # Occlusion and collision checkers
│   ├── configs/             # YAML configuration files
│   ├── core/                # Schedulers (default, sequential, nav-manip, nav-prepose, closed-loop)
│   ├── data_collector/      # Trajectory and visualization data collection
│   ├── dataclass/           # Data structures (reachability maps, configs)
│   ├── envs/                # Environment wrappers (ManiSkill, Gazebo, real)
│   ├── grasping_client/     # GraspNet, OWL-ViT, SAM client interfaces
│   ├── observation/         # Scene maintenance, gaze optimizer
│   ├── planning/            # Motion planning (VAMP integration)
│   ├── robot/               # Fetch robot interface, IK solvers
│   ├── samplers/            # Pre-pose and base samplers
│   ├── stage_planners/      # Stage planners (grasp, prepose, place, etc.)
│   └── utils/               # Utilities (perception, visualization, logging)
├── examples/                # Example scripts for each environment
├── experiments/             # Benchmark and evaluation scripts
├── tools/                   # Offline tools (visualization, map building)
├── resources/               # Robot URDF, collision models, config maps
├── third_party/             # Vendored third-party source dependencies
└── scripts/                 # Setup and benchmark runner scripts
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2025visibility,
  title={Visibility-Aware Mobile Grasping in Dynamic Environments},
  author={Hu, Tianrun and Xiao, Anxing and Hsu, David and Zhang, Hanbo},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
