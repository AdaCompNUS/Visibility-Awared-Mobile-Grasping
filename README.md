# Visibility-Awared Mobile Grasping in Dynamic Environments

> **Paper:** *Visibility-Aware Mobile Grasping in Dynamic Environments*
>
> This repository contains the code for our visibility-aware mobile grasping framework that enables a Fetch robot to grasp objects in cluttered, dynamic environments by jointly reasoning about base placement, arm planning, and active perception.

The public project/environment name is `mobile_grasping_in_dynamic`. The Python
import package is currently `grasp_anywhere`; keep that import path when running
the scripts in this repository.

## Prerequisites

- **OS:** Ubuntu 20.04
- **ROS:** Noetic (for Gazebo / real robot)
- **CUDA:** 12.4
- **Python:** 3.9
- **Conda** (Miniconda or Anaconda)
- **VAMP:** the motion planner source is vendored under `third_party/vamp`.
- **TRAC-IK:** `trac_ik_python` is a ROS dependency. Install it from ROS Noetic
  packages or build/source a TRAC-IK ROS workspace before running planners that
  use `grasp_anywhere.robot.ik.trac_ik_solver`.

ROS-side dependencies used by the real-robot code include `rospy`, `tf`,
`tf2_ros`, `cv_bridge`, `actionlib`, `sensor_msgs`, `geometry_msgs`,
`trajectory_msgs`, `move_base_msgs`, `control_msgs`, `nav_msgs`, and `std_msgs`.
Source the ROS workspace before running ROS/Gazebo or real-robot scripts.

## Installation

### 1. Create the Conda Environment

```bash
conda env create -f env.yml
conda activate mobile_grasping_in_dynamic
```

### 2. Install the Package

```bash
pip install -e .
```

### 3. Build IKFast (required for arm motion planning)

```bash
cd grasp_anywhere/robot/ik/ikfast/fetch
python setup.py build_ext --inplace
cd -
```

### 4. Download Pre-computed Resources

Large resource files (capability map, reachability map, etc.) are not stored in the repository. Download them with:

```bash
bash scripts/download_resources.sh
```

The downloader uses this Dropbox folder by default:

```text
https://www.dropbox.com/scl/fo/9gxri23a1fn4lmudmhat0/AJ3HfBHsj3XLkFANqXZsbb8?rlkey=5co0nluo78otf3103w5g9vwx9&st=yflfjy6u&dl=1
```

To mirror the resources elsewhere, set `MOBILE_GRASPING_RESOURCE_URL` to a
compatible archive download URL.

### 5. Download ManiSkill Assets (for simulation)

```bash
python -m mani_skill.utils.download_asset ReplicaCAD
python -m mani_skill.utils.download_asset ycb
```

## Third-Party Planner

VAMP is required for whole-body motion planning and is vendored in this
repository under `third_party/vamp`. Build or install that local copy following
its README, and ensure the Python module is importable as `vamp` in the active
environment.

We thank the original VAMP project and its authors for the motion planning
library used by this codebase.

## External Services

The grasp generation client expects a local HTTP service at
`http://localhost:4003`. It must expose `/sample_grasp` and
`/sample_grasp_with_context`.

Perception services for open-vocabulary detection and segmentation are not
included in this release yet. They will be wrapped and released separately. Code
paths that use OWL-ViT/SAM still expect compatible local services.

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
python experiments/run_maniskill_benchmark.py \
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

### Generate a Benchmark

```bash
python tools/generate_grasp_benchmark.py \
    --generate \
    --output_path resources/grasp_benchmark.json
```

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

### Visualize Results

```bash
# Benchmark scene visualization
python tools/visualize_benchmark.py

# Result distribution plots
python tools/visualize_benchmark_distribution.py results/benchmark_results.json
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
