#!/usr/bin/env python3
"""
Prepose Sampler Completeness Benchmark

This script benchmarks the prepose sampling completeness by measuring
success rate as a function of the TOTAL number of samples.

Total samples = base_samples × ee_batch_size

This test evaluates the submodule's ability to find valid prepose configurations
when given sufficient sampling budget. Success rate should converge to near 100%
given enough samples, as this tests the completeness of the sampling approach.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tqdm
from scipy.spatial.transform import Rotation as R

# Set working directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Fix library path to use conda environment's libraries
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}"
        # Re-exec the script with updated LD_LIBRARY_PATH
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Direct imports to avoid triggering grasp_anywhere.__init__ which imports rospy
import importlib.util  # noqa: E402

# Import vamp for collision checking
import vamp  # noqa: E402


def import_module_directly(name, path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import dataclass modules directly
reachability_map_module = import_module_directly(
    "grasp_anywhere.dataclass.reachability_map",
    PROJECT_ROOT / "grasp_anywhere" / "dataclass" / "reachability_map.py",
)
CapabilityMap = reachability_map_module.CapabilityMap
ReachabilityMap = reachability_map_module.ReachabilityMap

torso_map_module = import_module_directly(
    "grasp_anywhere.dataclass.torso_map",
    PROJECT_ROOT / "grasp_anywhere" / "dataclass" / "torso_map.py",
)
TorsoMap = torso_map_module.TorsoMap

# Import base_sampler (needs logger)
logger_module = import_module_directly(
    "grasp_anywhere.utils.logger",
    PROJECT_ROOT / "grasp_anywhere" / "utils" / "logger.py",
)
sys.modules["grasp_anywhere.utils.logger"] = logger_module

base_sampler_module = import_module_directly(
    "grasp_anywhere.samplers.base_sampler",
    PROJECT_ROOT / "grasp_anywhere" / "samplers" / "base_sampler.py",
)
BaseSampler = base_sampler_module.BaseSampler

# Import reachability_utils
reachability_utils = import_module_directly(
    "grasp_anywhere.utils.reachability_utils",
    PROJECT_ROOT / "grasp_anywhere" / "utils" / "reachability_utils.py",
)

# Import torso_utils (needs TorsoMap in sys.modules)
sys.modules["grasp_anywhere.dataclass.torso_map"] = torso_map_module
torso_utils_module = import_module_directly(
    "grasp_anywhere.utils.torso_utils",
    PROJECT_ROOT / "grasp_anywhere" / "utils" / "torso_utils.py",
)
query_best_torso = torso_utils_module.query_best_torso

# Import IKFast (compiled module)
import ikfast_fetch  # noqa: E402

# Joint limits for Fetch robot
JOINT_LIMITS_LOWER = np.array(
    [0.0, -1.6056, -1.221, -np.pi, -2.251, -np.pi, -2.16, -np.pi]
)
JOINT_LIMITS_UPPER = np.array(
    [0.38615, 1.6056, 1.518, np.pi, 2.251, np.pi, 2.16, np.pi]
)


def is_ik_solution_valid(solution):
    """Check if a solution is within joint limits."""
    solution = np.array(solution)
    return np.all(solution >= JOINT_LIMITS_LOWER) and np.all(
        solution <= JOINT_LIMITS_UPPER
    )


def compute_ik(end_effector_pose, free_params: Optional[List[float]] = None):
    """Compute inverse kinematics solutions."""
    if free_params is None:
        torso_lift = np.random.uniform(JOINT_LIMITS_LOWER[0], JOINT_LIMITS_UPPER[0])
        shoulder_lift = np.random.uniform(JOINT_LIMITS_LOWER[2], JOINT_LIMITS_UPPER[2])
        free_params = [torso_lift, shoulder_lift]

    if isinstance(end_effector_pose, np.ndarray) and end_effector_pose.shape == (4, 4):
        position = end_effector_pose[:3, 3].tolist()
        rotation = end_effector_pose[:3, :3].tolist()
    else:
        position, rotation = end_effector_pose
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                rotation = rotation.tolist()
            else:
                rotation = np.array(rotation).reshape(3, 3).tolist()
        else:
            rotation = np.array(rotation).reshape(3, 3).tolist()

    solutions = ikfast_fetch.get_ik(rotation, position, free_params)
    if solutions:
        solutions = [sol for sol in solutions if is_ik_solution_valid(sol)]
    return solutions


def transform_pose_to_base(world_pos, world_quat, base_pos, base_yaw):
    """Transform a pose from world frame to base frame."""
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos_world = np.array(world_pos) - np.array(base_pos)
    pos_base = R_inv @ pos_world
    rot_world = R.from_quat(world_quat)
    rot_base_inv = R.from_euler("z", -base_yaw)
    rot_base = rot_base_inv * rot_world
    quat_base = rot_base.as_quat()
    return pos_base.tolist(), quat_base.tolist()


def quaternion_from_matrix(matrix):
    """Extract quaternion [x, y, z, w] from 4x4 transformation matrix."""
    rot = R.from_matrix(matrix[:3, :3])
    return rot.as_quat()


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""

    # Each sample is one (base, EE pose, torso) tuple
    min_total_samples: int = 1  # Start with 1 sample
    max_total_samples: int = 20480  # Test up to 20480 samples
    sample_increment: int = 1  # Increment total samples by this amount
    ee_batch_size: int = 128  # EE samples per base sample (matches main system)
    manipulation_radius: float = 0.8
    prepose_radius: float = 0.25
    seed: int = 42
    render_mode: Optional[str] = None  # None for headless, "human" for visualization
    num_workers: int = 16  # Number of parallel workers


class MinimalRobot:
    """Minimal robot interface for prepose sampling benchmark."""

    def __init__(
        self,
        capability_map_path: str = "resources/capability_map.pkl",
        reachability_map_path: str = "resources/reachability_map.pkl",
        torso_map_path: str = "resources/torso_map.pkl",
        costmap_path: str = "resources/costmap.npz",
    ):
        # print("Loading capability map...")
        self.capability_map = CapabilityMap.from_file(capability_map_path)

        # print("Loading reachability map...")
        self.reachability_map = ReachabilityMap.from_file(reachability_map_path)

        # print("Loading torso map...")
        self.torso_map = TorsoMap.from_file(torso_map_path)

        # print("Initializing base sampler...")
        # Dynamic costmap must be True to update from pointclouds
        self.base_sampler = BaseSampler(costmap_path=costmap_path, dynamic_costmap=True)

        # print("Initializing VAMP planner...")
        self._init_vamp()

        # print("Robot interface initialized.")

    def _init_vamp(self):
        """Initialize VAMP for collision checking."""
        self.planning_env = vamp.Environment()
        (
            self.vamp_module,
            self.planner_func,
            self.plan_settings,
            self.simp_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            "fetch", "rrtc", sampler_name="halton"
        )
        self.sampler = self.vamp_module.halton()
        self.sampler.skip(0)

    def add_pointcloud(self, points, point_radius=0.03):
        """Add a pointcloud to the VAMP environment and update base sampler costmap."""
        if points is None or len(points) == 0:
            return

        # Update base sampler costmap for 2D navigation checks
        # This is critical so we don't sample base poses inside tables/walls
        self.base_sampler.update_from_pointcloud(points)

        # Add to VAMP environment for 3D collision checks
        if isinstance(points, np.ndarray):
            points_list = points.tolist()
        else:
            points_list = points

        r_min, r_max = 0.03, 0.1  # Approx for fetch
        self.planning_env.add_pointcloud(points_list, r_min, r_max, point_radius)

    def clear_pointclouds(self):
        """Clear all pointclouds from VAMP environment and reset costmap."""
        self.planning_env.clear_pointclouds()
        # Since dynamic_costmap=True, the next update_from_pointcloud will replace it

    def sample_ik(self, position, orientation, base_config, torso_lift=None):
        """Solve IK for the given target pose."""
        base_pos = [base_config[0], base_config[1], 0.0]
        base_yaw = base_config[2]
        ee_pos_b, ee_quat_b = transform_pose_to_base(
            list(position), list(orientation), base_pos, base_yaw
        )
        rot_mat_b = R.from_quat(ee_quat_b).as_matrix()

        if torso_lift is None:
            return compute_ik([ee_pos_b, rot_mat_b])

        torso_lift = float(torso_lift)
        shoulder_lift = float(
            np.random.uniform(JOINT_LIMITS_LOWER[2], JOINT_LIMITS_UPPER[2])
        )
        return compute_ik(
            [ee_pos_b, rot_mat_b], free_params=[torso_lift, shoulder_lift]
        )

    def validate_whole_body_config(self, arm_config, base_config):
        """Validate collision. Returns True if IN COLLISION."""
        return not self.vamp_module.validate_whole_body_config(
            arm_config, base_config, self.planning_env
        )


# Global worker state (initialized once per worker process)
_worker_robot = None
_worker_benchmark = None
_worker_scene_cache = {}


def init_worker(robot_config):
    """Initialize worker process with robot instance (called once per worker)."""
    global _worker_robot, _worker_benchmark

    # Create robot instance once per worker
    _worker_robot = MinimalRobot(
        capability_map_path=robot_config["capability_map_path"],
        reachability_map_path=robot_config["reachability_map_path"],
        torso_map_path=robot_config["torso_map_path"],
        costmap_path=robot_config["costmap_path"],
    )

    # Create benchmark instance once per worker
    config = BenchmarkConfig()
    _worker_benchmark = PreposeSamplerBenchmark(_worker_robot, config)


def process_task_wrapper(args):
    """Wrapper to unpack arguments for parallel processing."""
    return process_task_parallel(*args)


def process_task_parallel(task_info, num_samples):
    """Process a single task in parallel (worker function)."""
    global _worker_robot, _worker_benchmark, _worker_scene_cache

    start_time = time.time()

    scene_pcd_path = task_info["scene_pcd_path"]
    obj_center = task_info["position"]
    scene_id = task_info["scene_id"]

    # Load scene pointcloud (cache within worker)
    if scene_pcd_path not in _worker_scene_cache:
        try:
            pcd = o3d.io.read_point_cloud(scene_pcd_path)
            _worker_scene_cache[scene_pcd_path] = np.asarray(
                pcd.points, dtype=np.float32
            )
        except Exception as e:
            return {"success": False, "stats": {}, "error": str(e), "time": 0.0}

    scene_points = _worker_scene_cache[scene_pcd_path]

    # Update robot environment (clear and reload)
    _worker_robot.clear_pointclouds()
    _worker_robot.add_pointcloud(scene_points)

    # Set stable seed for this task
    task_seed = hash((scene_id, tuple(obj_center))) % (2**32)
    np.random.seed(task_seed)

    # Run the test using pre-initialized benchmark
    success, stats = _worker_benchmark.sample_prepose_with_budget(
        obj_center, num_samples
    )

    elapsed_time = time.time() - start_time

    return {
        "success": success,
        "stats": stats,
        "time": elapsed_time,
        "scene_id": scene_id,
    }


class PreposeSamplerBenchmark:
    """Benchmark for evaluating prepose sampler completeness."""

    def __init__(self, robot: MinimalRobot, config: BenchmarkConfig):
        self.robot = robot
        self.config = config

    def _create_pose_matrix(self, position, approach_vector):
        """Creates a 4x4 pose matrix for the gripper."""
        x_axis = approach_vector
        z_axis_ref = np.array([0, 0, 1])
        y_axis = np.cross(z_axis_ref, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.cross(np.array([0, 1, 0]), x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        pose_matrix = np.eye(4)
        pose_matrix[:3, 0] = x_axis
        pose_matrix[:3, 1] = y_axis
        pose_matrix[:3, 2] = z_axis
        pose_matrix[:3, 3] = position
        return pose_matrix

    def sample_prepose_with_budget(
        self,
        object_center_world: np.ndarray,
        total_samples: int,
    ) -> Tuple[bool, dict]:
        """
        Try to find a valid prepose with a total sampling budget.

        Each sample is one (base, EE pose, torso) tuple.
        We sample base poses and test EE poses until budget exhausted.

        Returns:
            (success, stats) where stats contains detailed rejection counts
        """
        ee_batch_size = self.config.ee_batch_size
        manipulation_radius = self.config.manipulation_radius
        radius = self.config.prepose_radius

        stats = {
            "reject_base": 0,
            "reject_reachability": 0,
            "reject_capability": 0,
            "reject_ik": 0,
            "reject_collision": 0,
            "reject_occlusion": 0,
            "total_samples_tested": 0,
            "total_base_tested": 0,
            "total_ee_tested": 0,
        }

        # Keep sampling until we exhaust the budget
        samples_used = 0
        while samples_used < total_samples:
            # Sample base pose
            stats["total_base_tested"] += 1
            base_config = self.robot.base_sampler.sample_base_pose(
                object_center_world.tolist(), manipulation_radius=manipulation_radius
            )
            if base_config is None:
                stats["reject_base"] += 1
                samples_used += 1
                stats["total_samples_tested"] += 1
                continue

            # Check reachability
            base_reach = reachability_utils.query_reachability_score(
                self.robot.reachability_map,
                base_config,
                object_center_world.tolist(),
            )
            if base_reach <= 0.04:
                stats["reject_reachability"] += 1
                samples_used += 1
                stats["total_samples_tested"] += 1
                continue

            # Get torso height
            torso_lift = query_best_torso(
                self.robot.torso_map,
                np.asarray(object_center_world, dtype=np.float64),
                np.asarray(base_config, dtype=np.float64),
            )
            torso_lift = float(
                np.clip(
                    torso_lift,
                    float(JOINT_LIMITS_LOWER[0]),
                    float(JOINT_LIMITS_UPPER[0]),
                )
            )

            # Sample batch of EE poses (up to remaining budget)
            remaining_budget = total_samples - samples_used
            batch_size = min(ee_batch_size, remaining_budget)

            random_dirs = np.random.randn(batch_size, 3)
            norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            random_dirs = random_dirs / norms
            ee_positions = object_center_world - random_dirs * radius

            for ee_pos in ee_positions:
                stats["total_ee_tested"] += 1
                stats["total_samples_tested"] += 1
                samples_used += 1

                approach_vec = object_center_world - ee_pos
                approach_vec = approach_vec / np.linalg.norm(approach_vec)
                prepose_matrix = self._create_pose_matrix(ee_pos, approach_vec)
                pos = prepose_matrix[:3, 3].tolist()
                quat = quaternion_from_matrix(prepose_matrix)

                target_pose_7d = [
                    pos[0],
                    pos[1],
                    pos[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3],
                ]

                # Check capability score
                score = reachability_utils.query_capability_score(
                    self.robot.capability_map, base_config, target_pose_7d
                )
                if score <= 0.04:
                    stats["reject_capability"] += 1
                    continue

                # Solve IK
                ik_solutions = self.robot.sample_ik(
                    pos, quat, base_config, torso_lift=torso_lift
                )
                if ik_solutions is None or len(ik_solutions) == 0:
                    stats["reject_ik"] += 1
                    continue

                # Validate solutions (collision and occlusion check)
                found_valid = False
                for ik_sol in ik_solutions:
                    if not self.robot.validate_whole_body_config(ik_sol, base_config):
                        found_valid = True
                        break
                    else:
                        stats["reject_collision"] += 1

                if found_valid:
                    return True, stats

                if samples_used >= total_samples:
                    break

        return False, stats

    def load_benchmark_scenarios(self, scene_pointcloud_dir: str) -> List[dict]:
        """Load tasks with scene pointcloud paths from benchmark file."""
        benchmark_path = PROJECT_ROOT / "resources" / "grasp_benchmark.json"

        with open(benchmark_path, "r") as f:
            benchmark_data = json.load(f)

        tasks = []
        for scene_id, scene_data in benchmark_data.items():
            # Use precomputed scene pointcloud that includes all objects
            scene_pcd_path = os.path.join(scene_pointcloud_dir, f"{scene_id}.ply")

            if not os.path.exists(scene_pcd_path):
                # print(f"Warning: Scene pointcloud not found for {scene_id}: {scene_pcd_path}")
                # print(f"  Please run: python tools/build_scene_pointclouds.py")
                continue

            for task in scene_data.get("grasp_tasks", []):
                pos = task.get("position", [])
                if len(pos) == 3:
                    tasks.append(
                        {
                            "scene_id": scene_id,
                            "scene_pcd_path": scene_pcd_path,
                            "position": np.array(pos, dtype=np.float64),
                        }
                    )

        return tasks

    def test_single_scenario(self, task_data: dict, num_base_samples: int) -> bool:
        """Test a single scenario with given number of base samples."""
        scene_pcd_path = task_data["scene_pcd_path"]
        obj_center = task_data["position"]
        scene_id = task_data["scene_id"]

        # Load scene pointcloud if not already loaded
        if not hasattr(self, "_current_scene") or self._current_scene != scene_pcd_path:
            self.robot.clear_pointclouds()
            try:
                pcd = o3d.io.read_point_cloud(scene_pcd_path)
                scene_points = np.asarray(pcd.points, dtype=np.float32)
                self.robot.add_pointcloud(scene_points)
                self._current_scene = scene_pcd_path
            except Exception as e:
                print(f"Error loading scene pointcloud: {e}")
                return False

        # Set stable seed for this task
        task_seed = hash((scene_id, tuple(obj_center))) % (2**32)
        np.random.seed(task_seed)

        total_samples = num_base_samples * self.config.ee_batch_size
        success, _ = self.sample_prepose_with_budget(obj_center, total_samples)
        return success

    def run_benchmark(
        self, scene_pointcloud_dir: str, max_scenarios: int = None
    ) -> dict:
        """Run the complete benchmark with parallel processing."""
        all_tasks = self.load_benchmark_scenarios(scene_pointcloud_dir)

        if max_scenarios is not None:
            all_tasks = all_tasks[:max_scenarios]

        # Generate all sample counts to test (1 by 1)
        sample_counts = list(
            range(
                self.config.min_total_samples,
                self.config.max_total_samples + 1,
                self.config.sample_increment,
            )
        )

        # print(f"\nRunning benchmark with {len(all_tasks)} scenarios")
        # print(f"Testing {len(sample_counts)} sample counts: {sample_counts[0]} to {sample_counts[-1]}")
        # print(f"Using {self.config.num_workers} parallel workers")

        results = {
            "sample_counts": sample_counts,
            "ee_batch_size": self.config.ee_batch_size,
            "success_rates": {},
            "success_counts": {},
            "rejection_stats": {},  # Detailed rejection statistics
            "timing_stats": {},  # Timing statistics per sample count
            "total_scenarios": len(all_tasks),
        }

        # Prepare robot config for workers
        robot_config = {
            "capability_map_path": "resources/capability_map.pkl",
            "reachability_map_path": "resources/reachability_map.pkl",
            "torso_map_path": "resources/torso_map.pkl",
            "costmap_path": "resources/costmap.npz",
        }

        # Create worker pool with initialization (robot loaded once per worker)
        with mp.Pool(
            processes=self.config.num_workers,
            initializer=init_worker,
            initargs=(robot_config,),
        ) as pool:
            # Test each sample count incrementally with progress bar
            pbar = tqdm.tqdm(
                sample_counts, desc="Testing sample counts", unit="samples"
            )
            for num_samples in pbar:
                # Parallel process all tasks with inner progress bar
                args_list = [(task, num_samples) for task in all_tasks]
                results_list = list(
                    tqdm.tqdm(
                        pool.imap(process_task_wrapper, args_list, chunksize=1),
                        total=len(args_list),
                        desc=f"  N={num_samples}",
                        leave=False,
                    )
                )

                # Aggregate results
                successes = sum(1 for r in results_list if r["success"])
                success_rate = successes / len(all_tasks)

                # Aggregate rejection statistics
                reject_stats = {
                    "reject_base": 0,
                    "reject_reachability": 0,
                    "reject_capability": 0,
                    "reject_ik": 0,
                    "reject_collision": 0,
                    "reject_occlusion": 0,
                    "total_samples_tested": 0,
                    "total_base_tested": 0,
                    "total_ee_tested": 0,
                }
                for r in results_list:
                    if "stats" in r:
                        for key in reject_stats:
                            reject_stats[key] += r["stats"].get(key, 0)

                # Aggregate timing statistics
                times = [r["time"] for r in results_list if "time" in r]
                timing_stats = {
                    "mean_time_per_task": np.mean(times) if times else 0.0,
                }

                results["success_counts"][str(num_samples)] = successes
                results["success_rates"][str(num_samples)] = success_rate
                results["rejection_stats"][str(num_samples)] = reject_stats
                results["timing_stats"][str(num_samples)] = timing_stats

                # Update progress bar with current success rate
                pbar.set_postfix(
                    {
                        "success_rate": f"{success_rate*100:.1f}%",
                        "successes": f"{successes}/{len(all_tasks)}",
                    }
                )

                # Save intermediate results every 100 samples
                if num_samples % 100 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_dir = Path("results")
                    results_dir.mkdir(exist_ok=True)
                    intermediate_path = (
                        results_dir
                        / f"prepose_completeness_intermediate_{timestamp}_n{num_samples}.json"
                    )
                    with open(intermediate_path, "w") as f:
                        json.dump(results, f, indent=2)

        return results


def create_publication_plot(results: dict, output_path: str):
    """Create publication-quality plot showing completeness convergence."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "text.usetex": False,
        }
    )

    fig, ax = plt.subplots()

    total_samples = results["total_sample_counts"]
    success_rates = [results["success_rates"][str(n)] * 100 for n in total_samples]

    # Main line with markers
    ax.plot(
        total_samples,
        success_rates,
        "o-",
        color="#2171B5",
        linewidth=2.5,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.5,
        markeredgecolor="#2171B5",
        label="Prepose Sampler",
        zorder=5,
    )

    # Fill area under curve
    ax.fill_between(total_samples, 0, success_rates, alpha=0.15, color="#2171B5")

    # Labels
    ax.set_xlabel("Number of Samples (Base × EE × Torso)", fontweight="medium")
    ax.set_ylabel("Success Rate (%)", fontweight="medium")
    ax.set_title("Prepose Sampler Completeness", fontweight="bold")

    # Get the plateau value for annotation
    plateau_rate = max(success_rates)
    ax.axhline(y=plateau_rate, color="#666666", linestyle=":", alpha=0.5, linewidth=1.5)
    ax.text(
        max(total_samples) * 0.55,
        plateau_rate + 1.5,
        f"Converges to: {plateau_rate:.1f}%",
        fontsize=11,
        color="#666666",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # Reference lines for 90%, 95%, 100%
    for thresh, color, label in [
        (90, "#2ca02c", "90%"),
        (95, "#ff7f0e", "95%"),
        (100, "#d62728", "100%"),
    ]:
        ax.axhline(y=thresh, color=color, linestyle="--", alpha=0.4, linewidth=1.2)
        ax.text(
            total_samples[0], thresh + 1.5, label, fontsize=10, color=color, alpha=0.8
        )

    # Axis limits
    ax.set_xlim(0, max(total_samples) * 1.05)
    ax.set_ylim(0, 105)

    # Grid styling
    ax.grid(True, linestyle="-", alpha=0.2, which="major")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation about what this measures
    ax.text(
        0.98,
        0.02,
        f'Tested on {results["total_scenarios"]} scenarios',
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    plt.tight_layout(pad=1.0)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(
        output_path.replace(".pdf", ".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # print(f"\nPlot saved to: {output_path}")
    # print(f"          and: {output_path.replace('.pdf', '.png')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Prepose Sampler Completeness"
    )
    parser.add_argument(
        "-n", "--num-tasks", type=int, default=None, help="Number of tasks to test"
    )
    parser.add_argument(
        "--min-samples", type=int, default=1, help="Minimum number of samples to test"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20480,
        help="Maximum number of samples to test",
    )
    parser.add_argument(
        "--sample-increment",
        type=int,
        default=1,
        help="Increment samples by this amount",
    )
    parser.add_argument(
        "--num-workers", type=int, default=16, help="Number of parallel workers"
    )
    parser.add_argument(
        "--ee-batch",
        type=int,
        default=128,
        help="EE batch size per base sample (must match main system)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scene-pcd-dir",
        type=str,
        default="resources/benchmark/scene_pointclouds",
        help="Directory containing precomputed scene pointclouds",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Prepose Sampler Completeness Benchmark")
    print("=" * 80)
    print(f"Scenarios: {args.num_tasks if args.num_tasks else 'all (400)'}")
    print(
        f"Sample range: {args.min_samples} to {args.max_samples} (increment: {args.sample_increment})"
    )
    print(f"Workers: {args.num_workers}")
    print("=" * 80)

    # Check if scene pointclouds exist
    if not os.path.exists(args.scene_pcd_dir):
        print(f"\nERROR: Scene pointcloud directory not found: {args.scene_pcd_dir}")
        print("Please run: python tools/build_scene_pointclouds.py")
        return

    config = BenchmarkConfig(
        min_total_samples=args.min_samples,
        max_total_samples=args.max_samples,
        sample_increment=args.sample_increment,
        ee_batch_size=args.ee_batch,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create a dummy robot just for the benchmark coordinator (workers will have their own)
    robot = MinimalRobot()
    benchmark = PreposeSamplerBenchmark(robot, config)

    results = benchmark.run_benchmark(
        scene_pointcloud_dir=args.scene_pcd_dir, max_scenarios=args.num_tasks
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / f"prepose_completeness_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    # print(f"\nResults saved to: {results_path}")

    # Create plot
    plot_path = str(results_dir / f"prepose_completeness_{timestamp}.pdf")
    create_publication_plot(results, plot_path)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total scenarios tested: {results['total_scenarios']}")
    print(f"EE batch size: {results['ee_batch_size']}")
    print(
        f"Sample range: {args.min_samples} to {args.max_samples} (increment: {args.sample_increment})"
    )
    print("\nSuccess rate increases with more samples (showing every 100 samples):")
    for n in results["sample_counts"][::100]:
        if str(n) in results["success_rates"]:
            rate = results["success_rates"][str(n)] * 100
            count = results["success_counts"][str(n)]
            stats = results["rejection_stats"][str(n)]
            print(
                f"  {n:6d} samples: {rate:6.2f}% ({count:3d}/{results['total_scenarios']:3d}) | "
                f"Rejected - Base:{stats['reject_base']}, Reach:{stats['reject_reachability']}, "
                f"Cap:{stats['reject_capability']}, IK:{stats['reject_ik']}, "
                f"Coll:{stats['reject_collision']}, Occl:{stats['reject_occlusion']}"
            )
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
