#!/usr/bin/env python3

import os
import glob
import time
from pathlib import Path

import numpy as np
import pybullet as pb
from vamp import pybullet_interface as vpb
import fire
import vamp

try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available; pointcloud loading will fail.")


def load_pointcloud(file_path, voxel_size=None):
    if o3d is None:
        raise RuntimeError("open3d is required to load pointclouds")
    pcd = o3d.io.read_point_cloud(file_path)
    
    if voxel_size is not None and voxel_size > 0:
        # Downsample using voxel grid
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return np.asarray(pcd.points)


def load_all_pointclouds(directory, voxel_size=None):
    ply_files = glob.glob(os.path.join(directory, "*.ply"))
    pcd_files = glob.glob(os.path.join(directory, "*.pcd"))
    files = ply_files + pcd_files
    if not files:
        raise FileNotFoundError(f"No pointclouds (*.ply|*.pcd) found in {directory}")

    pcs = []
    total_points = 0
    for f in files:
        try:
            pts = load_pointcloud(f, voxel_size)
            if pts is not None and len(pts) > 0:
                total_points += len(pts)
                pcs.append(pts)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if not pcs:
        raise RuntimeError("No valid point clouds loaded")

    combined = np.vstack(pcs)
    if voxel_size is not None and voxel_size > 0:
        print(f"Downsampled from {total_points} to {len(combined)} points (voxel_size={voxel_size})")
    
    return combined


def filter_points_by_radius(points, max_radius):
    d = np.linalg.norm(points, axis=1)
    return points[d <= max_radius]


def compute_bounds_xy(points, padding=0.5):
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    x_min = float(min_xy[0] - padding)
    x_max = float(max_xy[0] + padding)
    y_min = float(min_xy[1] - padding)
    y_max = float(max_xy[1] + padding)
    return x_min, x_max, y_min, y_max


def sample_points_for_visualization(points: np.ndarray, max_points: int = 75000):
    if len(points) <= max_points:
        return points.tolist()
    indices = np.random.choice(len(points), size=max_points, replace=False)
    return points[indices].tolist()


def main(
    pc_directory: str = "mp_collision_models",
    point_radius: float = 0.1,
    max_radius: float = 20.0,
    padding: float = 0.5,
    visualize: bool = True,
    interp_density: float = 0.03,
    voxel_size: float = 0.05,  # Voxel size for downsampling point clouds
):
    # Robot module
    robot = vamp.fetch
    
        #        INFO     Start arm config: [0.3, -0.16, 0.016, 0.77, 0.995, -0.71, 0.259, 0.475]                                                                                           fetch.py:683
        #    INFO     Goal arm config: [0.291, -1.158, 0.162, 1.046, 2.071, -2.415, 0.402, 0.541]                                                                                       fetch.py:684
        #    INFO     Start base config: [2.419, -3.295, -0.189]                                                                                                                        fetch.py:685
        #    INFO     Goal base config: [2.91, -2.46, 2.892] 

    # Example start/goal (from test_whole_body.py defaults)
    start_base = [2.419, -3.295, -0.189]  # x, y, theta
    goal_base = [2.91, -2.46, 2.892]  # x, y, theta
    # goal_base = [-3.119, 0.712, 1.57]  # x, y, theta

    # Define start and goal arm configurations (from test_base_transform.py)
    start_joints = [0.3, -0.16, 0.016, 0.77, 0.995, -0.71, 0.259, 0.475]
    goal_joints = [0.291, -1.158, 0.162, 1.046, 2.071, -2.415, 0.402, 0.541]

    # Build environment
    env = vamp.Environment()

    # Load and prepare pointclouds
    print(f"Loading pointclouds from: {pc_directory}")
    raw_pc = load_all_pointclouds(pc_directory, voxel_size)
    print(f"Loaded {len(raw_pc)} points.")

    pc = filter_points_by_radius(raw_pc, max_radius)
    print(f"Filtered to {len(pc)} points with radius <= {max_radius}")

    x_min, x_max, y_min, y_max = compute_bounds_xy(pc, padding)
    print(
        f"Computed bounds: x[{x_min:.2f}, {x_max:.2f}] y[{y_min:.2f}, {y_max:.2f}] (padding={padding})"
    )

    r_min, r_max = vamp.ROBOT_RADII_RANGES["fetch"]
    print(
        f"Adding points to environment (r_min={r_min}, r_max={r_max}, point_radius={point_radius})..."
    )
    t0 = time.time()
    build_time_ns = env.add_pointcloud(pc.tolist(), r_min, r_max, point_radius)
    print(
        f"CAPT construction: {build_time_ns * 1e-6:.3f} ms; add time: {(time.time()-t0):.3f} s"
    )

    # RNG and FCIT settings
    rng = robot.halton()

    neighbor_params = vamp.FCITNeighborParams(robot.dimension(), robot.space_measure())
    settings = vamp.FCITSettings(neighbor_params)
    # Reasonable defaults; adjust as needed
    settings.max_iterations = 200
    settings.max_samples = 8192
    settings.batch_size = 4096
    settings.reverse_weight = 5.0
    settings.optimize = True

    print("Planning whole-body motion with FCIT* (arm+base together)...")
    res = robot.fcit_wb(
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        env,
        settings,
        rng,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    if res.validate_paths() and len(res.arm_result.path) >= 2:
        print("Solved problem!")
        print(
            f"Arm waypoints: {len(res.arm_result.path)}; Base waypoints: {len(res.base_path)}"
        )
        print(f"Arm path cost: {res.arm_result.path.cost():.3f}")
        # FCIT-style timing and iterations
        print(
            f"Planning Time: {int(res.arm_result.nanoseconds/1000)}\u03bcs | Iterations: {res.arm_result.iterations} | Graph size: {sum(res.arm_result.size) if res.arm_result.size else 0}"
        )

        if visualize:
            # Optionally interpolate to synchronize and densify
            try:
                res.interpolate(interp_density)
                print(f"Interpolated whole-body path with density {interp_density}")
            except Exception:
                pass

            # Extract synchronized paths
            arm_path = res.arm_result.path
            base_path = res.base_path

            # Convert base path to simple lists
            base_configs = [bp.config for bp in base_path]

            # Create simulator
            robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
            sim = vpb.PyBulletSimulator(
                str(robot_dir / "fetch_spherized.urdf"),
                vamp.ROBOT_JOINTS["fetch"],
                True,
            )

            # Draw environment (sample points)
            viz_points = sample_points_for_visualization(pc, max_points=75000)
            print(f"Visualizing {len(viz_points)} sampled points in PyBullet")
            sim.draw_pointcloud(viz_points)

            # Verify lengths
            arm_len = len(arm_path)
            base_len = len(base_configs)
            if arm_len != base_len:
                print(
                    f"ERROR: Path lengths don't match even after interpolation! Arm: {arm_len}, Base: {base_len}"
                )
                return

            print(f"Starting animation with {arm_len} synchronized waypoints...")
            try:
                for i in range(arm_len):
                    # Arm config element may be a bound type; convert to list
                    cfg = arm_path[i]
                    if isinstance(cfg, list):
                        arm_cfg = cfg
                    elif isinstance(cfg, np.ndarray):
                        arm_cfg = cfg.tolist()
                    else:
                        arm_cfg = cfg.to_list()

                    base_cfg = base_configs[i]

                    quat = pb.getQuaternionFromEuler([0, 0, base_cfg[2]])
                    sim.client.resetBasePositionAndOrientation(
                        sim.skel_id, [base_cfg[0], base_cfg[1], 0], quat
                    )
                    sim.set_joint_positions(arm_cfg)
                    time.sleep(0.05)

                print(
                    "Whole-body motion visualization complete! Press Ctrl+C to exit..."
                )
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("Visualization stopped by user.")
            except Exception as e:
                print(f"Error during visualization: {e}")
    else:
        print("Failed to solve problem or mismatched path lengths.")
        if visualize:
            # Show start and goal configurations
            robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
            sim = vpb.PyBulletSimulator(
                str(robot_dir / "fetch_spherized.urdf"),
                vamp.ROBOT_JOINTS["fetch"],
                True,
            )

            viz_points = sample_points_for_visualization(pc, max_points=75000)
            sim.draw_pointcloud(viz_points)

            quat_start = pb.getQuaternionFromEuler([0, 0, start_base[2]])
            sim.client.resetBasePositionAndOrientation(
                sim.skel_id, [start_base[0], start_base[1], 0], quat_start
            )
            sim.set_joint_positions(start_joints)
            print("Showing start configuration. Press Enter to continue...")
            input()

            quat_goal = pb.getQuaternionFromEuler([0, 0, goal_base[2]])
            sim.client.resetBasePositionAndOrientation(
                sim.skel_id, [goal_base[0], goal_base[1], 0], quat_goal
            )
            sim.set_joint_positions(goal_joints)
            print("Showing goal configuration. Press Enter to exit...")
            input()


if __name__ == "__main__":
    fire.Fire(main)
