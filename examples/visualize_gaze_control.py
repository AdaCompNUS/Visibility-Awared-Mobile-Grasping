#!/usr/bin/env python3
"""
Visualize Gaze Control active perception.

This script demonstrates the robot's ability to maintain gaze on a target object
while moving its base/body. It visualizes:
1. The robot's future trajectory (transparent ghosts)
2. The active gaze control (head tracking) via attention heatmaps
"""

import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import sapien.core as sapien
from mani_skill.utils.building import actors
from scipy.spatial.transform import Rotation

from grasp_anywhere.envs.maniskill.maniskill_env_mpc import ManiSkillEnv

# Import Gaze Optimizer
from grasp_anywhere.observation.gaze_optimizer import GazeOptimizer
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.robot.kinematics import _create_transform_matrix, forward_kinematics

# Import Prepose Planner for active perception move


def calculate_head_angles(robot, base_params, joint_config_8dof, target_point):
    """
    Calculate head pan/tilt to look at target_point given a specific robot configuration.
    base_params: [x, y, theta]
    joint_config_8dof: [torso, shoulder_pan, ..., wrist_roll]
    target_point: [x, y, z] in map frame
    """
    # 1. FK to get head link poses in Base Frame
    # joint_config_8dof is [torso, 7 arm joints]. Forward kinematics needs [torso, 7 arm, head_pan, head_tilt]
    # We pass 0 for head initially to get the base transform of the head links
    full_joints = list(joint_config_8dof) + [0.0, 0.0]
    link_poses = forward_kinematics(full_joints)

    # 2. Transform Target from Map to Base Frame
    bx, by, bth = base_params
    T_map_base = _create_transform_matrix(
        [bx, by, 0], Rotation.from_euler("z", bth).as_matrix()
    )
    T_base_map = np.linalg.inv(T_map_base)
    target_base = (T_base_map @ np.append(target_point, 1))[:3]

    # 3. Compute Pan (Relative to Base -> Head Pan Link)
    T_base_head_pan = link_poses["head_pan_link"]
    T_head_pan_base = np.linalg.inv(T_base_head_pan)
    target_head_pan = (T_head_pan_base @ np.append(target_base, 1))[:3]
    x, y, z = target_head_pan
    pan_rel = float(np.arctan2(y, x))

    # 4. Compute Tilt
    # Tilt Angle is based on distance in XY plane vs Z height difference
    # Transform Tilt Joint frame to Pan Frame (aligned)
    T_base_head_tilt = link_poses["head_tilt_link"]
    T_pan_tilt = T_head_pan_base @ T_base_head_tilt
    tilt_origin_pan = T_pan_tilt[:3, 3]

    dist_xy = np.sqrt(x**2 + y**2)
    v_tilt_target = np.array([dist_xy, 0, z]) - tilt_origin_pan
    tilt_abs = float(np.arctan2(-v_tilt_target[2], v_tilt_target[0]))

    return pan_rel, tilt_abs


def main():
    parser = argparse.ArgumentParser(description="Visualize Gaze Control")
    parser.add_argument(
        "--config", type=str, default="grasp_anywhere/configs/maniskill_fetch.yaml"
    )
    parser.add_argument("--object-id", type=str, default="005_tomato_soup_can")
    parser.add_argument("--save-path", type=str, default="./gaze_control_viz.png")
    parser.add_argument(
        "--map-path",
        type=str,
        default="scene_map.ply",
        help="Path to pre-generated scene PCD",
    )
    args = parser.parse_args()

    # === 1. Setup Environment (Fast Mode) ===
    print("[1/6] Initializing Environment...")
    sim_env = ManiSkillEnv(
        env_id="ReplicaCAD_SceneManipulation-v1",
        robot_uids="fetch",
        render_mode="human",  # Enabled GUI
    )
    seed = 42
    sim_env.reset(seed=seed)

    # Load Scene Map if available
    scene_points = None
    if os.path.exists(args.map_path):
        print(f"Loading scene map from {args.map_path}...")
        pcd = o3d.io.read_point_cloud(args.map_path)
        scene_points = np.asarray(pcd.points)
        print(f"  Loaded {len(scene_points)} points.")
    else:
        print(
            f"Warning: Map {args.map_path} not found. Will use live environment scan."
        )

    # === 2. Place Object ===
    print(f"[2/6] Placing Object {args.object_id}...")
    position = np.array([0.8, 0.0, 0.75], dtype=np.float32)
    orientation = np.array([1, 0, 0, 0], dtype=np.float32)
    builder = actors.get_actor_builder(sim_env.env.scene, id=f"ycb:{args.object_id}")
    builder.initial_pose = sapien.Pose(p=position, q=orientation)
    builder.build(name=f"ycb_{args.object_id}")

    # === 3. Initialize Robot ===
    print("[3/6] Initializing Robot and Scheduler...")

    # Define static PCD paths for Fetch init (enables reachability map usage if internal logic supports it)
    static_pcd_paths = [args.map_path] if os.path.exists(args.map_path) else []

    fetch_robot = Fetch(
        config_path=args.config,
        robot_env=sim_env,
        static_pcd_paths=static_pcd_paths,  # Pass map here
    )

    # === Reset Arm to Default Config ===
    print("Resetting arm to default configuration...")
    # Default config found in project: [torso, shoulder_pan, ..., wrist_roll]
    default_config = [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0]
    planning_joint_names = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]
    target_qpos_dict = dict(zip(planning_joint_names, default_config))

    # Access internal sapien robot to set qpos directly
    # sim_env.env is the gym environment, sim_env.env.agent.robot is the Sapien articulation

    # Use unwrapped to avoid warnings and ensure access to inner attributes
    unwrapped_env = sim_env.env.unwrapped
    agent = unwrapped_env.agent
    robot_entity = agent.robot

    qpos = robot_entity.get_qpos()
    active_joints = robot_entity.get_active_joints()

    # Update qpos with default config values
    # qpos is (1, N_dof), so we index [0, i]
    for i, joint in enumerate(active_joints):
        if joint.name in target_qpos_dict:
            qpos[0, i] = target_qpos_dict[joint.name]
            # Also set drive target for position control to hold this pose
            if hasattr(joint, "set_drive_target"):
                joint.set_drive_target(target_qpos_dict[joint.name])

    robot_entity.set_qpos(qpos)
    # Step simulation to stabilize
    sim_env.env.step(np.zeros(sim_env.env.action_space.shape))

    from grasp_anywhere.core.sequential_scheduler import SequentialScheduler

    scheduler = SequentialScheduler(robot=fetch_robot, config_path=args.config)

    # Initialize Collision Env with Map
    if scene_points is not None:
        print("Initializing Collision Environment with Static Map...")
        fetch_robot.clear_pointclouds()
        fetch_robot.add_pointcloud(
            scene_points,
            filter_robot=False,  # Map shouldn't contain robot
            point_radius=0.03,
        )

    # Move to initial looking pose (somewhat adjacent)
    # Move base to look at object
    # Move base to look at object

    fetch_robot.robot_env.agent.robot.set_pose(
        sapien.Pose(p=[-0.5, 0.8, 0], q=Rotation.from_euler("z", -1.0).as_quat())
    )

    # Point head at object
    print("Pointing head at object...")
    fetch_robot.point_head_at(position.tolist())
    for _ in range(20):
        sim_env.env.step(np.zeros(sim_env.env.action_space.shape))

    # === 4. Generate Future Trajectory (Arc) ===
    # === 4. Active Perception Plan (Pre-Pose) ===
    print("[4/6] Executing Active Perception Move (Pre-Pose)...")

    # 1. Update Robot Scene/Collision
    if scene_points is None:
        # Fallback to live scan if map missing
        fetch_robot.clear_pointclouds()
        fetch_robot.add_pointcloud(
            fetch_robot.scene.current_environment(),
            filter_robot=True,
            point_radius=0.03,
        )
        planner_points = fetch_robot.scene.current_environment
    else:
        # Already added map above, just pass it to planner
        planner_points = scene_points

    # 2. Define "Object PCD" (just a small cloud around the known center for planning)
    # The prepose planner expects a PCD to look at/grasp.
    # We create a synthetic one at the spawn position.
    center = position[:3]
    num_pts = 100
    obj_pcd = np.random.randn(num_pts, 3) * 0.05 + center

    # 3. Use Scheduler's Prepose Planner
    # Ensure update of scheduler's planners with current scene if needed

    # 4. Plan
    # get_current_environment is a callable
    config_generator, _ = scheduler.prepose_planner.plan(
        obj_pcd,
        camera_pose=fetch_robot.get_camera_pose(),  # Not strictly used by some planners but good for args
        combined_points=planner_points,
    )

    # 5. Execute & Capture Trajectory
    # We want to capture the trajectory points for visualization.
    # The default execute_prepose doesn't return the path, just success.
    # However, it calls `move_to_config_with_replanning` which plans internally.
    # To get the path *for visualization*, we can do the planning step manually here using the config found.

    future_ee_poses = []
    robot_trajectory_poses = []  # Standard naming for swept volume function

    if config_generator:
        # Get the first valid plan proposed
        try:
            target_base_config, target_arm_config = next(config_generator)
            print(f"  Found Valid Pre-Pose: Base={target_base_config}")

            # Now, PLAN the motion to this target using Whole Body Planner to get the trajectory
            current_base = fetch_robot.get_base_params()
            current_arm = fetch_robot.get_arm_joint_values()
            current_torso = fetch_robot.get_torso_position()
            start_joints = [current_torso] + current_arm

            # Plan Real Path
            plan_res = fetch_robot.plan_whole_body_motion(
                start_joints=start_joints,
                goal_joints=target_arm_config,
                start_base=current_base,
                goal_base=target_base_config,
                planner="rrtc",
            )

            if plan_res and plan_res.get("success"):
                arm_path = plan_res["arm_path"]  # List of 8dof
                base_path = plan_res["base_configs"]  # List of 3dof
                print(f"  Trajectory Planned: {len(base_path)} waypoints")

                # === Run Gaze Optimizer Offline ===
                # 1. Construct 11-DOF trajectory for Optimizer (Base+Arm)
                # Optimizer expects: [x, y, theta, torso, shoulder_pan, ..., wrist_roll]
                traj_11dof = []
                for b, a in zip(base_path, arm_path):
                    # b=[x,y,th], a=[torso, 7_arm...]
                    # 11-dof = b + a
                    traj_11dof.append(list(b) + list(a))

                # 2. Init Optimizer
                # We need a mock robot or just use fetch_robot since optimizer calls point_head_at
                # (which we won't execute live here).
                # Actually optimizer.update() calls robot.point_head_at().
                # We can monkeypatch it or just calculate manually.
                # BETTER: Use optimizer._compute_weighted_target() directly!

                optimizer = GazeOptimizer(
                    fetch_robot, lookahead_window=20, decay_rate=0.95
                )
                optimizer.set_trajectory(traj_11dof)

                # 3. Generate Full States with Gaze

                gaze_targets = []  # Collect gaze targets for heatmap
                gaze_scores = []  # Collect velocity/weight scores

                for i in range(len(base_path)):
                    # Compute gaze target
                    target_point = optimizer._compute_weighted_target(i)
                    gaze_targets.append(target_point)

                    # Calculate score based on velocity (max valid velocity at this step)
                    step_score = 0.0
                    if optimizer.joint_velocities is not None:
                        # Get max velocity among key joints to represent "intensity"
                        vels = [
                            optimizer.joint_velocities[k][i]
                            for k in optimizer.key_joints
                            if k in optimizer.joint_velocities
                        ]
                        if vels:
                            step_score = max(vels)
                    gaze_scores.append(step_score)

                    # Calculate EE Pose (independent of head)
                    b_cfg = base_path[i]
                    a_cfg = arm_path[i]

                    ee_mat = fetch_robot.get_end_effector_pose(
                        joint_values=a_cfg, base_config=b_cfg
                    )
                    robot_trajectory_poses.append(ee_mat)

                    # Store for ghosts (if we visualize full body)
                    # For now just collecting EE poses for swept volume is critical
                    if i % 5 == 0:  # Sparse ghosts
                        future_ee_poses.append(ee_mat)

                print("  Gaze-aware trajectory computed.")

                # Execute it on robot (visual execution) if needed, or just warp to end
                print("  Warping robot to final pose for viz context...")
                fetch_robot.vamp_module.set_base_params(
                    target_base_config[2], target_base_config[0], target_base_config[1]
                )
                # We can't easily warp sim completely without reset, just step logic

            else:
                print("  Motion Planning to Config Failed.")
        except StopIteration:
            print("  No valid pre-pose found by generator.")
    else:
        print("  Prepose planning returned no generator.")

    # Point head at object after arrival (simulated final state)
    fetch_robot.point_head_at(center.tolist())
    for _ in range(10):
        sim_env.env.step(np.zeros(sim_env.env.action_space.shape))

    # Visualize trajectory...

    # === 5. Capture Info (No RT Switch) ===
    print("[5/6] Capturing Info (Standard Rendering)...")

    # Wait a bit for simulator to settle
    for _ in range(10):
        sim_env.env.step(np.zeros(sim_env.env.action_space.shape))

    # Capture info
    rgb = fetch_robot.get_rgb()
    depth = fetch_robot.get_depth()

    # === 6. Visualize ===
    print("[6/6] Generating Visualization...")

    # Create attention map (Simulated 'Object Attention' - centered on object)
    # Project object center to image
    cam_pose = fetch_robot.get_camera_pose()
    K = fetch_robot.get_camera_intrinsic()

    # World to Cam
    T_world_cam = np.linalg.inv(cam_pose)
    obj_pos_cam = T_world_cam @ np.append(position, 1.0)
    obj_pos_cam = obj_pos_cam[:3]

    # Project to UV
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = rgb.shape[:2]

    # --- Generate Heatmap from Gaze Targets ---
    # print(f"Generating Gaze Attention Heatmap from {len(gaze_targets)} points...")

    # Normalize scores for coloring
    if len(gaze_scores) > 0:
        max_score = max(gaze_scores) if max(gaze_scores) > 1e-6 else 1.0
        norm_scores = [s / max_score for s in gaze_scores]
    else:
        norm_scores = []

    # Use a colored canvas for the heatmap
    attention_canvas = np.zeros((H, W, 3), dtype=np.float32)

    # Project all gaze targets
    gaze_pts_uv = []

    for idx, g_pt in enumerate(gaze_targets):
        if g_pt is None:
            continue
        # World to Cam
        pt_cam = (T_world_cam @ np.append(g_pt, 1.0))[:3]
        if pt_cam[2] > 0.01:  # In front of camera
            u_g = int(pt_cam[0] * fx / pt_cam[2] + cx)
            v_g = int(pt_cam[1] * fy / pt_cam[2] + cy)
            gaze_pts_uv.append((u_g, v_g))

            # Color based on score (Blue -> Red)
            score = norm_scores[idx]
            # Map 0..1 to HSV or just simple interpolation
            # Simple Blue (0,0,1) to Red (1,0,0) via Green
            # actually cv2.applyColorMap is easier if we had a grayscale image,
            # but we want to draw blobs directly.
            # Heatmap color: (B, G, R)
            # Low score = Blue, High score = Red
            color = (1.0 - score, score * 0.5, score)  # Simple approximation

            # Draw circle
            if 0 <= u_g < W and 0 <= v_g < H:
                # Accumulate? Or just Max?
                # For "Attention", accumulation is good for dwell, but color needs to reflect intensity.
                # Let's draw weighted circles on a separate float buffer and normalization.
                radius = 15  # Larger radius for attention
                cv2.circle(attention_canvas, (u_g, v_g), radius, color, -1)

    # Blur the colored canvas
    # Check if empty
    if len(gaze_pts_uv) > 0:
        attention_map_colored = cv2.GaussianBlur(attention_canvas, (51, 51), 20)
        # Clip
        attention_map_colored = np.clip(attention_map_colored, 0, 1)

        # Override the standard heatmap behavior
        # We will directly blend this colored map onto the original image
        heatmap_overlay = (rgb / 255.0) * 0.6 + attention_map_colored * 0.4
        heatmap_overlay = np.clip(heatmap_overlay, 0, 1)
        heatmap_overlay = (heatmap_overlay * 255).astype(np.uint8)

        # NOTE: create_attention_heatmap typically expects a single channel mask.
        # Since we did custom colored painting, we skip create_attention_heatmap
        # for the heatmap part and use our manual overlay.
    else:
        print("Warning: No valid gaze targets projected.")
        heatmap_overlay = rgb.copy()

    # Use light transparent colors for ghosts (Light Blue/Cyan with low alpha)
    ghost_colors = [(0.6, 0.9, 1.0)] * len(future_ee_poses)

    # Initialize Geometry List for 3D View
    geoms = []

    # === Volumetric Heatmap Field (Voxels) ===
    print("Generating Volumetric Gaze Field (Space Filling)...")

    # === Volumetric Heatmap Field (Voxels) ===
    print("Generating Volumetric Gaze Field (Space Filling)...")

    # Collect ALL points that contribute to the field FIRST to determine bounds
    all_sources = []
    all_weights = []

    # 1. Add Gaze Targets (DISABLED as per user request to focus on joints)
    # if len(gaze_targets) > 0:
    #     for i, t in enumerate(gaze_targets):
    #         if t is not None:
    #             all_sources.append(t)
    #             all_weights.append(norm_scores[i] if i < len(norm_scores) else 0.0)

    # 2. Add Joint Trajectories (Weighted by their velocity)
    # 2. Add Joint/Sphere Trajectories (Weighted by their velocity)
    used_spheres = False
    if (
        hasattr(optimizer, "sphere_trajectories")
        and optimizer.sphere_trajectories is not None
    ):
        print("Using GPU Sphere Trajectories for Field Generation (Dense)...")
        # Extract from GPU
        # (B, N, 3)
        sphere_traj_np = optimizer.sphere_trajectories.cpu().numpy()
        # (B, N)
        sphere_vel_np = optimizer.sphere_velocities.cpu().numpy()

        # Compute global max velocity for normalization
        global_max_vel = sphere_vel_np.max() if sphere_vel_np.size > 0 else 1.0
        if global_max_vel < 1e-6:
            global_max_vel = 1.0

        # Flatten and add
        # Shape: (TotalPoints, 3)
        flat_sources = sphere_traj_np.reshape(-1, 3)
        # Shape: (TotalPoints,)
        flat_weights = sphere_vel_np.reshape(-1) / global_max_vel

        all_sources.extend(flat_sources.tolist())
        all_weights.extend(flat_weights.tolist())
        used_spheres = True

    if not used_spheres and (
        "base_path" in locals()
        and base_path
        and "arm_path" in locals()
        and arm_path
        and hasattr(optimizer, "joint_velocities")
    ):
        # Consistent Joint Map
        FIELD_JOINT_MAP = {
            "base": "base_link",
            "torso": "torso_lift_link",
            "shoulder_lift": "shoulder_lift_link",
            "elbow": "elbow_flex_link",
            "wrist_flex": "wrist_flex_link",
            "gripper": "gripper_link",
        }

        global_max_vel = 1.0
        for k, vels in optimizer.joint_velocities.items():
            if len(vels) > 0:
                global_max_vel = max(global_max_vel, np.max(vels))

        # Temporary storage for interpolation
        raw_joint_data = {k: {"pos": [], "weight": []} for k in FIELD_JOINT_MAP.keys()}

        stride = 1  # Use all waypoints
        for i in range(0, len(base_path), stride):
            base_cfg = base_path[i]
            arm_cfg = arm_path[i]

            pan, tilt = 0.0, 0.0

            joints_10dof = np.array(list(arm_cfg) + [pan, tilt])
            link_poses_base = forward_kinematics(joints_10dof)

            bx, by, btheta = base_cfg
            T_world_base = _create_transform_matrix(
                [bx, by, 0], Rotation.from_euler("z", btheta).as_matrix()
            )

            for opt_name, link_name in FIELD_JOINT_MAP.items():
                if link_name in link_poses_base:
                    T_base_link = link_poses_base[link_name]
                    pos = (T_world_base @ T_base_link)[:3, 3]

                    w = 0.0
                    if opt_name in optimizer.joint_velocities:
                        vels = optimizer.joint_velocities[opt_name]
                        if i < len(vels):
                            w = vels[i]

                    norm_w = w / global_max_vel

                    raw_joint_data[opt_name]["pos"].append(pos)
                    raw_joint_data[opt_name]["weight"].append(norm_w)

        # Use all trajectory points directly (User confirmed trajectory is already dense enough)
        for name, data in raw_joint_data.items():
            positions = data["pos"]
            weights = data["weight"]

            for p, w in zip(positions, weights):
                all_sources.append(p)
                all_weights.append(w)

    # Check if we have sources to define bounds
    if len(all_sources) > 0:
        start_pts = np.array(all_sources)
        min_b = start_pts.min(axis=0) - 0.01
        max_b = start_pts.max(axis=0) + 0.01

        if min_b[2] < 0.0:
            min_b[2] = 0.0

        # Increase grid step to 0.06 (from 0.04) to make the point cloud "lighter" / more transparent
        step_size = 0.06
        x_range = np.arange(min_b[0], max_b[0], step_size)
        y_range = np.arange(min_b[1], max_b[1], step_size)
        z_range = np.arange(min_b[2], max_b[2], step_size)

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        targets_np = np.array(all_sources)
        scores_np = np.array(all_weights)

        # Filter weak scores to save compute?
        # No, we need weak scores for shape occupancy!
        # Just filter strictly zero if needed, but occupancy relies on existence.
        # We'll use random subset if too huge, but 0.03 spacing isn't insane.

        rel_targets = targets_np
        rel_scores = scores_np

        if len(rel_targets) > 0:
            print(
                f"  Computing Gaze Field using Stable Gaussian Accumulation ({len(rel_targets)} sources)..."
            )

            # Decoupled Field Generation:
            # 1. Occupancy Field: Uniform weight (1.0) -> Defines the "Volume" / Shape (consistent size)
            # 2. Intensity Field: Weighted by score -> Defines the "Color" (heatmap)

            sigma = 0.08  # Reduced from 0.15 to prevent excessive inflation
            sigma_sq_2 = 2 * sigma**2

            occupancy_vals = np.zeros(len(grid_points))
            intensity_vals = np.zeros(len(grid_points))

            threshold_dist = 2.5 * sigma

            for t_idx, target in enumerate(rel_targets):
                score = rel_scores[t_idx]
                if score < 0.01:
                    continue

                min_b = target - threshold_dist
                max_b = target + threshold_dist

                mask = (
                    (grid_points[:, 0] >= min_b[0])
                    & (grid_points[:, 0] <= max_b[0])
                    & (grid_points[:, 1] >= min_b[1])
                    & (grid_points[:, 1] <= max_b[1])
                    & (grid_points[:, 2] >= min_b[2])
                    & (grid_points[:, 2] <= max_b[2])
                )

                if np.any(mask):
                    d_sq = np.sum((grid_points[mask] - target) ** 2, axis=1)
                    radial_decay = np.exp(-d_sq / sigma_sq_2)

                    # Update Occupancy (Shape) - use MAX to trace the volume shell
                    occupancy_vals[mask] = np.maximum(
                        occupancy_vals[mask], radial_decay
                    )

                    # Update Intensity (Color) - weighted
                    # We want the max intensity at this point in space
                    intensity_vals[mask] = np.maximum(
                        intensity_vals[mask], score * radial_decay
                    )

            # Threshold on OCCUPANCY (Geometry) for consistent size
            # 0.1 corresponds to exp(-d2/2s2) = 0.1 => d ~ 2.15 sigma
            mask = occupancy_vals > 0.1
            valid_points = grid_points[mask]

            # Use intensity for values
            valid_vals = intensity_vals[mask]

            if len(valid_points) > 0:
                print(f"  Generating VoxelGrid from {len(valid_points)} voxels...")

                # Create PointCloud
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(valid_points)

                # Colors: Based on INTENSITY (valid_vals)
                colors = np.zeros((len(valid_points), 3))
                v_min, v_max = valid_vals.min(), valid_vals.max()
                denom = v_max - v_min if v_max - v_min > 1e-6 else 1.0
                norm_v = np.clip((valid_vals - v_min) / denom, 0, 1)

                # Color Mapping Tuning
                # Outer (low val) -> Transparent/White/No Color (we act by culling or coloring fainter)
                # Inner (high val) -> Colorful (Red/Yellow)

                import matplotlib.pyplot as plt

                cmap = plt.get_cmap("jet")
                mapped_colors = cmap(norm_v)[:, :3]

                pcd_temp.colors = o3d.utility.Vector3dVector(mapped_colors)

                # Use PointCloud directly to simulate "Transparency" / Gaseous look
                # (VoxelGrid is solid cubes and opaque in Open3D standard shader)
                print("  Adding dense PointCloud to simulate transparent field...")
                geoms.append(pcd_temp)

    # === Multi-Joint Continuous Trajectories ===
    print(
        "Visualizing Continuous Joint Trajectories (Colored by Gaze Weight/Velocity)..."
    )

    if (
        not used_spheres  # Skip skeleton if using detailed sphere field
        and "base_path" in locals()
        and base_path
        and "arm_path" in locals()
        and arm_path
        and hasattr(optimizer, "joint_velocities")
    ):
        # We will draw continuous lines (streamlines) for key joints
        JOINT_MAP = {
            "base": "base_link",
            "torso": "torso_lift_link",
            "shoulder_lift": "shoulder_lift_link",
            "elbow": "elbow_flex_link",
            "wrist_flex": "wrist_flex_link",
            "gripper": "gripper_link",
        }

        # Store points for line sets
        joint_trails = {k: [] for k in JOINT_MAP.keys()}
        gaze_trail = []

        stride = 1  # Full density for smoothness
        for i in range(0, len(base_path), stride):
            # Kinematics
            base_cfg = base_path[i]
            arm_cfg = arm_path[i]
            pan, tilt = 0.0, 0.0
            if i < len(gaze_targets) and gaze_targets[i] is not None:
                pan, tilt = calculate_head_angles(
                    fetch_robot, base_cfg, arm_cfg, gaze_targets[i]
                )
                gaze_trail.append(gaze_targets[i])

            joints_10dof = np.array(list(arm_cfg) + [pan, tilt])
            link_poses_base = forward_kinematics(joints_10dof)

            bx, by, btheta = base_cfg
            T_world_base = _create_transform_matrix(
                [bx, by, 0], Rotation.from_euler("z", btheta).as_matrix()
            )

            for opt_name, link_name in JOINT_MAP.items():
                if link_name in link_poses_base:
                    T_base_link = link_poses_base[link_name]
                    T_world_link = T_world_base @ T_base_link
                    pos = T_world_link[:3, 3]
                    joint_trails[opt_name].append(pos)

        # Find global max velocity for normalization to comparable scales
        max_vel = 0.0
        for opt_name in JOINT_MAP.keys():
            if opt_name in optimizer.joint_velocities:
                v_arr = optimizer.joint_velocities[opt_name]
                if len(v_arr) > 0:
                    max_vel = max(max_vel, np.max(v_arr))

        if max_vel < 1e-6:
            max_vel = 1.0

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("jet")

        for name, points in joint_trails.items():
            if len(points) > 1:
                lines = [[j, j + 1] for j in range(len(points) - 1)]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)

                # Dynamic Coloring based on Velocity/Weight
                # Weight ~ Velocity
                colors = []
                vels = np.array(optimizer.joint_velocities.get(name, []))

                # Smooth the velocities to avoid jagged "segment by segment" colors
                if len(vels) > 0:
                    window = 5
                    # Simple moving average
                    smoothed_vels = np.convolve(
                        vels, np.ones(window) / window, mode="same"
                    )
                else:
                    smoothed_vels = vels

                for j in range(len(points) - 1):
                    # Map velocity to color
                    # Use smoothed velocity at index j
                    val = 0.0
                    idx = j
                    if idx < len(smoothed_vels):
                        val = smoothed_vels[idx]

                    norm_v = np.clip(val / max_vel, 0, 1)
                    # Get RGB from cmap
                    c = cmap(norm_v)[:3]
                    colors.append(c)

                line_set.colors = o3d.utility.Vector3dVector(colors)
                # geoms.append(line_set) # DISABLED: User wants "no line at all", just field.

        # Gaze Target Trail (Yellow Line) - This is the output, so keep it distinct?
        # if len(gaze_trail) > 1:
        #     g_lines = [[j, j+1] for j in range(len(gaze_trail)-1)]
        #     g_line_set = o3d.geometry.LineSet()
        #     g_line_set.points = o3d.utility.Vector3dVector(gaze_trail)
        #     g_line_set.lines = o3d.utility.Vector2iVector(g_lines)
        #     g_line_set.paint_uniform_color([1.0, 1.0, 0.0]) # Keep Target Yellow as Reference
        #     geoms.append(g_line_set)

    # Need Scene PCD
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = (z > 0) & (z < 3.0)
    x_c = (u_grid - cx) * z / fx
    y_c = (v_grid - cy) * z / fy
    pts_cam = np.stack([x_c, y_c, z], axis=-1).reshape(-1, 3)
    pts_valid = pts_cam[valid.flatten()]
    colors_valid = rgb.reshape(-1, 3)[valid.flatten()] / 255.0

    pts_world = (cam_pose[:3, :3] @ pts_valid.T).T + cam_pose[:3, 3]

    # 2. Add Scene Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)
    geoms.append(pcd)

    # Coordinate frame
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    # === 7. Add Robot Visualization ===
    print("[7/6] Adding Robot Mesh to Visualization...")

    # Define Mesh Map (Absolute path)
    # Define Mesh Map (Relative path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    mesh_dir = os.path.join(project_root, "resources/fetch_ext/meshes")

    # Map Link Name -> {file, color}
    # Note: Using collision STL files for many links because Open3D often lacks DAE support
    robot_visuals = {
        "base_link": ("base_link_collision.STL", [0.356, 0.361, 0.376]),
        "torso_lift_link": ("torso_lift_link_collision.STL", [1.0, 1.0, 1.0]),
        "head_pan_link": ("head_pan_link_collision.STL", [0.356, 0.361, 0.376]),
        "head_tilt_link": ("head_tilt_link_collision.STL", [0.086, 0.506, 0.767]),
        "shoulder_pan_link": ("shoulder_pan_link_collision.STL", [1.0, 1.0, 1.0]),
        "shoulder_lift_link": (
            "shoulder_lift_link_collision.STL",
            [0.086, 0.506, 0.767],
        ),
        "upperarm_roll_link": ("upperarm_roll_link_collision.STL", [1.0, 1.0, 1.0]),
        "elbow_flex_link": ("elbow_flex_link_collision.STL", [0.086, 0.506, 0.767]),
        "forearm_roll_link": ("forearm_roll_link_collision.STL", [1.0, 1.0, 1.0]),
        "wrist_flex_link": ("wrist_flex_link_collision.STL", [0.086, 0.506, 0.767]),
        "wrist_roll_link": ("wrist_roll_link_collision.STL", [1.0, 1.0, 1.0]),
        "gripper_link": ("gripper_link.STL", [0.356, 0.361, 0.376]),
        "r_gripper_finger_link": ("r_gripper_finger_link.STL", [0.356, 0.361, 0.376]),
        "l_gripper_finger_link": ("l_gripper_finger_link.STL", [0.356, 0.361, 0.376]),
        "bellows_link": ("bellows_link.STL", [0.0, 0.0, 0.0]),
        "torso_fixed_link": ("torso_fixed_link.STL", [0.086, 0.506, 0.767]),
    }

    # Get Current State
    curr_base = fetch_robot.get_base_params()
    curr_joints = fetch_robot.get_current_planning_joints_with_head()

    if curr_joints is not None:
        link_poses = forward_kinematics(np.array(curr_joints))

        # Handle bellows manually (same as torso_lift)
        if "torso_lift_link" in link_poses:
            link_poses["bellows_link"] = link_poses["torso_lift_link"]

        # Base Transform
        bx, by, btheta = curr_base
        T_world_base = _create_transform_matrix(
            [bx, by, 0], Rotation.from_euler("z", btheta).as_matrix()
        )

        for link_name, (mesh_file, color) in robot_visuals.items():
            if link_name in link_poses:
                mesh_path = os.path.join(mesh_dir, mesh_file)
                if os.path.exists(mesh_path):
                    try:
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        # Fallback for empty/unsupported formats
                        if not mesh.has_triangles():
                            base_name = os.path.splitext(mesh_file)[0]
                            # Try collision file pattern
                            alt_candidates = [
                                base_name + "_collision_2.stl",
                                base_name.replace("link", "link_collision_2") + ".stl",
                                mesh_file.replace(".dae", ".STL"),
                            ]
                            for alt in alt_candidates:
                                alt_path = os.path.join(mesh_dir, alt)
                                if os.path.exists(alt_path):
                                    mesh = o3d.io.read_triangle_mesh(alt_path)
                                    if mesh.has_triangles():
                                        break

                        if mesh.has_triangles():
                            mesh.compute_vertex_normals()
                            mesh.paint_uniform_color(color)

                            # Transform
                            T_base_link = link_poses[link_name]
                            T_world_link = T_world_base @ T_base_link
                            mesh.transform(T_world_link)

                            geoms.append(mesh)
                        else:
                            print(f"Warning: Failed to load mesh {mesh_file} (empty)")
                    except Exception as e:
                        print(f"Error loading {mesh_file}: {e}")
                else:
                    print(f"Warning: Mesh file not found: {mesh_path}")

    print("Showing interactive 3D window... (Close window to finish)")
    o3d.visualization.draw_geometries(
        geoms, window_name="Gaze Control Viz", width=1024, height=768
    )

    # Save standard images
    cv2.imwrite(
        args.save_path.replace(".png", "_raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )
    print("Done.")


if __name__ == "__main__":
    main()
