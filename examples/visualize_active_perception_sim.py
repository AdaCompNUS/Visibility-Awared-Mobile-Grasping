#!/usr/bin/env python3
"""
Generate active perception visualization using ManiSkill simulation.

This script uses your actual ManiSkill simulation environment to create
paper-quality visualizations with REAL robot data, trajectories, and scenes.
"""

import argparse
import os
import sys
import time
import uuid

# Fix library path to use conda environment's libraries
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}"
        # Re-exec the script with updated LD_LIBRARY_PATH
        os.execv(sys.executable, [sys.executable] + sys.argv)

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import sapien  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from grasp_anywhere.core.sequential_scheduler import SequentialScheduler
from grasp_anywhere.envs.maniskill.maniskill_env_mpc import ManiSkillEnv


class RTManiSkillEnv(ManiSkillEnv):
    """
    Subclass of ManiSkillEnv that forces Ray-Tracing (shader_dir="rt")
    by hooking into gym.make during initialization.
    """

    def __init__(self, *args, **kwargs):
        original_make = gym.make

        def hooked_make(*make_args, **make_kwargs):
            make_kwargs["shader_dir"] = "rt"
            return original_make(*make_args, **make_kwargs)

        gym.make = hooked_make
        try:
            super().__init__(*args, **kwargs)
        finally:
            gym.make = original_make


from mani_skill.utils.building import actors  # noqa: E402

from grasp_anywhere.robot.fetch import Fetch  # noqa: E402
from grasp_anywhere.utils.active_perception_viz import (  # noqa: E402
    visualize_active_perception_dual_panel,
)


def extract_swept_volume_from_trajectory(
    robot_trajectory_poses,
    gripper_width=0.1,
    gripper_depth=0.15,
    num_samples_per_pose=50,
):
    """Extract swept volume points from robot trajectory."""
    all_points = []

    for pose in robot_trajectory_poses:
        local_points = np.random.randn(num_samples_per_pose, 3)
        local_points[:, 0] *= gripper_depth / 6
        local_points[:, 1] *= gripper_width / 6
        local_points[:, 2] *= gripper_width / 6

        world_points = (pose[:3, :3] @ local_points.T).T + pose[:3, 3]
        all_points.append(world_points)

    return np.vstack(all_points)


def compute_collision_risk(swept_volume_points, obstacle_points, threshold=0.05):
    """Compute collision risk based on proximity to obstacles."""
    from scipy.spatial import cKDTree

    tree = cKDTree(obstacle_points)
    distances, _ = tree.query(swept_volume_points, k=1)
    risk_scores = np.exp(-distances / threshold)
    risk_scores = np.clip(risk_scores, 0, 1)

    return risk_scores


def create_attention_map_from_mask(rgb_image, object_mask, blur_sigma=20):
    """Create smooth attention map from object segmentation mask."""
    import cv2

    attention = object_mask.astype(np.float32)
    attention = cv2.GaussianBlur(attention, (0, 0), blur_sigma)

    if attention.max() > 0:
        attention = attention / attention.max()

    return attention


def generate_trajectory_waypoints(pre_grasp_pose, grasp_pose, num_waypoints=5):
    """Generate intermediate waypoints between pre-grasp and grasp poses."""
    from scipy.spatial.transform import Rotation, Slerp

    waypoints = []

    pos_start = pre_grasp_pose[:3, 3]
    pos_end = grasp_pose[:3, 3]

    rot_start = Rotation.from_matrix(pre_grasp_pose[:3, :3])
    rot_end = Rotation.from_matrix(grasp_pose[:3, :3])

    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)

        pos = pos_start * (1 - t) + pos_end * t

        slerp = Slerp([0, 1], Rotation.concatenate([rot_start, rot_end]))
        rot = slerp(t)

        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos

        waypoints.append(T)

    return waypoints


def main():
    parser = argparse.ArgumentParser(
        description="Generate active perception visualization with ManiSkill simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="grasp_anywhere/configs/maniskill_fetch.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./active_perception_maniskill.png",
        help="Path to save the visualization",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=7,
        help="Number of trajectory waypoints to visualize",
    )
    parser.add_argument(
        "--object-id",
        type=str,
        default="006_mustard_bottle",
        help="YCB object ID to grasp (e.g., 006_mustard_bottle, 005_tomato_soup_can, 003_cracker_box)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Active Perception Visualization - ManiSkill Simulation")
    print("=" * 70)

    # Load config
    # with open(args.config, "r") as f:
    #     config = yaml.safe_load(f)

    render_mode = None  # Force headless mode to disable GUI
    # render_mode = config.get("debug", {}).get("render_mode", "human")
    # if render_mode == "None":
    #     render_mode = None

    # === 1. Initialize ManiSkill Environment ===
    # === 1. Initialize ManiSkill Environment (Fast Mode) ===
    print("\n[1/8] Initializing ManiSkill environment (Fast Mode)...")
    sim_env = ManiSkillEnv(
        env_id="ReplicaCAD_SceneManipulation-v1",
        robot_uids="fetch",
        render_mode=render_mode,
    )

    # Reset environment
    seed = 42
    sim_env.reset(seed=seed)

    # === 2. Place Object in Scene ===
    print(f"[2/8] Placing YCB object '{args.object_id}' in scene...")

    # Place object on table
    position = np.array([0.8, 0.0, 0.75], dtype=np.float32)  # On table
    orientation = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )  # Identity quaternion

    builder = actors.get_actor_builder(sim_env.env.scene, id=f"ycb:{args.object_id}")
    builder.initial_pose = sapien.Pose(p=position, q=orientation)
    actor_name = f"ycb_{args.object_id}_{uuid.uuid4().hex[:8]}"
    builder.build(name=actor_name)

    print(f"  Object placed at: {position}")

    # === 3. Initialize Robot ===
    print("[3/8] Initializing Fetch robot...")

    # Use canonical map if available
    canonical_map_path = os.path.expanduser(
        "~/grasp_anywhere_data/canonical_maps/ReplicaCAD_SceneManipulation-v1_scene_0_seed_42.npy"
    )
    static_pcd_paths = (
        [canonical_map_path] if os.path.exists(canonical_map_path) else []
    )

    fetch_robot = Fetch(
        config_path=args.config,
        robot_env=sim_env,
        static_pcd_paths=static_pcd_paths,
    )

    # Initialize scheduler
    scheduler = SequentialScheduler(robot=fetch_robot, config_path=args.config)

    # === 4. Move to Observation Pose (Pre-Pose) ===
    print("[4/8] Moving to observation pose using pre-pose planner...")

    # Use object position as point cloud (scheduler expects Nx3 array)
    object_pcd = position.reshape(-1, 3)

    # Get initial observation
    snapshot = fetch_robot.robot_env.get_sensor_snapshot()
    # rgb_init = snapshot["rgb"]
    depth_init = snapshot["depth"]

    camera_intrinsic = snapshot["intrinsics"]
    joint_names, joint_positions = snapshot["joint_states"]
    joint_dict = dict(zip(joint_names, joint_positions))
    camera_extrinsic_init = fetch_robot.compute_camera_pose_from_joints(joint_dict)

    # Update scene with initial observation
    fetch_robot.scene.update(
        depth_init,
        camera_intrinsic,
        camera_extrinsic_init,
        joint_dict,
        enable_icp_alignment=False,
    )

    # Sync collision environment with observation to avoid table collision
    fetch_robot.clear_pointclouds()
    fetch_robot.add_pointcloud(
        fetch_robot.scene.current_environment(),
        filter_robot=True,
        point_radius=0.03,
    )

    # Plan pre-pose
    config_generator, object_center = scheduler.prepose_planner.plan(
        object_pcd,
        camera_extrinsic_init,
        fetch_robot.scene.current_environment,
    )

    prepose_successful = False
    if config_generator is not None:
        # Execute pre-pose movement with replanning enabled
        prepose_successful = scheduler.prepose_planner.execute_prepose(
            config_generator,
            enable_replanning=True,
            enable_pcd_alignment=False,
        )

    if not prepose_successful:
        print("  WARNING: Pre-pose failed. Attempting fallback OBSERVE stage...")

        # Fallback to general observation stage
        success, msg = scheduler.move_planner.run_observe_object(
            object_pcd,
            fetch_robot.scene.current_environment(),
            manipulation_radius=2 * scheduler.manipulation_radius,
            enable_replanning=True,
            enable_pcd_alignment=False,
        )

        if success:
            print("  ✓ Moved to observation pose via fallback (OBSERVE stage)")
            prepose_successful = True

        else:
            print(f"  ERROR: Observe stage failed too: {msg}")
            print("  Continuing with current pose...")

    else:
        print("  ✓ Moved to observation pose successfully")
        # Point head at object

        fetch_robot.point_head_at(object_center.tolist())

        # Wait for motion to complete
        print("  Waiting for robot to reach observation pose...")
        while not fetch_robot.is_motion_done():
            time.sleep(0.1)

        # Extra small wait for simulator to settle
        time.sleep(0.5)
        print("  ✓ Robot reached observation pose")

    # === SWITCH TO RAY TRACING ===
    print("\n[RT SWITCH] Switching to Ray Tracing for high-quality rendering...")

    # 1. Save current state (Raw Values)
    # Access agent via unwrapped env to ensure we get the latest handle
    current_agent = sim_env.env.unwrapped.agent
    qpos = current_agent.robot.get_qpos()
    if hasattr(qpos, "cpu"):
        saved_qpos = qpos.detach().cpu().numpy()
    elif isinstance(qpos, np.ndarray):
        saved_qpos = qpos.copy()
    else:
        saved_qpos = np.array(qpos)
    current_pose = current_agent.robot.get_pose()
    saved_pos = np.array(current_pose.p).flatten()
    saved_quat = np.array(current_pose.q).flatten()
    saved_qpos = saved_qpos.flatten()

    print(f"  Saved Robot Pose: {saved_pos}")
    print(f"  Saved Robot QPos (first 3): {saved_qpos[:3]}")

    # 2. Close current environment
    sim_env.close()

    # 3. Initialize RT Environment
    print("[RT SWITCH] Initializing RT Environment...")
    sim_env = RTManiSkillEnv(
        env_id="ReplicaCAD_SceneManipulation-v1",
        robot_uids="fetch",
        render_mode=render_mode,
    )
    sim_env.reset(seed=seed)

    # 4. Re-place Object
    builder = actors.get_actor_builder(sim_env.env.scene, id=f"ycb:{args.object_id}")
    builder.initial_pose = sapien.Pose(p=position, q=orientation)
    actor_name = f"ycb_{args.object_id}_rt"
    builder.build(name=actor_name)

    # 5. Re-initialize Fetch Wrapper
    fetch_robot = Fetch(
        config_path=args.config,
        robot_env=sim_env,
        static_pcd_paths=static_pcd_paths,
    )

    # 6. Restore Robot State (AFTER Fetch init, using fresh agent handle)
    new_agent = sim_env.env.unwrapped.agent
    new_agent.robot.set_qpos(saved_qpos)
    new_agent.robot.set_pose(sapien.Pose(p=saved_pos, q=saved_quat))

    print(f"  Restored Robot Pose: {new_agent.robot.get_pose().p}")

    # 7. Wait for RT Accumulation
    print("[RT SWITCH] Waiting 2.5s for RT accumulation...")
    for _ in range(5):
        sim_env.env.unwrapped.scene.update_render()
        time.sleep(0.5)

    print("[RT SWITCH] RT Ready.")
    # ==============================

    # === 5. Capture Observation from Pre-Pose ===
    print("[5/8] Capturing observation from observation pose...")

    rgb = fetch_robot.get_rgb()
    depth = fetch_robot.get_depth()
    camera_intrinsic = fetch_robot.get_camera_intrinsic()
    camera_extrinsic = fetch_robot.get_camera_pose()

    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")

    # === 6. Create Object Mask ===
    print("[6/8] Creating object segmentation mask...")

    # Project object position to image
    object_pos_cam = np.linalg.inv(camera_extrinsic) @ np.append(position, 1)
    object_pos_cam = object_pos_cam[:3]

    # Create mask based on depth and proximity to object
    H, W = depth.shape
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

    # Project object to image plane
    u_obj = int(fx * object_pos_cam[0] / object_pos_cam[2] + cx)
    v_obj = int(fy * object_pos_cam[1] / object_pos_cam[2] + cy)

    # Create circular mask around projected object center
    y, x = np.ogrid[:H, :W]
    mask_radius = 80  # pixels
    mask = ((x - u_obj) ** 2 + (y - v_obj) ** 2 < mask_radius**2).astype(np.uint8)

    # Refine mask using depth
    object_depth = object_pos_cam[2]
    depth_mask = np.abs(depth - object_depth) < 0.15  # Within 15cm of object depth
    mask = (mask & depth_mask).astype(np.uint8)

    print(f"  Mask created: {mask.sum()} pixels")

    # Create attention map
    attention_map = create_attention_map_from_mask(rgb, mask)

    # === 7. Generate Future Trajectory ===
    print(f"[7/8] Generating {args.num_waypoints}-waypoint future trajectory...")

    # Get current end-effector pose using forward kinematics
    torso_pos = fetch_robot.get_torso_position()
    arm_joints = fetch_robot.get_arm_joint_values()
    current_config = [torso_pos] + arm_joints

    # Get EE pose in base frame
    ee_pos_base, ee_quat_base = fetch_robot.vamp_module.eefk(current_config)

    # Transform to world frame
    base_config = fetch_robot.get_base_params()  # [x, y, theta]
    from grasp_anywhere.robot.utils.transform_utils import transform_pose_to_world

    ee_pos_world, ee_quat_world = transform_pose_to_world(
        [base_config[0], base_config[1], 0], base_config[2], ee_pos_base, ee_quat_base
    )

    # Create current EE pose matrix
    current_ee_pose = np.eye(4)
    current_ee_pose[:3, :3] = Rotation.from_quat(ee_quat_world).as_matrix()
    current_ee_pose[:3, 3] = ee_pos_world

    # Create final grasp pose (approach toward object)
    grasp_pose = np.eye(4)
    grasp_pose[:3, 3] = position + np.array(
        [0, 0, 0.05]
    )  # Slightly above object center
    # Keep same orientation as current pose
    grasp_pose[:3, :3] = current_ee_pose[:3, :3]

    # Generate waypoints from current pose to grasp
    trajectory_poses = generate_trajectory_waypoints(
        current_ee_pose, grasp_pose, num_waypoints=args.num_waypoints
    )

    print(f"  Generated {len(trajectory_poses)} waypoints")
    print(f"  Start (current): {trajectory_poses[0][:3, 3]}")
    print(f"  End (grasp):     {trajectory_poses[-1][:3, 3]}")

    # === 8. Create Visualization ===
    print("[8/8] Creating active perception visualization...")

    # Get scene point cloud
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (z.flatten() > 0) & (z.flatten() < 3.0)
    points_cam = points_cam[valid]

    # Transform to world frame
    points_world = (camera_extrinsic[:3, :3] @ points_cam.T).T + camera_extrinsic[:3, 3]
    colors = rgb.reshape(-1, 3)[valid] / 255.0

    # Extract swept volume
    swept_volume = extract_swept_volume_from_trajectory(trajectory_poses)

    # Compute collision risk
    collision_risk = compute_collision_risk(swept_volume, points_world)

    # Create visualization
    panel_a, panel_b_geom = visualize_active_perception_dual_panel(
        rgb_image=rgb,
        object_attention_map=attention_map,
        robot_trajectory_poses=trajectory_poses,
        swept_volume_points=swept_volume,
        collision_risk_scores=collision_risk,
        object_mask=mask,
        scene_pcd=points_world,
        scene_colors=colors,
        save_path=args.save_path,
        show_visualization=False,  # Non-interactive mode
    )

    print("\n" + "=" * 70)
    print("✓ Visualization completed successfully!")
    print("=" * 70)
    print("Panel A (Object Attention) saved to:")
    print(f"  {args.save_path.replace('.png', '_panel_a.png')}")
    print("Visualization details:")

    print(f"  • Object: {args.object_id}")
    print(f"  • Trajectory waypoints: {len(trajectory_poses)}")
    print(f"  • Swept volume points: {len(swept_volume)}")
    print(f"  • Scene points: {len(points_world)}")
    print(f"  • Average collision risk: {collision_risk.mean():.3f}")

    # Save raw RT image
    rt_image_path = args.save_path.replace(".png", "_rt_raw.png")
    # Convert RGB to BGR for cv2
    cv2.imwrite(rt_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Raw Photo-Realistic Image saved to:\n  {rt_image_path}")
    print("=" * 70)

    # Cleanup
    sim_env.close()


if __name__ == "__main__":
    main()
