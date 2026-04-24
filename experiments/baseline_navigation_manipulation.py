#!/usr/bin/env python3
import base64
import os
import time
from io import BytesIO

import numpy as np
import requests
import yaml

from grasp_anywhere.grasping_client.sam_client import SamClient, SamConfig
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.stage_planners.grasp_stage import GraspPlanner
from grasp_anywhere.stage_planners.prepose_stage import PreposePlanner
from grasp_anywhere.utils.gui_utils import ClickPointCollector
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning
from grasp_anywhere.utils.perception_utils import depth2pc, get_pcd_from_mask

# Local grasp generation service URL. Override with GRASPNET_URL if needed.
GRASPNET_URL = os.environ.get("GRASPNET_URL", "http://localhost:4003")


def _convert_pil_image_to_base64(image):
    pass

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def _predict_grasps_http(image_rgb, image_depth, segmap, K):
    """
    Sends RGB, Depth, Segmap, and Intrinsics to the GraspNet server and gets grasp predictions.
    Follows examples/grasp_script.py behavior.
    """
    from PIL import Image

    segmap_id = 1
    image_depth_mm = (image_depth * 1000).astype(np.uint32)

    image_rgb_pil = Image.fromarray(image_rgb)
    image_depth_pil = Image.fromarray(image_depth_mm)
    segmap_pil = Image.fromarray(segmap)

    image_rgb_base64 = _convert_pil_image_to_base64(image_rgb_pil)
    image_depth_base64 = _convert_pil_image_to_base64(image_depth_pil)
    segmap_base64 = _convert_pil_image_to_base64(segmap_pil)

    if isinstance(K, np.ndarray):
        K = K.flatten().tolist()

    payload = {
        "image_rgb": image_rgb_base64,
        "image_depth": image_depth_base64,
        "segmap": segmap_base64,
        "K": K,
        "segmap_id": segmap_id,
    }

    url = GRASPNET_URL + "/sample_grasp"
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        pred_grasps_cam = np.array(result["pred_grasps_cam"])
        scores = np.array(result["scores"])
        return pred_grasps_cam, scores
    except requests.exceptions.RequestException as e:
        print(f"ERROR: GraspNet HTTP request failed: {e}")
        return None, None


def quaternion_from_yaw(yaw):
    half = yaw / 2.0
    return [0.0, 0.0, float(np.sin(half)), float(np.cos(half))]


def transform_points_camera_to_world(points_cam, camera_extrinsic):
    if points_cam is None or len(points_cam) == 0:
        return None
    points_cam_h = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
    return (camera_extrinsic @ points_cam_h.T).T[:, :3]


def compute_environment_pointcloud_world(depth, K, camera_extrinsic):
    # Use manipulation-friendly voxel size
    observed_pcd_cam, _ = depth2pc(depth, K, voxel_size=0.05)
    if observed_pcd_cam is None or len(observed_pcd_cam) == 0:
        return np.empty((0, 3), dtype=np.float32)
    observed_pcd_world = transform_points_camera_to_world(
        observed_pcd_cam, camera_extrinsic
    )
    # Filter ground
    observed_pcd_world = observed_pcd_world[observed_pcd_world[:, 2] > 0.3]
    return observed_pcd_world


def load_static_collision_points():
    """Load static environment point cloud from resources/mp_collision_models."""
    try:
        import open3d as o3d
    except Exception as e:
        print(f"Failed to import Open3D for loading static collision models: {e}")
        return None

    pcd_files = [
        "coffee_table.ply",
        "open_kitchen.ply",
        "rls_2.ply",
        "sofa.ply",
        "table.ply",
        "wall.ply",
        "workstation.ply",
    ]

    combined_points = None
    for p in pcd_files:
        pcd_path = os.path.join("resources", "mp_collision_models", p)
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            if len(pcd.points) > 0:
                pts = np.asarray(pcd.points)
                combined_points = (
                    pts
                    if combined_points is None
                    else np.vstack((combined_points, pts))
                )
        except Exception as e:
            print(f"Could not load {pcd_path}: {e}")

    return combined_points


def main():
    print("=== Whole-body ablation experiment: base-only prepose, then grasp ===")

    # Initialize robot and helpers
    robot = Fetch(config_path="grasp_anywhere/configs/real_fetch.yaml")
    # For table target configs (pose/joints/head), reuse MovePlanner's dictionary without using its navigation API
    from grasp_anywhere.stage_planners.move_stage import MovePlanner

    move_planner = MovePlanner(robot)
    prepose_planner = PreposePlanner(
        robot=robot,
        manipulation_radius=0.8,
        enable_visualization=False,
    )
    grasp_planner = GraspPlanner(
        robot=robot,
        enable_visualization=False,
    )
    point_collector = ClickPointCollector()
    sam_client = SamClient(SamConfig())

    # Load static collision environment from config
    with open("grasp_anywhere/configs/real_fetch.yaml", "r") as f:
        config = yaml.safe_load(f)
    env_config = config.get("environment", {})
    pcd_path_base = env_config.get("pcd_path", "")
    pcd_files = env_config.get("pcd_files", [])

    collision_points = []
    if pcd_files:
        for pcd_file in pcd_files:
            pcd_path = os.path.join(pcd_path_base, pcd_file)
            try:
                import open3d as o3d

                pcd = o3d.io.read_point_cloud(pcd_path)
                pcd = pcd.voxel_down_sample(voxel_size=0.05)
                if len(pcd.points) > 0:
                    points = np.asarray(pcd.points)
                    if not collision_points:
                        collision_points = list(points)
                    else:
                        collision_points = np.vstack((collision_points, points))
            except Exception as e:
                print(f"Could not load {pcd_path}: {e}")
    collision_points = np.array(collision_points) if collision_points else None

    robot.clear_pointclouds()
    if collision_points is not None:
        robot.add_pointcloud(collision_points, filter_robot=True, point_radius=0.03)

    # 1) Go to table via MovePlanner.run interface
    print("Step 1: Moving to 'table' via MovePlanner.run using static map...")
    ok, msg = move_planner.run(
        "table",
        collision_points,
        enable_replanning=True,
        enable_pcd_alignment=False,
    )
    print(msg)
    if not ok:
        print("Move to table failed; exiting.")
        return
    time.sleep(0.5)

    # 2) Observe and ask human to point -> interactive segmentation
    print("Step 2: Observing and collecting interactive points for segmentation...")
    rgb = robot.get_rgb()
    depth = robot.get_depth()
    K = robot.get_camera_intrinsic()
    camera_extrinsic = robot.get_camera_pose()
    assert (
        rgb is not None
        and depth is not None
        and K is not None
        and camera_extrinsic is not None
    )

    points, labels = point_collector.collect_points(rgb)
    # Use the first point for SAM (SamClient currently wraps single point)
    if points and len(points) > 0:
        mask = sam_client.segment_point(rgb, points[0], labels[0])
    else:
        mask = None
        print("No points collected.")

    # Build object PCD in world and env PCD
    mask_bool = mask.astype(bool)
    object_pcd_cam = get_pcd_from_mask(depth, mask_bool, K)
    object_pcd_world = transform_points_camera_to_world(
        object_pcd_cam, camera_extrinsic
    )
    env_pcd_world = collision_points

    # 3) Prepose plan: take base config only
    print("Step 3: Planning pre-pose (base config only)...")
    config_generator, object_center_world = prepose_planner.plan(
        object_pcd_world, camera_extrinsic, env_pcd_world
    )
    if config_generator is None:
        print("Prepose planning failed; exiting.")
        return

    # Get first config from generator
    base_config, arm_config = None, None
    for base_config, arm_config in config_generator:
        break  # Just get the first one

    if base_config is None:
        print("Prepose planning failed; exiting.")
        return

    # 4) Move to the planned base_config with replanning while keeping current arm config unchanged
    print(
        "Step 4: Moving base to prepose base config with replanning (keeping arm unchanged)..."
    )
    current_joints_8dof = robot.get_current_planning_joints()
    if current_joints_8dof is None or len(current_joints_8dof) != 8:
        print("ERROR: Could not get current 8-DOF joints; exiting.")
        return
    ok, msg = move_to_config_with_replanning(
        robot=robot,
        goal_joints=current_joints_8dof,
        goal_base=list(map(float, base_config)),
        enable_replanning=True,
        enable_pcd_alignment=False,
        max_replan_attempts=5,
    )
    print(msg)
    if not ok:
        print("Move to prepose base config failed; exiting.")
        return
    if object_center_world is not None:
        robot.point_head_at(list(map(float, object_center_world)), frame_id="map")
    time.sleep(2.0)

    # 5) Re-observe and compute grasp poses using HTTP
    print("Step 5: Re-observing and computing grasp pose via GraspNet HTTP...")
    rgb = robot.get_rgb()
    depth = robot.get_depth()
    K = robot.get_camera_intrinsic()
    camera_extrinsic = robot.get_camera_pose()

    points, labels = point_collector.collect_points(rgb)
    if points and len(points) > 0:
        mask = sam_client.segment_point(rgb, points[0], labels[0])
    else:
        mask = None

    pred_grasps_cam, scores = _predict_grasps_http(rgb, depth, mask, K)
    if pred_grasps_cam is None or len(pred_grasps_cam) == 0:
        print("No grasps returned from GraspNet; exiting.")
        return

    # Choose the best-scored grasp
    best_idx = int(np.argsort(scores)[::-1][0])
    top_grasp_cam = pred_grasps_cam[best_idx]
    if top_grasp_cam.shape == (16,):
        top_grasp_cam = top_grasp_cam.reshape(4, 4)
    top_grasp_world = camera_extrinsic @ top_grasp_cam

    # 6) Execute grasp via GraspPlanner.run
    print("Step 6: Executing grasp...")
    success, msg = grasp_planner.run(
        top_grasp_world, camera_extrinsic, collision_points
    )
    print(f"Grasp result: {success}, msg: {msg}")


if __name__ == "__main__":
    main()
