import os
import sys

from mani_skill.utils.building import actors

# --- PRE-IMPORT FIXES (MUST BE FIRST) ---
# LD_LIBRARY_PATH fix
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path.startswith(conda_lib):
        print(f"[Setup] Prepending {conda_lib} to LD_LIBRARY_PATH and re-executing...")
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[Setup] Failed to re-exec: {e}")

import gc
import json
from unittest.mock import MagicMock

import numpy as np
import sapien.core as sapien
from PIL import Image

# Mock pytorch_kinematics
mock_pk = MagicMock()
sys.modules["pytorch_kinematics"] = mock_pk
sys.modules["pytorch_kinematics.ik"] = mock_pk
sys.modules["pytorch_kinematics_ms"] = mock_pk

# Imports after env setup
try:
    import gymnasium as gym
    import trimesh.transformations as tra
except ImportError as e:
    print(f"[Error] Failed to import dependencies: {e}")
    sys.exit(1)


def get_camera_pose(target_pos, dist=2.0, pitch=-1.0, yaw=0.0):
    # Calculate Camera Position
    z = dist * np.sin(-pitch)
    xy = dist * np.cos(-pitch)
    x = xy * np.cos(yaw)
    y = xy * np.sin(yaw)
    cam_pos = target_pos + np.array([x, y, z])

    # Forward vector (Camera -> Target)
    # Robotics Convention: Camera looks down +X
    forward = target_pos - cam_pos
    forward /= np.linalg.norm(forward)

    # World Up
    world_up = np.array([0.0, 0.0, 1.0])

    # Handle Singularity (Looking Straight Down/Up)
    if abs(np.dot(forward, world_up)) > 0.99:
        # Looking straight down (-Z)
        # Global: Fwd = (0,0,-1)
        # We want Camera X = (0,0,-1)
        # ROTATED 90 DEGREES:
        # Cam Z (Up) aligned with World X instead of World Y
        
        # Hardcode the basis for looking down:
        cam_x = np.array([0.0, 0.0, -1.0])
        cam_z = np.array([1.0, 0.0, 0.0])
        cam_y = np.cross(cam_z, cam_x)
    else:
        # Normal LookAt logic for X-Forward
        # Fwd = X
        # Right = Cross(Fwd, WorldUp) = -Y (roughly)
        # Up (Z) = Cross(Fwd, Right) ? No.

        # Helper:
        # cam_x = forward
        # cam_y = cross(world_up, forward) (This is "Left" effectively, or Right depending on cross order)
        # Cross(Z, F) -> Right?
        # Let's compute a temp "Left" vector

        cam_x = forward

        # Left = WorldUp x Forward
        # (e.g. Up(0,0,1) x Fwd(1,0,0) = (0,1,0) Left?)
        left = np.cross(world_up, forward)
        left /= np.linalg.norm(left)

        # Local Z (Up) = Cross(X, Left) ? No. X(Fwd) x Y(Left) = Z(Up)
        cam_y = left
        cam_z = np.cross(cam_x, cam_y)

    mat44 = np.eye(4)
    mat44[:3, 0] = cam_x
    mat44[:3, 1] = cam_y
    mat44[:3, 2] = cam_z
    mat44[:3, 3] = cam_pos

    quat = tra.quaternion_from_matrix(mat44)
    pos = mat44[:3, 3]
    return sapien.Pose(p=pos, q=quat)


def main():
    benchmark_file = "resources/grasp_benchmark.json"
    output_dir = "outputs/benchmark_viz"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(benchmark_file):
        print(f"Benchmark file not found: {benchmark_file}")
        return

    with open(benchmark_file, "r") as f:
        data = json.load(f)

    scene_ids = sorted(list(data.keys()))
    try:
        scene_ids.sort(key=lambda x: int(x.split("_")[1]))
    except Exception:
        pass

    # target_scenes = scene_ids[:5]
    target_scenes = ["scene_0", "scene_1", "scene_2", "scene_3", "scene_5"]
    print(f"Visualizing scenes: {target_scenes}")

    for scene_id in target_scenes:
        if scene_id not in data:
            continue

        print(f"Processing {scene_id}...")

        # Create fresh environment for each scene to flush Vulkan resources
        # Use shader_dir="rt" directly as kwarg
        try:
            env = gym.make(
                "ReplicaCAD_SceneManipulation-v1",
                robot_uids="fetch",
                obs_mode="none",
                render_mode="rgb_array",
                shader_dir="rt",
            )
        except TypeError:
            # Fallback if shader_dir is not accepted by gym.make for some reason (rare)
            print("Warning: shader_dir not accepted, trying without")
            env = gym.make(
                "ReplicaCAD_SceneManipulation-v1",
                robot_uids="fetch",
                obs_mode="none",
                render_mode="rgb_array",
            )

        try:
            scene_data = data[scene_id]
            seed = scene_data.get("seed", 0)
            env.reset(seed=seed)

            # Hide robot
            # agent = env.unwrapped.agent
            # if agent is not None:
            #     agent.robot.set_pose(sapien.Pose(p=[-1000, -1000, 0]))

            scene = env.unwrapped.scene

            # Place Objects
            grasp_tasks = scene_data["grasp_tasks"]
            obj_positions = []

            print(f"  Adding {len(grasp_tasks)} objects...")
            for i, task in enumerate(grasp_tasks):
                model_id = task["model_id"]
                pos = np.array(task["position"])
                orn = np.array(task["orientation"])
                obj_positions.append(pos)

                builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
                builder.initial_pose = sapien.Pose(p=pos, q=orn)
                builder.build(name=f"obj_{i}")

                # Highlight using Marker only (removed point light to save descriptors)
                marker_pos = pos + np.array([0, 0, 0.25])
                m_builder = scene.create_actor_builder()
                # Emissive green marker
                m_builder.add_sphere_visual(radius=0.03, material=[0.0, 1.0, 0.0, 1.0])
                m_builder.initial_pose = sapien.Pose(p=marker_pos)
                m_builder.build_static(name=f"marker_{i}")

            # Camera Pose
            # Use fixed center for all scenes to keep camera position constant
            # The scenes are roughly in x=[0, 4] and y=[-8, -1]
            center = np.array([2.0, -4.5, 0.0])
            target_cam_z = 6.0  # Higher to see the whole area
            cam_dist = target_cam_z - center[2]

            pose = get_camera_pose(
                center, dist=cam_dist, pitch=np.deg2rad(-90), yaw=0
            )

            # Add Camera
            cam_name = "viz_cam"
            camera = None

            if hasattr(scene, "add_camera"):
                camera = scene.add_camera(
                    name=cam_name,
                    width=1920,
                    height=1080,
                    fovy=np.deg2rad(60),
                    near=0.1,
                    far=100,
                    pose=pose,
                )
            else:
                s = getattr(scene, "scene", getattr(scene, "_scene", None))
                if s and hasattr(s, "add_camera"):
                    camera = s.add_camera(
                        name=cam_name,
                        width=1920,
                        height=1080,
                        fovy=np.deg2rad(60),
                        near=0.1,
                        far=100,
                        pose=pose,
                    )

            if camera is None:
                print(
                    f"[Error] Could not find add_camera on scene object ({type(scene)}). Skipping."
                )
                continue

            # Global Lighting only
            scene.set_ambient_light([0.5, 0.5, 0.5])
            scene.add_directional_light([1, 1, -1], [0.9, 0.9, 0.9], shadow=True)
            scene.add_directional_light([-1, -0.5, -1], [0.5, 0.5, 0.6], shadow=True)

            scene.update_render()
            camera.take_picture()

            rgba = camera.get_picture("Color")
            if hasattr(rgba, "cpu"):
                rgba = rgba.cpu().numpy()
            elif isinstance(rgba, list):
                # If list contains tensors (on cuda), move them to cpu
                if len(rgba) > 0 and hasattr(rgba[0], "cpu"):
                    rgba = [x.cpu().numpy() for x in rgba]
                rgba = np.array(rgba)

            # Robustly squeeze extra batch dimensions
            while rgba.ndim > 3 and rgba.shape[0] == 1:
                rgba = rgba[0]

            # Ensure we have (H, W, C)
            if rgba.ndim != 3:
                print(f"[Warning] Unexpected rgba shape: {rgba.shape}")

            rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            save_path = os.path.join(output_dir, f"{scene_id}_rt.png")
            img.save(save_path)
            print(f"Saved visualization to {save_path}")

        except Exception as e:
            print(f"[Error] Failed processing {scene_id}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            env.close()
            gc.collect()


if __name__ == "__main__":
    main()
