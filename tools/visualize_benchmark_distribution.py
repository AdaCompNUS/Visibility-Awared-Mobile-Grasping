import glob
import os
import sys

# --- PRE-IMPORT FIXES ---
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

try:
    import gymnasium as gym
    import trimesh
    import trimesh.transformations as tra
    from mani_skill.utils.building import actors
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

    forward = target_pos - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(forward, world_up)) > 0.99:
        cam_x = np.array([0.0, 0.0, -1.0])
        cam_z = np.array([1.0, 0.0, 0.0])
        cam_y = np.cross(cam_z, cam_x)
    else:
        cam_x = forward
        left = np.cross(world_up, forward)
        left /= np.linalg.norm(left)
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
    # Find latest results file
    # Assuming we are running from the root of grasp_anywhere
    # Adjust paths if necessary

    # Try absolute path first if relative fails, or assume CWD is root
    results_dir = "results"
    full_results_dir = os.path.abspath(results_dir)

    if not os.path.exists(full_results_dir):
        # Fallback to absolute path derived from file location if running from tools
        # But usually we run from root.
        pass

    results_files = glob.glob(
        os.path.join(full_results_dir, "benchmark_results_*.json")
    )
    if not results_files:
        print(f"No results found in {full_results_dir}")
        return

    # Sort by time
    results_files.sort(key=os.path.getmtime, reverse=True)
    results_path = results_files[0]
    print(f"Using results file: {results_path}")

    benchmark_path = "resources/grasp_benchmark.json"
    output_dir = "outputs/benchmark_viz_distribution"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(benchmark_path):
        print(f"Benchmark file not found: {benchmark_path}")
        return

    with open(results_path, "r") as f:
        results_data = json.load(f)
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    # Filter scenes that are in results
    if "scenes" not in results_data:
        print("Invalid results format: missing 'scenes' key")
        return

    scene_ids = sorted(list(results_data["scenes"].keys()))

    # Try to sort naturally if scene_X
    try:
        scene_ids.sort(key=lambda x: int(x.split("_")[1]))
    except Exception:
        pass

    print(f"Visualizing scenes: {scene_ids}")

    for scene_id in scene_ids:
        print(f"Processing {scene_id}...")

        # Initialize Env with RT
        try:
            env = gym.make(
                "ReplicaCAD_SceneManipulation-v1",
                robot_uids="fetch",
                obs_mode="none",
                render_mode="rgb_array",
                shader_dir="rt",
            )
        except TypeError:
            print("Warning: shader_dir not accepted, trying without")
            env = gym.make(
                "ReplicaCAD_SceneManipulation-v1",
                robot_uids="fetch",
                obs_mode="none",
                render_mode="rgb_array",
            )

        try:
            # Check if scene exists in benchmark data
            if scene_id not in benchmark_data:
                print(f"Warning: {scene_id} in results but not in benchmark file.")
                continue

            scene_layout = benchmark_data[scene_id]
            seed = scene_layout.get("seed", 0)
            env.reset(seed=seed)

            # Hide robot (move far away)
            agent = env.unwrapped.agent
            if agent is not None:
                agent.robot.set_pose(sapien.Pose(p=[-1000, -1000, 0]))

            scene = env.unwrapped.scene

            # Get Results for this scene
            scene_results = results_data["scenes"][scene_id]
            task_results_map = {
                t["task_id"]: t for t in scene_results["tasks"]
            }  # assumption on structure

            grasp_tasks = scene_layout["grasp_tasks"]

            print(f"  Adding {len(grasp_tasks)} objects...")
            for i, task in enumerate(grasp_tasks):
                model_id = task["model_id"]
                pos = np.array(task["position"])
                orn = np.array(task["orientation"])

                builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
                builder.initial_pose = sapien.Pose(p=pos, q=orn)
                builder.build(name=f"ycb_{model_id}_{i}")

                # Determine Marker Color based on Result
                color = [0.6, 0.6, 0.6, 1.0]  # Default Grey

                # Check result
                result = task_results_map.get(i)
                if not result:
                    result = task_results_map.get(str(i))

                if result:
                    if result["success"]:
                        color = [0.1, 0.9, 0.1, 1.0]  # Bright Green
                    else:
                        color = [0.9, 0.1, 0.1, 1.0]  # Bright Red

                # Continuous ring marker around object using torus mesh
                ring_major_radius = 0.15  # Distance from object center
                ring_minor_radius = 0.03  # Thickness of the ring tube
                ring_height = pos[2] + 0.02  # Slightly above ground

                # Create torus mesh
                torus_mesh = trimesh.creation.torus(
                    major_radius=ring_major_radius,
                    minor_radius=ring_minor_radius,
                    major_sections=64,
                    minor_sections=16,
                )

                # Save to temporary file
                temp_mesh_path = f"/tmp/ring_mesh_{i}.obj"
                torus_mesh.export(temp_mesh_path)

                # Create emissive material - flat color unaffected by lighting
                render_mat = sapien.render.RenderMaterial()
                render_mat.base_color = [0.0, 0.0, 0.0, 1.0]  # Dark base
                render_mat.emission = color  # Emissive glow with result color (RGBA)

                # Add ring as visual
                m_builder = scene.create_actor_builder()
                m_builder.add_visual_from_file(temp_mesh_path, material=render_mat)
                ring_pos = [pos[0], pos[1], ring_height]
                m_builder.initial_pose = sapien.Pose(p=ring_pos)
                m_builder.build_static(name=f"ring_{i}")

            # Camera Pose (Top Down) - Using fixed center from visualize_benchmark.py
            center = np.array([2.0, -4.5, 0.0])
            target_cam_z = 6.0
            cam_dist = target_cam_z - center[2]

            pose = get_camera_pose(center, dist=cam_dist, pitch=np.deg2rad(-90), yaw=0)

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
                print("[Error] Could not find add_camera")
                continue

            # Global Lighting - single top-down light
            scene.set_ambient_light([0.3, 0.3, 0.3])
            scene.add_directional_light([0, 0, -1], [1.0, 1.0, 1.0], shadow=True)

            scene.update_render()
            camera.take_picture()

            rgba = camera.get_picture("Color")
            if hasattr(rgba, "cpu"):
                rgba = rgba.cpu().numpy()
            elif isinstance(rgba, list):
                if len(rgba) > 0 and hasattr(rgba[0], "cpu"):
                    rgba = [x.cpu().numpy() for x in rgba]
                rgba = np.array(rgba)

            while rgba.ndim > 3 and rgba.shape[0] == 1:
                rgba = rgba[0]

            rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            save_path = os.path.join(output_dir, f"{scene_id}_distribution_rt.png")
            img.save(save_path)
            print(f"Saved visualization to {save_path}")

        except Exception as e:
            print(f"[Error] processing {scene_id}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            env.close()
            gc.collect()


if __name__ == "__main__":
    main()
