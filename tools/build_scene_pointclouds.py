#!/usr/bin/env python3
"""
Build complete scene pointclouds for benchmark testing.

This script extracts full environment pointclouds (stage + all objects) from ManiSkill
scenes and saves them for offline benchmark testing. This allows the prepose completeness
benchmark to run without depending on ManiSkill or the agent.

Usage:
    conda activate prepose_bench
    python tools/build_scene_pointclouds.py --benchmark resources/grasp_benchmark.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import uuid

import gymnasium as gym
import numpy as np
import open3d as o3d
import sapien
import tqdm
import trimesh
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh

# Fix library path to use conda environment's libraries
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}"
        # Re-exec the script with updated LD_LIBRARY_PATH
        os.execv(sys.executable, [sys.executable] + sys.argv)


def get_actor_mesh_safe(actor):
    """Safely extract mesh from actor, handling different API versions."""
    try:
        # Try the standard method first
        return get_actor_visual_mesh(actor)
    except AttributeError:
        # Fallback for older/newer API - manually extract from render component
        try:
            pass

            # Try to find render body component directly
            for comp in actor._components:
                if hasattr(comp, "render_shapes"):
                    meshes = []
                    for shape in comp.render_shapes:
                        # Extract geometry from render shape
                        if hasattr(shape, "geometry") and hasattr(
                            shape.geometry, "vertices"
                        ):
                            verts = np.array(shape.geometry.vertices)
                            faces = np.array(shape.geometry.indices).reshape(-1, 3)
                            if len(verts) > 0 and len(faces) > 0:
                                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                                meshes.append(mesh)
                    if meshes:
                        return trimesh.util.concatenate(meshes)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"    Warning: Could not extract mesh: {e}")
        return None


def extract_scene_pointcloud(
    env, seed, grasp_tasks, num_samples=100000, voxel_size=0.03
):
    """
    Extract complete scene pointcloud including stage and all objects.

    Args:
        env: ManiSkill environment
        seed: Scene seed
        grasp_tasks: List of grasp tasks containing object placements
        num_samples: Number of points to sample using Poisson disk sampling
        voxel_size: Voxel size for final downsampling (meters)

    Returns:
        np.ndarray: Scene pointcloud (N, 3)
    """
    # Reset environment to load the scene
    env.reset(seed=seed)
    scene_obj = env.unwrapped.scene

    # First, place all benchmark YCB objects in the scene
    for i, task in enumerate(grasp_tasks):
        model_id = task["model_id"]
        position = np.asarray(task["position"], dtype=np.float32).reshape(-1, 3)[0]
        orientation = np.asarray(task["orientation"], dtype=np.float32).reshape(-1, 4)[
            0
        ]

        builder = actors.get_actor_builder(scene_obj, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=position, q=orientation)
        actor_name = f"ycb_{model_id}_{uuid.uuid4().hex[:8]}"
        actor = builder.build(name=actor_name)

    # NOW extract ALL meshes from the scene (following reference code exactly)
    # Rule: actor name ending with "scene_background" => stage; else => object
    stage_parts = []
    object_parts = []
    for actor in scene_obj.get_all_actors():
        name = getattr(actor, "name", "") or ""
        m = get_actor_mesh_safe(actor)
        if m is None or m.is_empty:
            continue
        T = actor.pose.to_transformation_matrix()
        m.apply_transform(T)
        if name.lower().endswith("scene_background"):
            stage_parts.append(m)
        else:
            object_parts.append(m)

    # Combine all mesh parts (stage + objects)
    full_parts = []
    if len(stage_parts) > 0:
        full_parts.extend(stage_parts)
    if len(object_parts) > 0:
        full_parts.extend(object_parts)

    if len(full_parts) == 0:
        print("  Warning: No mesh parts found!")
        return np.empty((0, 3), dtype=np.float32)

    full_mesh = trimesh.util.concatenate(full_parts)

    # Convert trimesh to Open3D mesh
    vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(full_mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Use Poisson disk sampling for uniform surface distribution (following reference code)
    pcd = o3d_mesh.sample_points_poisson_disk(
        number_of_points=num_samples, init_factor=5
    )
    full_pts = np.asarray(pcd.points, dtype=np.float32)

    # Filter ground points (z >= 0.07 to remove floor properly)
    full_pts = full_pts[full_pts[:, 2] >= 0.07]

    # Additional voxel downsampling for final density control
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(full_pts)
    pcd_voxel = pcd_voxel.voxel_down_sample(voxel_size=voxel_size)
    full_pts = np.asarray(pcd_voxel.points, dtype=np.float32)

    print(
        f"  Sampled {full_pts.shape[0]} points from full scene mesh (stage + objects)"
    )

    return full_pts


def process_single_scene(args):
    """Process a single scene (for parallel execution)."""
    scene_id, scene_data, output_dir, num_samples, voxel_size = args

    output_path = os.path.join(output_dir, f"{scene_id}.ply")

    if os.path.exists(output_path):
        return f"Pointcloud for {scene_id} already exists, skipping."

    seed = scene_data.get("seed")
    grasp_tasks = scene_data.get("grasp_tasks", [])

    if not grasp_tasks:
        return f"Warning: No grasp tasks for {scene_id}, skipping."

    try:
        # Initialize environment for this process
        env = gym.make(
            "ReplicaCAD_SceneManipulation-v1",
            obs_mode="rgb+depth+segmentation",
            render_mode=None,  # Headless
        )

        # Extract full scene pointcloud
        scene_pcd = extract_scene_pointcloud(
            env, seed, grasp_tasks, num_samples=num_samples, voxel_size=voxel_size
        )

        env.close()

        if scene_pcd.shape[0] > 0:
            # Save as PLY file
            trimesh.points.PointCloud(scene_pcd).export(output_path)
            return f"{scene_id}: Saved {scene_pcd.shape[0]} points to {output_path}"
        else:
            return f"Warning: Empty pointcloud for {scene_id}"

    except Exception as e:
        import traceback

        return f"Error processing {scene_id}: {e}\n{traceback.format_exc()}"


def build_scene_pointclouds(
    benchmark_path, output_dir, num_samples=100000, voxel_size=0.03, num_workers=16
):
    """
    Build complete scene pointclouds for all scenes in benchmark (parallel).

    Args:
        benchmark_path: Path to grasp_benchmark.json
        output_dir: Directory to save scene pointclouds
        num_samples: Number of points for Poisson disk sampling
        voxel_size: Voxel size for downsampling (meters)
        num_workers: Number of parallel workers
    """
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    scene_args = [
        (scene_id, scene_data, output_dir, num_samples, voxel_size)
        for scene_id, scene_data in benchmark_data.items()
    ]

    print(f"Processing {len(scene_args)} scenes with {num_workers} parallel workers...")

    # Process scenes in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(process_single_scene, scene_args),
                total=len(scene_args),
                desc="Building Scene Pointclouds",
            )
        )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    for result in results:
        print(result)

    print("\nDone! All scene pointclouds have been generated.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Build complete scene pointclouds for benchmark testing"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="resources/grasp_benchmark.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resources/benchmark/scene_pointclouds",
        help="Output directory for scene pointclouds",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of points for Poisson disk sampling",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel size for final downsampling (meters)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    build_scene_pointclouds(
        args.benchmark,
        args.output_dir,
        num_samples=args.num_samples,
        voxel_size=args.voxel_size,
        num_workers=args.num_workers,
    )
