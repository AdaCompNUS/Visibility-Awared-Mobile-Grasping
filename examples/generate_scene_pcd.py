#!/usr/bin/env python3
"""
Generate a scene point cloud map for the ManiSkill ReplicaCAD environment.
This map can be used for motion planning.
"""

import argparse

import gymnasium as gym
import numpy as np
import trimesh
from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh


def main():
    parser = argparse.ArgumentParser(description="Generate Scene Point Cloud")
    parser.add_argument(
        "--output", type=str, default="scene_map.ply", help="Output PLY file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument(
        "--num-points", type=int, default=200000, help="Number of points to sample"
    )
    args = parser.parse_args()

    # Initialize Environment (Headless)
    print(f"Initializing Environment (Seed: {args.seed})...")
    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        obs_mode="rgb+depth+segmentation",  # Mode doesn't strictly matter for mesh extraction
        render_mode=None,
    )
    env.reset(seed=args.seed)

    # Extract Meshes
    print("Extracting scene meshes...")
    scene_obj = env.unwrapped.scene
    stage_parts = []

    for actor in scene_obj.get_all_actors():
        name = getattr(actor, "name", "") or ""

        # We generally want the static scene elements (walls, floor, heavy furniture)
        # In ReplicaCAD, these are usually "scene_background" or have specific naming conventions.
        # For simplicity, we'll strip known dynamic objects if any, but in a fresh reset
        # without extra objects, the scene actors should be the static environment.
        # The 'fetch' robot actors should be excluded.

        if "fetch" in name.lower():
            continue

        m = get_actor_visual_mesh(actor)
        if m is None or m.is_empty:
            continue

        T = actor.pose.to_transformation_matrix()
        m.apply_transform(T)

        # Add to collection
        stage_parts.append(m)
        print(f"  Added actor: {name}")

    if not stage_parts:
        print("Error: No static meshes found in scene!")
        return

    # Concatenate Meshes
    print(f"Concatenating {len(stage_parts)} mesh parts...")
    full_mesh = trimesh.util.concatenate(stage_parts)

    # Convert to Open3D Mesh
    import open3d as o3d

    print("Converting to Open3D Mesh...")
    vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(full_mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()

    # Sample Points (Poisson Disk or FPS equivalent)
    # The user requested FPS behavior directly from mesh.
    # Poisson Disk Sampling provides uniformly distributed points on the surface,
    # similar to FPS but strictly better suitable for meshes.
    print(
        f"Sampling {args.num_points} points using Poisson Disk Sampling (approximating FPS)..."
    )
    pcd = o3d_mesh.sample_points_poisson_disk(
        number_of_points=args.num_points, init_factor=5
    )

    # Export
    print(f"Saving to {args.output}...")
    o3d.io.write_point_cloud(args.output, pcd)
    print("Done.")

    env.close()


if __name__ == "__main__":
    main()
