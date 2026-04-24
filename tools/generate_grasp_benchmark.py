import argparse
import json
import os
import random
import shutil
import uuid

import gymnasium as gym
import numpy as np
import sapien
import tqdm
import trimesh
from grasp_oracle import GraspOracleCfg, oracle_is_graspable
from grasp_pipeline import GraspPipelineCfg, run_grasp_pipeline_for_actor
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh
from scipy.spatial import ConvexHull

from grasp_anywhere.utils.logger import log

# Set GRASPNET_URL env var or edit this default before running.
GRASP_SERVICE_URL = os.environ.get("GRASPNET_URL", "http://localhost:4003")
GRASP_PIPELINE_CFG = GraspPipelineCfg()
GRASP_ORACLE_CFG = GraspOracleCfg()

# A list of YCB objects to be used in the benchmark
YCB_OBJECTS = [
    "002_master_chef_can",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "061_foam_brick",
    "063-a_marbles",
    "063-b_marbles",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "072-c_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "077_rubiks_cube",
]


def _body_actor_name(body: object) -> str:
    """
    Match `tools/grasp_oracle.py` naming behavior: contacts expose `bodies[*]` which may
    carry names either directly or via `entity.name`.
    """
    name = getattr(body, "name", "") or ""
    if not name and hasattr(body, "entity"):
        name = getattr(body.entity, "name", "") or ""
    return name


def _snapshot_scene(scene: object, *, exclude_name_substr: str = ""):
    """
    Snapshot actor poses so we can restore the environment after failed spawn attempts.
    Mirrors the oracle's snapshot/restore strategy.
    """
    excl = (exclude_name_substr or "").lower()
    snap = []
    for a in scene.get_all_actors():
        name = (getattr(a, "name", "") or "").lower()
        if excl and excl in name:
            continue
        snap.append((a, a.pose))
    return snap


def _restore_scene_pose(snap) -> None:
    for a, p in snap:
        a.set_pose(p)


def _has_actor_env_collision(scene: object, actor_name: str) -> bool:
    """
    Returns True if the given actor is currently in contact with anything not itself.
    Used right after spawn (before the drop) to detect penetrations/invalid placements.
    """
    t = (actor_name or "").lower()
    for c in scene.get_contacts():
        a0 = _body_actor_name(c.bodies[0]).lower()
        a1 = _body_actor_name(c.bodies[1]).lower()
        # Ignore self-collisions (if any)
        if t and (t in a0) and (t in a1):
            continue
        if t and ((t in a0) != (t in a1)):
            return True
    return False


class BenchmarkGenerator:
    """
    A class to generate a mobile grasping benchmark in ReplicaCAD scenes.
    """

    def __init__(
        self, output_path, num_scenes, num_objects_per_scene, render_mode="human"
    ):
        self.output_path = output_path
        self.num_scenes = num_scenes
        self.num_objects_per_scene = num_objects_per_scene
        self.render_mode = render_mode
        self.benchmark_data = {}

    def generate(self):
        """
        Generates the benchmark data.
        """
        # Remove dependency on local ReplicaCAD dataset directories. We rely on seeds
        # to select scenes deterministically inside the simulator.
        if self.num_scenes is None or self.num_scenes <= 0:
            seeds_to_process = list[int](range(1))
        else:
            seeds_to_process = list[int](range(self.num_scenes))

        # Create a single environment to be reused
        env = gym.make(
            "ReplicaCAD_SceneManipulation-v1",
            obs_mode="rgb+depth+segmentation",
            render_mode=self.render_mode,
            # robot_uids="fetch",
        )

        for seed in tqdm.tqdm(seeds_to_process, desc="Processing scenes"):
            scene_id = f"scene_{seed}"
            log.info(f"Processing scene: {scene_id} (seed={seed})")
            self.process_scene(env, seed, scene_id)

        env.close()

    def process_scene(self, env, seed, scene_id):
        """
        Processes a single scene to find placement locations and generate grasp tasks.
        """
        # Reset the environment to the new scene using the seed
        env.reset(seed=seed)

        # Build separated world-space meshes for stage vs objects by name rule
        # Rule: actor name ending with "scene_background" => stage; else => object
        scene_obj = env.unwrapped.scene
        stage_parts = []
        object_parts = []
        for actor in scene_obj.get_all_actors():
            name = getattr(actor, "name", "") or ""
            # log.info(f"Actor name: {name}")
            m = get_actor_visual_mesh(actor)
            if m is None or m.is_empty:
                continue
            T = actor.pose.to_transformation_matrix()
            m.apply_transform(T)
            if name.lower().endswith("scene_background"):
                stage_parts.append(m)
            else:
                object_parts.append(m)

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        debug_dir = os.path.join(project_root, "debug", "placement_surfaces", scene_id)
        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir)
        # if len(stage_parts) > 0:
        #     canonical_dir = os.path.join(
        #         project_root, "resources", "benchmark", "canonical_maps"
        #     )
        #     os.makedirs(canonical_dir, exist_ok=True)
        #     canonical_map_path = os.path.join(canonical_dir, f"{scene_id}.ply")
        #     if not os.path.exists(canonical_map_path):
        #         stage_mesh = trimesh.util.concatenate(stage_parts)
        #         voxel_size = 0.08
        #         vg = stage_mesh.voxelized(voxel_size)
        #         stage_pts = vg.points.astype(np.float32)
        #         stage_pts = stage_pts[stage_pts[:, 2] >= 0.05]
        #         trimesh.points.PointCloud(stage_pts.astype(np.float32)).export(
        #             canonical_map_path
        #         )
        scene_mesh = trimesh.util.concatenate(object_parts)
        # scene_mesh = trimesh.util.concatenate(stage_parts + object_parts)

        full_parts = []
        if len(stage_parts) > 0:
            full_parts.extend(stage_parts)
        if len(object_parts) > 0:
            full_parts.extend(object_parts)
        # if len(full_parts) > 0:
        #     full_mesh = trimesh.util.concatenate(full_parts)
        #     # Convert trimesh to Open3D mesh
        #     vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
        #     triangles = np.asarray(full_mesh.faces, dtype=np.int32)
        #     o3d_mesh = o3d.geometry.TriangleMesh()
        #     o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        #     o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        #     # Use Poisson disk sampling for uniform surface distribution in one step
        #     num_samples = 100000
        #     pcd = o3d_mesh.sample_points_poisson_disk(
        #         number_of_points=num_samples, init_factor=5
        #     )
        #     full_pts = np.asarray(pcd.points, dtype=np.float32)
        #     log.info(
        #         f"Poisson disk sampled {full_pts.shape[0]} points from mesh surface"
        #     )
        #     project_root = os.path.abspath(
        #         os.path.join(os.path.dirname(__file__), os.pardir)
        #     )
        #     full_dir = os.path.join(project_root, "resources", "benchmark", "full_maps")
        #     os.makedirs(full_dir, exist_ok=True)
        #     full_map_path = os.path.join(full_dir, f"{scene_id}.ply")
        #     trimesh.points.PointCloud(full_pts).export(full_map_path)

        log.info(f"Finding placement surfaces for scene {scene_id}...")
        placement_surfaces = self.find_placement_surfaces(
            scene_mesh, debug_dir=debug_dir
        )
        log.info(
            f"Found {len(placement_surfaces)} placement surfaces for scene {scene_id}."
        )

        if not placement_surfaces:
            log.warning(f"No suitable placement surfaces found for scene {scene_id}")
            return

        # No debug file exports

        grasp_tasks = []
        while len(grasp_tasks) < self.num_objects_per_scene:
            model_id = random.choice(YCB_OBJECTS)
            log.info(
                f"Attempting to place object {len(grasp_tasks)+1}/{self.num_objects_per_scene}: {model_id}"
            )

            # Use sequential reset by passing the seed and already placed tasks
            actor = self.place_object_in_sim(
                env, seed, model_id, placement_surfaces, grasp_tasks
            )

            if actor is not None:
                # Freeze pose immediately after placement verification (before any
                # further simulation steps in grasp pipeline / oracle checks).
                pose_now = actor.pose
                pos_now = np.asarray(pose_now.p, dtype=np.float32).reshape(3).copy()
                quat_now = np.asarray(pose_now.q, dtype=np.float32).reshape(4).copy()
                grasps_world, scores, _, _ = run_grasp_pipeline_for_actor(
                    env=env,
                    actor=actor,
                    grasp_service_url=GRASP_SERVICE_URL,
                    cfg=GRASP_PIPELINE_CFG,
                )
                graspable = oracle_is_graspable(
                    env=env,
                    actor=actor,
                    grasp_poses_world=grasps_world,
                    cfg=GRASP_ORACLE_CFG,
                )
                if not graspable:
                    log.warning(f"Oracle check failed (ungraspable): {model_id}")
                    actor.remove_from_scene()
                    continue

                print("Successfully placed object")
                grasp_tasks.append(
                    {
                        "model_id": model_id,
                        "position": pos_now.tolist(),
                        "orientation": quat_now.tolist(),
                    }
                )

        canonical_map_path = f"resources/benchmark/canonical_maps/{scene_id}.ply"
        entry = {
            "seed": seed,
            "grasp_tasks": grasp_tasks,
            "canonical_map_path": canonical_map_path,
        }
        self.benchmark_data[scene_id] = entry
        log.info(f"Generated {len(grasp_tasks)} grasp tasks for scene {scene_id}.")

    def place_object_in_sim(
        self, env, seed, model_id, placement_surfaces, grasp_tasks, max_trials=100
    ):
        """
        Tries to place a YCB object in the scene at a stable pose without collisions.
        Returns the final pose if successful, otherwise None.

        Args:
            env: The simulation environment
            seed: The scene seed for resetting
            model_id: The YCB object model ID
            placement_surfaces: List of available placement surface point clouds
            grasp_tasks: List of objects already successfully placed
            max_trials: Maximum number of placement attempts
        """
        min_distance_between_objects = (
            0.20  # Minimum 20cm distance between object centers
        )
        placed_object_positions = [np.array(t["position"]) for t in grasp_tasks]

        for _ in range(max_trials):
            # Reset environment to ensure a clean state for every trial
            env.reset(seed=seed)
            scene = env.unwrapped.scene

            # Re-spawn all previously successful objects
            for obj_idx, task in enumerate(grasp_tasks):
                builder = actors.get_actor_builder(
                    env.scene, id=f"ycb:{task['model_id']}"
                )
                builder.initial_pose = sapien.Pose(
                    p=task["position"], q=task["orientation"]
                )
                builder.build(
                    name=f"ycb_{task['model_id']}_prev_{obj_idx}_{uuid.uuid4().hex}"
                )

            # 1. Sample a plane cluster point and a preliminary pose (Z-up)
            selected_plane_pts = random.choice(placement_surfaces)
            point = selected_plane_pts[
                np.random.randint(selected_plane_pts.shape[0])
            ].reshape(1, 3)
            target_plane_z = float(selected_plane_pts[:, 2].mean())

            # 2. Check if this point is too close to any existing object
            point_position = point[0].astype(np.float32)
            too_close = False
            for existing_pos in placed_object_positions:
                distance = np.linalg.norm(point_position - existing_pos)
                if distance < min_distance_between_objects:
                    too_close = True
                    break

            if too_close:
                continue  # Try another placement location

            z_angle = random.uniform(0, 2 * np.pi)
            orientation = trimesh.transformations.quaternion_from_euler(0, 0, z_angle)

            # 3. Load the object into the scene
            builder = actors.get_actor_builder(env.scene, id=f"ycb:{model_id}")
            # Add a small offset to avoid initial penetration (raise along Z)
            placement_position = point[0].astype(np.float32)
            # Keep x,y equal to the sampled point; raise z slightly above the surface
            placement_position[2] += np.float32(0.2)
            placement_orientation = np.asarray(orientation, dtype=np.float32)
            #
            builder.initial_pose = sapien.Pose(
                p=placement_position, q=placement_orientation
            )
            actor_name = f"ycb_{model_id}_{uuid.uuid4().hex}"
            #
            actor = builder.build(name=actor_name)

            # 4. Collision check right after spawn
            env.step(None)
            if self.render_mode is not None:
                env.render()
            if _has_actor_env_collision(scene, actor_name):
                log.warning(f"Spawn collision detected for {actor_name}")
                continue

            # 5. Let the simulation settle
            initial_pos = np.asarray(actor.pose.p).copy()
            for _ in range(100):
                env.step(None)
                if self.render_mode is not None:
                    env.render()

            # 6. Check for validity relative to the sampled support plane
            pose_p_arr = np.asarray(actor.pose.p)
            final_z = float(pose_p_arr.reshape(-1, 3)[0, 2])
            lower_ok = final_z >= (target_plane_z - 0.05)
            upper_ok = final_z <= (target_plane_z + 0.3)
            if not (lower_ok and upper_ok):
                log.warning(
                    f"Height check failed: final_z={final_z}, target_plane_z={target_plane_z}"
                )
                continue

            # 7. Check stability: measure movement after additional simulation steps
            final_pos = np.asarray(actor.pose.p)
            movement = np.linalg.norm(final_pos - initial_pos)

            if movement > 0.3:  # Reject if object moved more than 30cm
                log.warning(f"Stability check failed: object moved {movement:.3f}m")
                continue

            return actor

        log.warning(f"Failed to place object {model_id} after {max_trials} trials.")
        return None

    def find_placement_surfaces(
        self,
        scene_mesh,
        min_height=0.1,
        max_height=1.7,
        min_area=0.04,
        height_bin_size=0.02,
        debug_dir=None,
    ):
        """
        Point-cloud-based plane detection (Z-up after Y->Z conversion).
        Returns a list of plane point clusters (np.ndarray of shape [N,3]).
        Only the top surface for each horizontal support footprint is kept.
        """
        scene_points = 200000
        up_dot = 0.95
        grid_size = 0.05
        min_cluster_points = 500

        pts, face_ids = trimesh.sample.sample_surface(scene_mesh, scene_points)
        pts = pts.astype(np.float32)
        face_normals = scene_mesh.face_normals
        normals = face_normals[np.asarray(face_ids, dtype=np.int64)]

        up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        height_idx = 2
        proj_idx = [0, 1]

        dot_up = (normals * up_vec.reshape(1, 3)).sum(axis=1)
        keep = dot_up > up_dot
        pts_keep = pts[keep]
        if pts_keep.shape[0] == 0:
            return []

        proj = pts_keep[:, proj_idx]
        grid = np.floor(proj / grid_size).astype(np.int64)
        height_bins = np.floor(pts_keep[:, height_idx] / height_bin_size).astype(
            np.int64
        )
        cell_to_indices = {}
        for i, cell in enumerate(grid):
            key = (int(cell[0]), int(cell[1]), int(height_bins[i]))
            if key in cell_to_indices:
                cell_to_indices[key].append(i)
            else:
                cell_to_indices[key] = [i]

        visited = set()
        clusters = []
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        height_neighbors = [-1, 0, 1]
        for cell in list(cell_to_indices.keys()):
            if cell in visited:
                continue
            queue = [cell]
            visited.add(cell)
            cluster_cells = [cell]
            while queue:
                cx, cy, cz = queue.pop()
                for dx, dy in neighbors:
                    for dz in height_neighbors:
                        nc = (cx + dx, cy + dy, cz + dz)
                        if nc in cell_to_indices and nc not in visited:
                            visited.add(nc)
                            queue.append(nc)
                            cluster_cells.append(nc)

            idxs = []
            footprint = set()
            for cc in cluster_cells:
                idxs.extend(cell_to_indices[cc])
                footprint.add((cc[0], cc[1]))
            if len(idxs) < min_cluster_points:
                continue
            clusters.append(
                {
                    "indices": np.asarray(idxs, dtype=np.int64),
                    "footprint": footprint,
                }
            )

        plane_clusters = []
        for cl in clusters:
            cluster_pts = pts_keep[cl["indices"]]
            avg_h = float(cluster_pts[:, height_idx].mean())
            pts_2d = cluster_pts[:, proj_idx]
            area = 0.0
            if pts_2d.shape[0] >= 3:
                centered = pts_2d - pts_2d.mean(axis=0, keepdims=True)
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                if s.shape[0] >= 2 and s[1] > 1e-8:
                    hull = ConvexHull(pts_2d)
                    area = float(hull.volume)
            if area >= min_area and avg_h >= min_height and avg_h <= max_height:
                plane_clusters.append(
                    {
                        "points": cluster_pts.astype(np.float32),
                        "avg_h": avg_h,
                        "footprint": cl["footprint"],
                    }
                )

        if plane_clusters:
            stats = [
                (
                    float(info["points"][:, height_idx].min()),
                    float(info["points"][:, height_idx].max()),
                    float(info["avg_h"]),
                    int(info["points"].shape[0]),
                )
                for info in plane_clusters
            ]
            log.info(
                "Placement cluster heights (min, max, mean, count): %s",
                stats,
            )

        # Compute per-footprint max average height across detected plane clusters only
        footprint_max_h = {}
        for info in plane_clusters:
            avg_h = info["avg_h"]
            for coord in info["footprint"]:
                if coord in footprint_max_h:
                    if avg_h > footprint_max_h[coord]:
                        footprint_max_h[coord] = avg_h
                else:
                    footprint_max_h[coord] = avg_h

        top_clusters = []
        height_epsilon = height_bin_size * 1.5
        for info in plane_clusters:
            avg_h = info["avg_h"]
            coords = list(info["footprint"]) if info["footprint"] else []
            if not coords:
                continue
            num_top = 0
            for coord in coords:
                if avg_h >= footprint_max_h[coord] - height_epsilon:
                    num_top += 1
            ratio = float(num_top) / float(len(coords))
            if ratio >= 0.6:
                top_clusters.append(info["points"])

        if debug_dir is not None:
            all_dir = os.path.join(debug_dir, "all_surfaces")
            top_dir = os.path.join(debug_dir, "top_surfaces")
            os.makedirs(all_dir, exist_ok=True)
            os.makedirs(top_dir, exist_ok=True)
            for idx, info in enumerate(plane_clusters):
                export_path = os.path.join(all_dir, f"surface_{idx:03d}.ply")
                trimesh.points.PointCloud(info["points"]).export(export_path)
            for idx, pts in enumerate(top_clusters):
                export_path = os.path.join(top_dir, f"surface_{idx:03d}.ply")
                trimesh.points.PointCloud(pts).export(export_path)

        return top_clusters

    def save_benchmark(self):
        """
        Saves the benchmark data to a JSON file.
        """
        log.info(f"Saving benchmark to {self.output_path}")
        with open(self.output_path, "w") as f:
            json.dump(self.benchmark_data, f, indent=4)


def visualize_benchmark(benchmark_path):
    """
    Visualizes a benchmark from a JSON file.
    """
    log.info(f"Visualizing benchmark from {benchmark_path}")

    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        obs_mode="rgb",
        render_mode="human",
    )

    for scene_id, scene_data in benchmark_data.items():
        log.info(f"Visualizing scene: {scene_id}")

        seed = scene_data.get("seed")
        log.info(f"Visualizing scene: {scene_id} (seed={seed})")

        # Load the correct scene using the seed
        env.reset(seed=seed)
        viewer = env.render()

        for grasp_task in scene_data["grasp_tasks"]:
            model_id = grasp_task["model_id"]
            position = grasp_task["position"]
            orientation = grasp_task["orientation"]

            builder = actors.get_actor_builder(env.scene, id=f"ycb:{model_id}")
            p = np.asarray(position, dtype=np.float32).reshape(-1, 3)[0]
            q = np.asarray(orientation, dtype=np.float32).reshape(-1, 4)[0]
            builder.initial_pose = sapien.Pose(p=p, q=q)
            actor_name = f"ycb_{model_id}_{uuid.uuid4().hex}"
            log.info(f"Visualize: building actor with name: {actor_name}")
            builder.build(name=actor_name)

        log.info("Press 'q' to go to the next scene.")
        while True:
            env.step(None)
            env.render()
            if viewer.window.key_press("q"):
                break
    env.close()


def main():
    """
    Main function to generate or visualize the benchmark.
    """
    parser = argparse.ArgumentParser(
        description="Generate or visualize a mobile grasping benchmark."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new benchmark.",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        metavar="PATH",
        help="Visualize an existing benchmark from a JSON file.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help='ManiSkill render mode for benchmark generation. Use "None" to disable rendering.',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="resources/grasp_benchmark.json",
        help="Path to save the generated benchmark file.",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=20,
        help="Number of scenes to include in the benchmark.",
    )
    parser.add_argument(
        "--num_objects_per_scene",
        type=int,
        default=20,
        help="Number of objects to place in each scene.",
    )
    args = parser.parse_args()
    render_mode = args.render_mode
    if render_mode == "None":
        render_mode = None

    if args.generate:
        generator = BenchmarkGenerator(
            output_path=args.output_path,
            num_scenes=args.num_scenes,
            num_objects_per_scene=args.num_objects_per_scene,
            render_mode=render_mode,
        )
        generator.generate()
        generator.save_benchmark()
    elif args.visualize:
        visualize_benchmark(args.visualize)
    else:
        log.error("Please specify either --generate or --visualize.")


if __name__ == "__main__":
    main()
