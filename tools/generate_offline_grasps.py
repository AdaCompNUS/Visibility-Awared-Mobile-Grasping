#!/usr/bin/env python3
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import requests
import sapien
import trimesh
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh

from grasp_anywhere.utils.visualization_utils import visualize_grasps_pcd

# YCB objects from benchmark
YCB_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
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
    "036_wood_block",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
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
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "077_rubiks_cube",
]


def sample_rotations(n_rotations):
    """Sample uniform random rotations on SO(3) (deterministic seed)."""

    def _quat_to_R(x, y, z, w):
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    rng = np.random.RandomState(0)
    rotations = []
    for _ in range(n_rotations):
        # Uniform random quaternion (Hopf fibration)
        u1, u2, u3 = rng.rand(3)
        qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        rotations.append(_quat_to_R(qx, qy, qz, qw))

    return rotations


def load_ycb_mesh(env, model_id):
    """Load YCB object mesh from environment."""
    scene_obj = env.unwrapped.scene
    builder = actors.get_actor_builder(scene_obj, id=f"ycb:{model_id}")
    builder.initial_pose = sapien.Pose(p=[0, 0, 0])
    temp_actor = builder.build(name=f"temp_{model_id}")

    # ManiSkill v3 returns a wrapper Actor; the underlying sapien entity is in _objs[0]
    entity = temp_actor._objs[0]
    mesh = get_actor_visual_mesh(entity)
    T = entity.pose.to_transformation_matrix()
    mesh.apply_transform(T)

    temp_actor.remove_from_scene()
    return mesh


def predict_grasps_from_pointcloud(object_pts, service_url):
    """Send point cloud to grasping service and get grasp predictions."""
    payload = {"object_pts": object_pts.tolist()}

    url = service_url + "/sample_grasp_from_pointcloud"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    pred_grasps = np.array(result["pred_grasps_cam"]).reshape(-1, 4, 4)
    scores = np.array(result["scores"]).reshape(-1)

    return pred_grasps, scores


def farthest_point_sampling(grasps, n_samples):
    """Sample n_samples grasps using farthest point sampling based on position."""
    positions = grasps[:, :3, 3]

    selected = np.empty((n_samples,), dtype=np.int64)
    selected[0] = np.random.randint(len(positions))

    # min distance to the selected set so far
    d2 = np.sum((positions - positions[selected[0]]) ** 2, axis=1)
    for i in range(1, n_samples):
        selected[i] = int(np.argmax(d2))
        new_d2 = np.sum((positions - positions[selected[i]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return selected


def generate_grasps_for_object(
    env, model_id, service_url, num_rotations, num_samples, score_threshold
):
    """Generate grasps for a single object with rotation augmentation."""
    print(f"Generating grasps for {model_id}...")

    mesh = load_ycb_mesh(env, model_id)
    object_pts_original = trimesh.sample.sample_surface(mesh, num_samples)[0].astype(
        np.float32
    )
    rotations = sample_rotations(num_rotations)

    all_grasps = []
    all_scores = []

    for i, R in enumerate(rotations):
        print(f"  Rotation {i+1}/{num_rotations}")

        object_pts_rotated = (R @ object_pts_original.T).T
        grasps_rotated, scores = predict_grasps_from_pointcloud(
            object_pts_rotated, service_url
        )

        mask = scores >= score_threshold
        grasps_rotated = grasps_rotated[mask]
        scores = scores[mask]

        R_inv_4x4 = np.eye(4)
        R_inv_4x4[:3, :3] = R.T

        # Keep shape stable even if 0 grasps survive thresholding (=> (0, 4, 4))
        grasps_original = np.einsum("ij,njk->nik", R_inv_4x4, grasps_rotated)
        all_grasps.append(grasps_original)
        all_scores.append(scores)

    all_grasps = np.vstack(all_grasps)
    all_scores = np.concatenate(all_scores)

    # Prefer high score grasps, then make them spatially uniform via FPS
    order = np.argsort(all_scores)[::-1]
    order = order[:9216]
    selected_indices = farthest_point_sampling(all_grasps[order], 256)
    final_grasps = all_grasps[order][selected_indices]
    final_scores = all_scores[order][selected_indices]

    return final_grasps, final_scores, object_pts_original


def main():
    service_url = os.environ.get("GRASPNET_URL", "http://localhost:4003")
    output_dir = Path("resources/benchmark/grasps")
    num_rotations = 20
    num_samples = 2048
    score_threshold = 0.2

    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("ReplicaCAD_SceneManipulation-v1", obs_mode="rgb")
    env.reset(seed=0)

    for model_id in YCB_OBJECTS:
        grasps, scores, points = generate_grasps_for_object(
            env, model_id, service_url, num_rotations, num_samples, score_threshold
        )

        output_path = output_dir / f"{model_id}.h5"
        with h5py.File(output_path, "w") as f:
            f.create_dataset("grasps", data=grasps)

        print(f"Saved {len(grasps)} grasps to {output_path}")
        visualize_grasps_pcd(
            grasps,
            scores,
            points,
            window_name=f"Offline grasps: {model_id} (N={len(grasps)})",
        )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
