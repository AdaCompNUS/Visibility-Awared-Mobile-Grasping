import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import sapien
import trimesh
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh
from PIL import Image

from grasp_anywhere.utils.visualization_utils import visualize_grasps_pcd


@dataclass
class GraspPipelineCfg:
    num_views: int = 10
    view_radius: float = 0.3
    view_height: float = 0.3
    settle_steps_per_view: int = 2
    max_grasps_out: int = 256
    crop_radius: float = 0.6
    crop_z_below: float = 0.2
    crop_z_above: float = 1.7
    timeout_s: float = 30.0
    # Virtual camera parameters (no dependency on robot-mounted cameras)
    width: int = 640
    height: int = 480
    fov_y_deg: float = 60.0
    visualize: bool = False


def _to_numpy(x: object) -> np.ndarray:
    return x.detach().cpu().numpy()


def _png_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> sapien.Pose:
    """
    SAPIEN camera convention: (forward, right, up) = (x, -y, z).
    We build a camera pose whose +X axis points from eye -> target.
    """
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    left = np.cross(up, forward)
    left = left / (np.linalg.norm(left) + 1e-9)
    up2 = np.cross(forward, left)

    R = np.stack([forward, left, up2], axis=1).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    q = trimesh.transformations.quaternion_from_matrix(T).astype(np.float32)
    return sapien.Pose(p=eye.tolist(), q=q.tolist())


def _intrinsic_from_fov(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    fov_y = float(fov_y_deg) * np.pi / 180.0
    fy = 0.5 * float(height) / np.tan(0.5 * fov_y)
    fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _camera_rgb_depth_seg(
    cam: Camera, scene: object, seg_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene.update_render()
    cam.capture()
    obs = cam.get_obs(rgb=True, depth=True, segmentation=True, position=False)

    rgb = _to_numpy(obs["rgb"])[0, ..., :3]
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_m = _to_numpy(obs["depth"])[0, ..., 0].astype(np.float32) / 1000.0
    seg_ids = _to_numpy(obs["segmentation"])[0, ..., 0].astype(np.int32)
    mask01 = (seg_ids == int(seg_id)).astype(np.uint8)
    return rgb, depth_m, mask01


def _visualize_rendered_images(
    rgb: np.ndarray, depth_m: np.ndarray, mask01: np.ndarray
) -> None:
    rgb_u8 = rgb.astype(np.uint8)
    depth = depth_m.astype(np.float32)
    m = mask01.astype(np.uint8)

    depth_vis = depth.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0.0
    dmax = float(np.max(depth_vis)) if depth_vis.size else 1.0
    if dmax <= 1e-6:
        dmax = 1.0
    depth_u8 = np.clip(depth_vis / dmax * 255.0, 0.0, 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    overlay = rgb_u8.copy()
    overlay[m > 0] = (
        0.7 * overlay[m > 0] + 0.3 * np.array([0, 255, 0], dtype=np.uint8)
    ).astype(np.uint8)

    cv2.imshow("grasp_pipeline/rgb", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    cv2.imshow("grasp_pipeline/depth", depth_color)
    cv2.imshow("grasp_pipeline/rgb_mask", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def _sample_actor_mesh_points_world(actor: object, n_points: int) -> np.ndarray:
    mesh = get_actor_visual_mesh(actor)
    if mesh is None or mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32)
    T = np.asarray(actor.pose.to_transformation_matrix())
    if T.ndim == 3:
        T = T[0]
    mesh = mesh.copy()
    mesh.apply_transform(T)
    pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    return pts.astype(np.float32)


def _crop_points_world(
    pts_world: np.ndarray,
    center_world: np.ndarray,
    crop_radius: float,
    crop_z_below: float,
    crop_z_above: float,
) -> np.ndarray:
    if pts_world.shape[0] == 0:
        return pts_world.astype(np.float32)
    p = pts_world.astype(np.float32)
    c = center_world.astype(np.float32).reshape(1, 3)
    dxy2 = (p[:, 0] - c[0, 0]) ** 2 + (p[:, 1] - c[0, 1]) ** 2
    keep = dxy2 <= float(crop_radius) ** 2
    keep &= p[:, 2] >= float(center_world[2] - crop_z_below)
    keep &= p[:, 2] <= float(center_world[2] + crop_z_above)
    return p[keep]


def _predict_grasps_graspnet_service(
    *,
    service_url: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    segmap01: np.ndarray,
    K: np.ndarray,
    timeout_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_mm = (depth_m.astype(np.float32) * 1000.0).astype(np.uint16)
    payload = {
        "image_rgb": _png_base64(Image.fromarray(rgb.astype(np.uint8))),
        "image_depth": _png_base64(Image.fromarray(depth_mm)),
        "segmap": _png_base64(Image.fromarray(segmap01.astype(np.uint8))),
        "K": K.reshape(-1).tolist(),
        "segmap_id": 1,
    }
    r = requests.post(
        service_url.rstrip("/") + "/sample_grasp",
        json=payload,
        timeout=float(timeout_s),
    )
    r.raise_for_status()
    out = r.json()
    grasps_cam = np.asarray(out["pred_grasps_cam"], dtype=np.float32).reshape(-1, 4, 4)
    scores = np.asarray(out["scores"], dtype=np.float32).reshape(-1)
    order = np.argsort(scores)[::-1]
    return grasps_cam[order], scores[order]


def _farthest_point_sampling_positions(
    grasps: np.ndarray, n_samples: int
) -> np.ndarray:
    pos = grasps[:, :3, 3].astype(np.float32)
    n = min(int(n_samples), int(pos.shape[0]))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    sel = np.empty((n,), dtype=np.int64)
    sel[0] = 0
    d2 = np.sum((pos - pos[sel[0]]) ** 2, axis=1)
    for i in range(1, n):
        sel[i] = int(np.argmax(d2))
        nd2 = np.sum((pos - pos[sel[i]]) ** 2, axis=1)
        d2 = np.minimum(d2, nd2)
    return sel


def run_grasp_pipeline_for_actor(
    *,
    env: object,
    actor: object,
    grasp_service_url: str,
    cfg: GraspPipelineCfg = GraspPipelineCfg(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Pipeline (exactly):
    1) multiview renders around target (rgb/depth/seg)
    2) query service; fuse all grasps into world; farthest sample to cfg.max_grasps_out (default 256)
    3) crop fused point cloud around target and surrounding env
    4) visualize cropped point cloud + sampled grasps

    Returns:
        grasps_world (M,4,4), scores (M,),
        points_world_crop (P,3), colors_crop (P,3) or None
    """
    entity = actor._objs[0]
    seg_id = int(entity.per_scene_id)
    center_world = np.asarray(entity.pose.p, dtype=np.float32).reshape(3)
    # Create a virtual camera (not attached to the robot) and move it around the object.
    # Intrinsics are defined purely by cfg to avoid dependencies on robot-mounted cameras.
    K = _intrinsic_from_fov(int(cfg.width), int(cfg.height), float(cfg.fov_y_deg))

    virt = Camera(
        CameraConfig(
            uid="virt_grasp_cam",
            pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
            width=int(cfg.width),
            height=int(cfg.height),
            intrinsic=K,
            near=0.01,
            far=5.0,
            shader_pack="minimal",
        ),
        scene=env.unwrapped.scene,
    )
    scene = env.unwrapped.scene
    grasps_world_all = []
    scores_all = []

    for v in range(int(cfg.num_views)):
        yaw = float(2.0 * np.pi * float(v) / float(max(1, int(cfg.num_views))))
        cam_pos = center_world + np.array(
            [
                np.cos(yaw) * float(cfg.view_radius),
                np.sin(yaw) * float(cfg.view_radius),
                float(cfg.view_height),
            ],
            dtype=np.float32,
        )

        virt.camera.set_local_pose(_look_at_pose(cam_pos, center_world))
        for _ in range(int(max(1, int(cfg.settle_steps_per_view)))):
            env.step(None)

        rgb, depth_m, mask01 = _camera_rgb_depth_seg(virt, scene, seg_id)
        if cfg.visualize:
            _visualize_rendered_images(rgb, depth_m, mask01)
        if int(np.count_nonzero(mask01)) == 0:
            continue

        grasps_cam, scores = _predict_grasps_graspnet_service(
            service_url=grasp_service_url,
            rgb=rgb,
            depth_m=depth_m,
            segmap01=mask01,
            K=K,
            timeout_s=float(cfg.timeout_s),
        )
        keep = scores >= 0.4
        grasps_cam = grasps_cam[keep]
        scores = scores[keep]
        if grasps_cam.shape[0] == 0:
            continue

        ext = np.asarray(virt.get_params()["extrinsic_cv"], dtype=np.float32)
        T_cam_world = np.eye(4, dtype=np.float32)
        T_cam_world[:3, :] = ext
        T_world_cam = np.linalg.inv(T_cam_world).astype(np.float32)

        grasps_world = np.einsum(
            "ij,njk->nik", T_world_cam.astype(np.float32), grasps_cam.astype(np.float32)
        ).astype(np.float32)
        grasps_world_all.append(grasps_world)
        scores_all.append(scores.astype(np.float32))

    if len(grasps_world_all) == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            None,
        )

    grasps_world = np.concatenate(grasps_world_all, axis=0).astype(np.float32)
    scores = np.concatenate(scores_all, axis=0).astype(np.float32)

    order = np.argsort(scores)[::-1]
    order = order[: min(len(order), 8192)]
    grasps_top = grasps_world[order]
    scores_top = scores[order]
    idx = _farthest_point_sampling_positions(grasps_top, int(cfg.max_grasps_out))
    grasps_keep = grasps_top[idx]
    scores_keep = scores_top[idx]

    # Visualization point cloud: sample simulator meshes around the object (no depth/pointcloud fusion).
    scene = env.unwrapped.scene
    env_pts_list = []
    for a in scene.get_all_actors():
        if a is entity:
            continue
        env_pts_list.append(_sample_actor_mesh_points_world(a, n_points=20000))
    env_pts = (
        np.concatenate(env_pts_list, axis=0).astype(np.float32)
        if env_pts_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    obj_pts = _sample_actor_mesh_points_world(entity, n_points=50000)

    env_crop = _crop_points_world(
        env_pts, center_world, cfg.crop_radius, cfg.crop_z_below, cfg.crop_z_above
    )
    obj_crop = _crop_points_world(
        obj_pts, center_world, cfg.crop_radius, cfg.crop_z_below, cfg.crop_z_above
    )

    pts_crop = (
        np.concatenate([env_crop, obj_crop], axis=0).astype(np.float32)
        if (env_crop.size or obj_crop.size)
        else np.zeros((0, 3), dtype=np.float32)
    )
    cols_crop = None
    if pts_crop.shape[0] > 0:
        env_n = int(env_crop.shape[0])
        obj_n = int(obj_crop.shape[0])
        cols_crop = np.concatenate(
            [
                np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (env_n, 1)),
                np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (obj_n, 1)),
            ],
            axis=0,
        )

    if cfg.visualize:
        visualize_grasps_pcd(
            pred_grasps=grasps_keep,
            scores=scores_keep,
            points=pts_crop,
            colors=cols_crop,
            window_name="Cropped point cloud + sampled grasps (world)",
        )

    return grasps_keep, scores_keep, pts_crop, cols_crop
