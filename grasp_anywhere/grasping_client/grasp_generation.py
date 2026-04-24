import base64
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image

from grasp_anywhere.grasping_client.config import GraspingConfig, GraspingMode
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.pointcloud_transfer import (
    crop_point_cloud,
    densify_point_cloud,
    get_zoom_transform,
    transfer_grasps_back,
    transfer_point_cloud,
)
from grasp_anywhere.utils.visualization_utils import visualize_grasps_pcd


def _convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def _predict_grasps_depth_segmentation(
    config: GraspingConfig, image_rgb, image_depth, segmap, K
):
    """
    Sends RGB, Depth, Segmap, and Intrinsics to the server and gets grasp predictions.
    Internal function.
    """
    if image_rgb is None or image_depth is None or segmap is None or K is None:
        raise ValueError(
            "Missing required arguments for DEPTH_SEGMENTATION mode: rgb, depth, segmap, K"
        )

    segmap_id = 1
    image_depth_mm = (image_depth * config.depth_image_scaling).astype(np.uint32)

    image_rgb_pil = Image.fromarray(image_rgb)
    image_depth_pil = Image.fromarray(image_depth_mm)
    segmap_pil = Image.fromarray(segmap)

    payload = {
        "image_rgb": _convert_pil_image_to_base64(image_rgb_pil),
        "image_depth": _convert_pil_image_to_base64(image_depth_pil),
        "segmap": _convert_pil_image_to_base64(segmap_pil),
        "K": K.flatten().tolist() if isinstance(K, np.ndarray) else K,
        "segmap_id": segmap_id,
    }

    url = f"{config.url}/sample_grasp"
    log.info("Sending request to GraspNet server (Depth Mode)...")

    response = requests.post(url, json=payload, timeout=config.timeout)
    response.raise_for_status()
    result = response.json()
    pred_grasps_cam = np.array(result["pred_grasps_cam"]).reshape(-1, 4, 4)
    scores = np.array(result["scores"])

    # Sort grasps by score
    sorted_indices = np.argsort(scores)[::-1]
    return pred_grasps_cam[sorted_indices], scores[sorted_indices]


def _predict_grasps_pointcloud_segmentation(
    config: GraspingConfig, full_pc, segment_pc
):
    """
    Sends Full Pointcloud and Segment Pointcloud to the server and gets grasp predictions.
    Internal function.
    """
    if full_pc is None or segment_pc is None:
        raise ValueError(
            "Missing required arguments for POINTCLOUD_SEGMENTATION mode: full_pc, segment_pc"
        )

    payload = {
        "full_pc": full_pc.tolist(),
        "segment_pc": segment_pc.tolist(),
    }

    url = f"{config.url}/sample_grasp_with_context"
    response = requests.post(url, json=payload, timeout=config.timeout)
    response.raise_for_status()
    result = response.json()

    pred_grasps = np.array(result["pred_grasps_cam"]).reshape(-1, 4, 4)
    scores = np.array(result["scores"])

    # Sort grasps by score
    sorted_indices = np.argsort(scores)[::-1]
    return pred_grasps[sorted_indices], scores[sorted_indices]


def _predict_grasps_pointcloud_zoom_segmentation(
    config: GraspingConfig, full_pc, segment_pc, visualize=False
):
    """
    Zooms into the object, calls the pointcloud interface, and transforms grasps back.
    Internal function.
    """
    if full_pc is None or segment_pc is None:
        raise ValueError(
            "Missing required arguments for POINTCLOUD_ZOOM_SEGMENTATION mode: full_pc, segment_pc"
        )

    # 1. Zoom logic
    obj_center_cam = np.mean(segment_pc, axis=0)
    T_zoom = get_zoom_transform(obj_center_cam, target_distance=config.target_zoom_dist)

    full_pcd_best = transfer_point_cloud(full_pc, T_zoom)
    segment_pcd_best = transfer_point_cloud(segment_pc, T_zoom)

    # 2. Crop and Densify
    obj_center_best = np.mean(segment_pcd_best, axis=0)

    full_pcd_best = crop_point_cloud(
        full_pcd_best, obj_center_best, radius=config.zoom_radius
    )
    segment_pcd_best = densify_point_cloud(segment_pcd_best)

    # 3. Predict Grasps (Returns grasps in Zoomed Camera Frame)
    # Recursively call the pointcloud function (which handles the network request)
    # Note: we bypass the implementation that checks config mode to direct call the worker
    pred_grasps_best, scores = _predict_grasps_pointcloud_segmentation(
        config, full_pcd_best, segment_pcd_best
    )

    # Filter out grasps with scores less than threshold
    valid_idxs = scores >= config.score_threshold
    pred_grasps_best = pred_grasps_best[valid_idxs]
    scores = scores[valid_idxs]

    # Visualize Zoomed View if requested
    if visualize:
        visualize_grasps_pcd(
            pred_grasps_best,
            scores,
            segment_pcd_best,
            window_name="Zoomed Pointcloud & Grasps",
        )

    # 4. Convert back to original camera frame
    pred_grasps_cam = transfer_grasps_back(pred_grasps_best, T_zoom)

    return pred_grasps_cam, scores


def predict_grasps(
    config: GraspingConfig,
    rgb: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    segmap: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    full_pc: Optional[np.ndarray] = None,
    segment_pc: Optional[np.ndarray] = None,
    visualize: bool = False,
):
    """
    Unified entry point for grasp prediction.
    Dispatches to the appropriate implementation based on config.mode.

    Args:
        config: GraspingConfig object containing 'mode' and parameters.
        rgb: (H, W, 3) Image, required for DEPTH_SEGMENTATION.
        depth: (H, W) Image, required for DEPTH_SEGMENTATION.
        segmap: (H, W) Image, required for DEPTH_SEGMENTATION.
        K: (3, 3) Intrinsics, required for DEPTH_SEGMENTATION.
        full_pc: (N, 3) Point Cloud, required for POINTCLOUD variants.
        segment_pc: (M, 3) Point Cloud, required for POINTCLOUD variants.
        visualize: Boolean, enables debug visualization for supported modes.

    Returns:
        pred_grasps_cam: (K, 4, 4) Grasps in camera frame.
        scores: (K,) Scores for each grasp.
    """
    try:
        if config.mode == GraspingMode.DEPTH_SEGMENTATION:
            return _predict_grasps_depth_segmentation(config, rgb, depth, segmap, K)

        elif config.mode == GraspingMode.POINTCLOUD_SEGMENTATION:
            return _predict_grasps_pointcloud_segmentation(config, full_pc, segment_pc)

        elif config.mode == GraspingMode.POINTCLOUD_ZOOM_SEGMENTATION:
            return _predict_grasps_pointcloud_zoom_segmentation(
                config, full_pc, segment_pc, visualize
            )

        else:
            raise ValueError(f"Unknown grasping mode: {config.mode}")
    except requests.exceptions.RequestException as e:
        log.error(f"Grasp prediction service call failed: {e}")
        return None, None
