import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from grasp_anywhere.grasping_client.config import GraspingConfig
from grasp_anywhere.utils.logger import log

CONTACT_GRASPNET_GRIPPER_DEPTH_M = 0.1034
EXECUTION_GRASP_DEPTH_OFFSET_M = 0.11
FETCH_MAX_OPENING_M = 0.10
FETCH_FINGER_APPROACH_BOUNDS_M = (-0.0290, 0.0304)
FETCH_FINGER_NORMAL_BOUNDS_M = (-0.0128, 0.0128)


def _convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def center_grasps_in_fetch_fingers(pred_grasps_cam, depth, segmap, K):
    """Center feasible target cross-sections inside Fetch's finger pads."""
    grasps = np.asarray(pred_grasps_cam).reshape(-1, 4, 4).copy()
    depth = np.asarray(depth)
    segmap = np.asarray(segmap)
    K = np.asarray(K).reshape(3, 3)

    valid = (segmap == 1) & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return grasps

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    points_cam = np.column_stack(
        (
            (cols - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1],
            z,
        )
    )

    approach_min, approach_max = FETCH_FINGER_APPROACH_BOUNDS_M
    normal_min, normal_max = FETCH_FINGER_NORMAL_BOUNDS_M
    for grasp in grasps:
        rotation = grasp[:3, :3]
        finger_center = grasp[:3, 3] + EXECUTION_GRASP_DEPTH_OFFSET_M * rotation[:, 2]
        local_points = (points_cam - finger_center) @ rotation
        in_swept_volume = (
            (local_points[:, 2] >= approach_min)
            & (local_points[:, 2] <= approach_max)
            & (local_points[:, 1] >= normal_min)
            & (local_points[:, 1] <= normal_max)
        )
        closing_coords = local_points[in_swept_volume, 0]
        if closing_coords.size < 2:
            continue

        closing_min = float(np.min(closing_coords))
        closing_max = float(np.max(closing_coords))
        if closing_max - closing_min > FETCH_MAX_OPENING_M:
            continue

        center_shift = 0.5 * (closing_min + closing_max)
        grasp[:3, 3] += center_shift * rotation[:, 0]

    return grasps


def predict_grasps(config: GraspingConfig, rgb, depth, segmap, K):
    """
    Request Contact-GraspNet predictions from RGB, depth, segmentation, and intrinsics.
    """
    if rgb is None or depth is None or segmap is None or K is None:
        raise ValueError(
            "Missing required arguments for Contact-GraspNet inference: rgb, depth, segmap, K"
        )

    segmap_id = 1
    depth_mm = (depth * config.depth_image_scaling).astype(np.uint32)

    rgb_pil = Image.fromarray(rgb)
    depth_pil = Image.fromarray(depth_mm)
    segmap_pil = Image.fromarray(segmap)

    payload = {
        "image_rgb": _convert_pil_image_to_base64(rgb_pil),
        "image_depth": _convert_pil_image_to_base64(depth_pil),
        "segmap": _convert_pil_image_to_base64(segmap_pil),
        "K": K.flatten().tolist() if isinstance(K, np.ndarray) else K,
        "segmap_id": segmap_id,
    }

    url = f"{config.url}/sample_grasp"
    log.info("Sending request to Contact-GraspNet...")

    response = requests.post(url, json=payload, timeout=config.timeout)
    response.raise_for_status()
    result = response.json()
    pred_grasps_cam = np.array(result["pred_grasps_cam"]).reshape(-1, 4, 4)
    scores = np.array(result["scores"])
    pred_grasps_cam = center_grasps_in_fetch_fingers(pred_grasps_cam, depth, segmap, K)

    # Sort grasps by score
    sorted_indices = np.argsort(scores)[::-1]
    return pred_grasps_cam[sorted_indices], scores[sorted_indices]
