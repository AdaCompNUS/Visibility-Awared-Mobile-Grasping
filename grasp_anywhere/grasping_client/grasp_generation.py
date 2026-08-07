import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from grasp_anywhere.grasping_client.config import GraspingConfig
from grasp_anywhere.utils.logger import log


def _convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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

    # Sort grasps by score
    sorted_indices = np.argsort(scores)[::-1]
    return pred_grasps_cam[sorted_indices], scores[sorted_indices]
