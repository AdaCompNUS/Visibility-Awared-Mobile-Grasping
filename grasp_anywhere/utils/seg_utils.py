from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def mask_from_uv(
    segmentation: np.ndarray,
    uv: Tuple[int, int],
    *,
    background_id: int = 0,
) -> Optional[np.ndarray]:
    """
    Segment object from a pixel (u,v) using rendered segmentation ids.

    This is the sim replacement for SAM point prompting:
      instance_id = seg_ids[v, u]
      mask = seg_ids == instance_id

    Returns:
        (H,W) uint8 mask with {0,1}, or None if uv is OOB / hits background.
    """
    seg = np.asarray(segmentation)
    # Deterministic for our ManiSkill setup: get_sensor_snapshot() currently returns (H, W, 1).
    # We intentionally do NOT try to be general-purpose here.
    if seg.ndim != 3 or seg.shape[-1] != 1:
        raise ValueError(
            f"Expected segmentation shape (H, W, 1) from ManiSkill, got shape={seg.shape}"
        )
    seg_ids = seg[..., 0].astype(np.int32, copy=False)
    H, W = seg_ids.shape
    u, v = int(uv[0]), int(uv[1])

    if not (0 <= u < W and 0 <= v < H):
        return None

    instance_id = int(seg_ids[v, u])
    if instance_id == int(background_id):
        return None

    mask = seg_ids == instance_id
    if not np.any(mask):
        return None

    return mask.astype(np.uint8)


def mask_from_segmentation_id(
    segmentation: np.ndarray,
    segmentation_id: int,
) -> np.ndarray:
    """
    Extract a binary mask for a specific segmentation ID (per_scene_id).
    """
    seg_ids = segmentation[..., 0].astype(np.int32, copy=False)
    mask = (seg_ids == segmentation_id).astype(np.uint8)
    return mask
