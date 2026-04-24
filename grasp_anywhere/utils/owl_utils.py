from typing import List, Tuple

import numpy as np

from grasp_anywhere.grasping_client.owl_client import OwlClient
from grasp_anywhere.utils.logger import log


def resolve_target_label(
    owl_client: OwlClient,
    rgb_image: np.ndarray,
    candidate_queries: List[str],
    click_uv: Tuple[int, int],
) -> str:
    """
    Detects objects from the candidate list and returns the label of the closest one to the click point.

    Args:
        owl_client: Initialized OwlClient.
        rgb_image: RGB image.
        candidate_queries: List of all possible object labels.
        click_uv: (u, v) tuple of the user's click.

    Returns:
        The label of the object closest to the click.

    Raises:
        RuntimeError: If no objects detected or none close enough.
    """
    boxes, scores, labels = owl_client.detect_objects(rgb_image, candidate_queries)

    if len(boxes) == 0:
        log.warning("No objects detected in the scene.")
        return None

    best_label = None
    min_dist = float("inf")
    u, v = click_uv

    for box, score, label in zip(boxes, scores, labels):
        # box is [x1, y1, x2, y2]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        dist = np.linalg.norm(np.array([cx, cy]) - np.array([u, v]))

        if dist < min_dist:
            min_dist = dist
            best_label = label

    # We could add a distance threshold here (e.g. 50-100 pixels) to avoid matching far away objects
    # but the user said "closest", so we'll stick to closest for now.

    log.info(f"Resolved target object to: '{best_label}' (dist={min_dist:.1f}px)")
    return best_label


def refine_target_point(
    owl_client: OwlClient,
    rgb_image: np.ndarray,
    candidate_queries: List[str],
    current_uv: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Refines the target point (u, v) by detecting objects from candidate_queries
    and selecting the one closest to the current_uv.

    Args:
        owl_client: The initialized OwlClient instance.
        rgb_image: The current RGB image.
        candidate_queries: List of object descriptions to detect.
        current_uv: The initial projected (u, v) coordinates.

    Returns:
        The refined (u, v) coordinates.

    Raises:
        RuntimeError: If detection fails or no valid object center is found.
    """
    if not candidate_queries or owl_client is None:
        return current_uv

    boxes, scores, labels = owl_client.detect_objects(rgb_image, candidate_queries)

    if len(boxes) > 0:
        # Project all box centers and find the one closest to projected (u, v)
        best_box = None
        best_label = None
        min_dist = float("inf")

        u, v = current_uv

        for box, score, label in zip(boxes, scores, labels):
            # box is [x1, y1, x2, y2]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            # Current projected point is (u, v)
            dist = np.linalg.norm(np.array([cx, cy]) - np.array([u, v]))

            if dist < min_dist:
                min_dist = dist
                best_box = (cx, cy)
                best_label = label

        if best_box is not None:
            new_u, new_v = int(best_box[0]), int(best_box[1])
            log.info(
                f"Updated target point from ({u}, {v}) to ({new_u}, {new_v}) "
                f"using Owl-ViT (Matched: '{best_label}', dist={min_dist:.1f}px)"
            )

            return new_u, new_v
        else:
            raise RuntimeError(
                "Owl-ViT detected boxes but failed to find a valid center."
            )
    else:
        # If we have a comprehensive list of objects, failing to find ANY of them
        # near the point suggests the point is not on a known graspable object.
        # However, we must adhere to "no fallbacks" -> explicit failure.
        raise RuntimeError(
            f"Owl-ViT did not find any of the candidate objects: {candidate_queries}"
        )
