import base64
import pickle
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


@dataclass
class SamConfig:
    url: str = "http://localhost:4001"


class SamClient:
    def __init__(self, config: SamConfig):
        self.config = config

    def segment_point(
        self, image_rgb: np.ndarray, point: Tuple[int, int], label: int = 1
    ) -> Optional[np.ndarray]:
        """
        Segment a single object given a point in the image.

        Args:
            image_rgb: (H, W, 3) numpy array
            point: (x, y) coordinates of the point
            label: 1 for foreground, 0 for background

        Returns:
            Binary mask as numpy array (H, W), or None if failed.
        """
        # Convert image to base64
        if isinstance(image_rgb, np.ndarray):
            img_pil = Image.fromarray(image_rgb)
        else:
            img_pil = image_rgb

        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Prepare payload
        # Server expects points as list of lists of lists: (nb_predictions, nb_points_per_mask, 2)
        # We handle single point case here
        points = [[[point[0], point[1]]]]
        labels = [[label]]

        payload = {
            "image": img_b64,
            "points": points,
            "labels": labels,
            "return_best": True,
        }

        url = f"{self.config.url.rstrip('/')}/sam_mask_by_point_set"

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse result
            if "result" in data and len(data["result"]) > 0:
                item = data["result"][0]
                segmentation_b64 = item["segmentation"]
                mask = pickle.loads(base64.b64decode(segmentation_b64))
                return mask
            return None

        except Exception as e:
            # For now, let's print/log/raise as appropriate.
            print(f"SAM request failed: {e}")
            return None

    def segment_bbox(
        self, image_rgb: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Segment a single object given a bounding box.

        Args:
            image_rgb: (H, W, 3) numpy array
            bbox: [x1, y1, x2, y2]

        Returns:
            Binary mask
        """
        # ... Similar implementation if needed later ...
