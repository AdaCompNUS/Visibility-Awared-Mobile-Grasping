import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image


@dataclass
class OwlConfig:
    url: str = "http://localhost:4000"
    score_threshold: float = 0.1  # Lower threshold for better recall


class OwlClient:
    def __init__(self, config: OwlConfig):
        self.config = config

    def detect_objects(
        self, image_rgb: np.ndarray, queries: List[str]
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Detect objects in RGB image using Owl-ViT server.

        Args:
            image_rgb: (H, W, 3) numpy array
            queries: List of text queries (e.g. ["a red apple", "a bottle"])

        Returns:
            boxes: List of [x1, y1, x2, y2] in pixel coordinates
            scores: List of scores
            labels: List of labels (corresponding to queries)
        """
        if not queries:
            return [], [], []

        # Convert image to base64
        img_pil = Image.fromarray(image_rgb)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        payload = {
            "image": img_b64,
            "text_queries": queries,
            "bbox_conf_threshold": self.config.score_threshold,
            "bbox_score_top_k": 20,
        }

        # Assume /owl_detect endpoint based on PerceptionService
        url = f"{self.config.url.rstrip('/')}/owl_detect"

        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Server returns normalized boxes [x1, y1, x2, y2]
        bboxes_norm = data["bboxes"]
        scores = data["scores"]
        box_names = data["box_names"]

        # Convert to pixel coordinates
        height, width = image_rgb.shape[:2]
        boxes_pixel = []
        for box in bboxes_norm:
            boxes_pixel.append(
                [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            )

        return boxes_pixel, scores, box_names
