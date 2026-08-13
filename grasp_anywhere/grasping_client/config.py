from dataclasses import dataclass
from enum import Enum


class GraspingMode(Enum):
    DEPTH_SEGMENTATION = "depth_segmentation"
    POINTCLOUD_SEGMENTATION = "pointcloud_segmentation"
    POINTCLOUD_ZOOM_SEGMENTATION = "pointcloud_zoom_segmentation"


@dataclass
class GraspingConfig:
    mode: GraspingMode = GraspingMode.DEPTH_SEGMENTATION
    url: str = "http://localhost:4003"
    timeout: int = 30
    zoom_radius: float = 0.5
    target_zoom_dist: float = 0.4
    score_threshold: float = 0.4
    depth_image_scaling: float = (
        1000.0  # Convert meters to mm for uint16/32 interaction
    )
