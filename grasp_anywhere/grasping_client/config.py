from dataclasses import dataclass


@dataclass
class GraspingConfig:
    url: str = "http://localhost:4003"
    timeout: int = 30
    depth_image_scaling: float = (
        1000.0  # Convert meters to mm for uint16/32 interaction
    )
