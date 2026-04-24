import os
from typing import List, Optional

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label

from grasp_anywhere.utils.logger import log


class BaseSampler:
    """
    Simple base sampler backed by a 2D costmap.
    - Loads costmap once from .npz with keys: costmap, resolution, origin_x, origin_y, width, height
    - For a 3D target point, selects the lowest-cost reachable cell within a radius and faces the target.
    - Cells with cost >= occupied_threshold (default 1.0) are treated as invalid.
    """

    def __init__(
        self, costmap_path: str = "resources/costmap.npz", dynamic_costmap: bool = False
    ) -> None:
        self.costmap_path = costmap_path
        self.dynamic_costmap = bool(dynamic_costmap)
        self.costmap, self.metadata = self._load_costmap(costmap_path)

    def _load_costmap(self, path: str):
        if not os.path.exists(path):
            log.warning(f"Costmap file not found: {path}")
            return None, None
        data = np.load(path)
        costmap = np.array(data["costmap"], dtype=float)
        meta = {
            "resolution": float(data["resolution"]),
            "origin_x": float(data["origin_x"]),
            "origin_y": float(data["origin_y"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
        }
        return costmap, meta

    def update_from_pointcloud(
        self,
        points_3d: np.ndarray,
        resolution: float = 0.05,
        height_min: float = 0.05,
        height_max: float = 1.5,
        max_distance: float = 2.0,
    ) -> None:
        """
        Build a 2D SDF costmap from a 3D world-frame point cloud.

        Assumptions:
        - Z-up coordinates
        - Input points_3d is (N,3) float array
        - No checks or error handling
        """
        pts = points_3d.astype(float)

        # Slice by height along Z
        mask = (pts[:, 2] >= height_min) & (pts[:, 2] <= height_max)
        filtered = pts[mask]

        # Project to XY plane
        points_2d = filtered[:, [0, 1]]

        # Grid extents and resolution
        padding = resolution * 5.0
        min_xy = np.min(points_2d, axis=0) - padding
        max_xy = np.max(points_2d, axis=0) + padding
        dims = np.ceil((max_xy - min_xy) / resolution).astype(int)

        # Rasterize occupancy
        occupancy = np.zeros((dims[1], dims[0]), dtype=bool)
        grid_xy = ((points_2d - min_xy) / resolution).astype(int)
        gx = np.clip(grid_xy[:, 0], 0, dims[0] - 1)
        gy = np.clip(grid_xy[:, 1], 0, dims[1] - 1)
        occupancy[gy, gx] = True

        # Close gaps, compute navigable area (largest hole)
        obstacle = binary_closing(occupancy, structure=np.ones((3, 3)))
        holes = ~obstacle
        labeled, _ = label(holes)
        counts = np.bincount(labeled.ravel())
        largest = np.argmax(counts[1:]) + 1
        navigable = labeled == largest

        # SDF and cost
        dist = distance_transform_edt(~obstacle) * resolution
        grad = np.exp(-dist / max_distance)
        cost = np.ones_like(grad, dtype=float)
        cost[navigable] = grad[navigable]

        self.costmap = cost
        self.metadata = {
            "resolution": float(resolution),
            "origin_x": float(min_xy[0]),
            "origin_y": float(min_xy[1]),
            "width": int(cost.shape[1]),
            "height": int(cost.shape[0]),
        }

    def _world_to_grid(self, x: float, y: float):
        res = self.metadata["resolution"]
        gx = int((x - self.metadata["origin_x"]) / res)
        gy = int((y - self.metadata["origin_y"]) / res)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int):
        res = self.metadata["resolution"]
        x = self.metadata["origin_x"] + gx * res
        y = self.metadata["origin_y"] + gy * res
        return x, y

    def _grid_to_world_vectorized(self, grid_pos):
        """Converts grid coordinates to world coordinates, vectorized."""
        grid_x, grid_y = grid_pos
        world_x = self.metadata["origin_x"] + grid_x * self.metadata["resolution"]
        world_y = self.metadata["origin_y"] + grid_y * self.metadata["resolution"]
        return np.array([world_x, world_y])

    def sample_base_pose(
        self,
        target_point: List[float],
        manipulation_radius: float,
    ) -> Optional[tuple[float, float, float]]:
        """
        Samples a base pose by converting the costmap into a probability distribution
        and sampling from it. Lower cost areas have a higher probability of being sampled.
        """
        if self.costmap is None:
            log.warning(
                "BaseSampler.sample_base_pose: no costmap available. Consider calling update_from_pointcloud()."
            )
            return None

        object_center = np.array(target_point)
        height = self.metadata["height"]
        width = self.metadata["width"]

        # Create a grid of world coordinates corresponding to each cell in the costmap
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        world_coords = self._grid_to_world_vectorized((grid_x, grid_y))

        # Calculate the distance from each cell to the object
        distances = np.linalg.norm(
            world_coords - object_center[:2, np.newaxis, np.newaxis], axis=0
        )

        radius_mask = distances <= manipulation_radius

        occupied_threshold = 0.9

        # Simple handling:
        # - Map non-finite to 1.0
        # - Clamp costs into [0, 1]
        # - Set out-of-radius cells to 1.0
        search_costs = np.asarray(self.costmap, dtype=float).copy()
        search_costs[~np.isfinite(search_costs)] = 1.0
        search_costs = np.clip(search_costs, 0.0, 1.0)
        search_costs[search_costs >= occupied_threshold] = 1.0
        search_costs[~radius_mask] = 1.0

        # Convert costs to probabilities uniformly: p = 1 - cost
        probabilities = 1.0 - search_costs
        probabilities[probabilities < 0] = 0
        probabilities = np.power(probabilities, 2)  # Sharpen distribution

        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-6:
            log.warning(f"No valid base poses found. Probability sum: {prob_sum}")
            return None
        else:
            # Normalize to form a probability distribution and sample
            probabilities /= prob_sum
            flat_probs = probabilities.flatten()
            num_cells = probabilities.size
            chosen_flat_index = np.random.choice(num_cells, p=flat_probs)

        sampled_idx = np.unravel_index(chosen_flat_index, probabilities.shape)

        best_grid_pos = (sampled_idx[1], sampled_idx[0])  # (x, y) order

        # This check is technically redundant due to how np.unravel_index works,
        # but it can provide an explicit guarantee against out-of-bounds errors.
        if not (0 <= best_grid_pos[0] < width and 0 <= best_grid_pos[1] < height):
            log.warning(f"Sampled grid position {best_grid_pos} is out of bounds.")
            return None

        base_pos_2d = self._grid_to_world(best_grid_pos[0], best_grid_pos[1])
        base_pos_2d = np.array(base_pos_2d)

        direction_vector = object_center[:2] - base_pos_2d
        yaw = np.arctan2(direction_vector[1], direction_vector[0])

        return (base_pos_2d[0], base_pos_2d[1], yaw)
