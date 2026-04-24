import threading
from typing import Callable, List, Optional, Tuple

import numpy as np
import open3d as o3d

from grasp_anywhere.utils.perception_utils import depth2pc, register_point_clouds_icp


class Scene:
    """
    PCD-native scene maintainer.
    - static_map (S): canonical world PCD, considered immutable.
    - observation_pcd (M): dynamic environment from depth updates, evolves based on mode.
    - goal_pcd (G): goal specification cloud, maintained separately.

    Supports four modes for ablation studies:
    - 'static': Environment is always the static map S. Updates are ignored.
    - 'latest': Environment is S + M, where M is completely replaced by the latest observation.
    - 'accumulated': Environment is S + M, where M is iteratively updated by clearing the
                    current frustum and merging the new observation. This accumulates history.
    - 'combine': Environment is S + M, where M is built by stacking all observations together
                 and downsampling to prevent excessive density.
    - 'ray_casting': Environment is S + M, where M is updated using ray casting to remove
                     free-space points while maintaining occluded regions.
    """

    def __init__(
        self,
        static_map_pcd: Optional[np.ndarray] = None,
        *,
        ground_z_threshold: float = 0.3,
        default_depth_range: Tuple[float, float] = (0.2, 3.0),
        enable_ground_filter: bool = True,
        robot_filter: Optional[
            Callable[[np.ndarray], Tuple[List[List[float]], Optional[object]]]
        ] = None,
        mode: str = "latest",
        merge_radius: float = 0.03,
        downsample_voxel_size: float = 0.05,
    ):
        raw_s = (
            np.asarray(static_map_pcd, dtype=np.float32)
            if static_map_pcd is not None
            else np.empty((0, 3), dtype=np.float32)
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_s)
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        self._S = np.asarray(pcd.points, dtype=np.float32)

        self._M = np.empty((0, 3), dtype=np.float32)
        self._G = np.empty((0, 3), dtype=np.float32)

        # Simple parameters (no config object)
        self.ground_z_threshold = ground_z_threshold
        self.default_depth_range = default_depth_range
        self.enable_ground_filter = enable_ground_filter

        self.mode = mode
        if self.mode not in [
            "static",
            "latest",
            "accumulated",
            "combine",
            "ray_casting",
        ]:
            raise ValueError(
                "mode must be one of 'static', 'latest', 'accumulated', 'combine', or 'ray_casting'"
            )

        self.merge_radius = merge_radius  # Used by 'accumulated' mode
        self.downsample_voxel_size = downsample_voxel_size  # Used by 'combine' mode

        self._lock = threading.Lock()
        self._robot_filter = robot_filter

    # -------------------- Public API --------------------
    def update(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        joint_dict: dict,
        z_range: Optional[Tuple[float, float]] = None,
        enable_icp_alignment: bool = False,
    ) -> None:
        """
        Convenience update from depth + intrinsics + extrinsics.
        Behavior depends on the 'mode' set during initialization.

        Args:
            joint_dict: Synchronized joint states for robot filtering
        """
        if self.mode == "static":
            return

        # 1) Convert depth to world-frame point cloud
        pcd_world = self._depth_to_world_pcd(depth, K, T_wc, z_range)

        # 1.5) Crop around center
        if pcd_world.size > 0:
            crop_center = T_wc[:3, 3]
            pcd_world = self._crop_sphere(
                pcd_world, crop_center.astype(np.float32), 2.5
            )

        # 2) Apply filtering pipeline: ground -> robot with synchronized state
        pcd_world = self._filter_points(pcd_world, self._robot_filter, joint_dict)

        # 3) Optional ICP alignment to static map
        if (
            enable_icp_alignment
            and self._S is not None
            and self._S.size > 0
            and pcd_world.size > 0
        ):
            aligned_pcd = register_point_clouds_icp(pcd_world, self._S)
            if aligned_pcd is not None and aligned_pcd.size > 0:
                pcd_world = aligned_pcd

        frustum = (
            np.asarray(K).reshape(3, 3),
            np.asarray(T_wc).reshape(4, 4),
            z_range or self.default_depth_range,
        )

        if self.mode == "latest":
            with self._lock:
                self._M = pcd_world
        elif self.mode == "accumulated":
            self._update_core(pcd_world, frustum=frustum)
        elif self.mode == "combine":
            self._update_combine(pcd_world)
        elif self.mode == "ray_casting":
            self._update_ray_casting(
                pcd_world, depth, K, T_wc, z_range or self.default_depth_range
            )

    def clear_observations(self) -> None:
        """Clear the dynamic observation map M."""
        with self._lock:
            self._M = np.empty((0, 3), dtype=np.float32)
            return None

    def set_goal_pcd(self, pcd_world: np.ndarray) -> None:
        """Set or replace the goal specification point cloud G."""
        with self._lock:
            self._G = np.asarray(pcd_world, dtype=np.float32)

    def get_goal_pcd(self) -> np.ndarray:
        with self._lock:
            return self._G.copy()

    # Convenience properties
    @property
    def belief(self) -> np.ndarray:
        """Current belief E(t) = S ∪ M(t). Read-only property."""
        return self.current_environment()

    @property
    def goal(self) -> np.ndarray:
        """Current goal specification point cloud G."""
        return self.get_goal_pcd()

    def current_environment(
        self,
        roi_center: Optional[Tuple[float, float, float]] = None,
        roi_radius: Optional[float] = None,
    ) -> np.ndarray:
        """Return E(t) = S ∪ M(t) as a point cloud. Optionally cropped to ROI."""
        with self._lock:
            if self.mode == "static":
                env = self._S.copy()
            elif self._S is None or len(self._S) == 0:
                env = self._M.copy()
            elif len(self._M) == 0:
                env = self._S.copy()
            else:
                env = self._merge_dedup(self._S, self._M, self.merge_radius)

        if roi_center is not None and roi_radius is not None and len(env) > 0:
            env = self._crop_sphere(
                env, np.asarray(roi_center, dtype=np.float32), float(roi_radius)
            )
        return env

    def current_observations(self) -> np.ndarray:
        """Return M(t) only for debugging/visualization."""
        with self._lock:
            return self._M.copy()

    # -------------------- Internal helpers --------------------

    def _depth_to_world_pcd(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        z_range: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        depth = np.nan_to_num(depth)
        K = np.asarray(K).reshape(3, 3)
        T = np.asarray(T_wc).reshape(4, 4)
        zmin, zmax = z_range or self.default_depth_range

        # Use shared depth-to-pointcloud util for consistency across codebase
        pc_cam, _ = depth2pc(depth, K, stride=4)
        if pc_cam is None or len(pc_cam) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Apply camera-depth range filter in camera frame (z axis)
        pc_cam = pc_cam[(pc_cam[:, 2] >= zmin) & (pc_cam[:, 2] <= zmax)]
        if pc_cam.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Transform to world frame: P_w = R * P_c + t
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        pc_world = (pc_cam.astype(np.float32) @ R.T) + t
        return pc_world.astype(np.float32)

    @staticmethod
    def _crop_sphere(
        points: np.ndarray, center: np.ndarray, radius: float
    ) -> np.ndarray:
        diffs = points[:, :2] - center[:2]
        d2 = np.sum(diffs * diffs, axis=1)
        mask_xy = d2 <= (radius * radius)
        if center.shape[0] == 3:
            mask_z = np.ones_like(mask_xy, dtype=bool)
        else:
            mask_z = np.ones_like(mask_xy, dtype=bool)
        return points[mask_xy & mask_z]

    @staticmethod
    def _points_in_frustum_mask(
        points: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        z_range: Tuple[float, float],
    ) -> np.ndarray:
        # Transform world points into camera frame: P_c = R^T (P_w - t)
        R = T_wc[:3, :3].astype(np.float32)
        t = T_wc[:3, 3].astype(np.float32)
        Pw = points
        Pc = (Pw - t) @ R

        z = Pc[:, 2]
        zmin, zmax = z_range
        mask_z = (z >= zmin) & (z <= zmax)
        if not np.any(mask_z):
            return np.zeros_like(z, dtype=bool)

        # Filter points that are behind or too far
        Pc_filtered = Pc[mask_z]
        z_filtered = Pc_filtered[:, 2]

        # Project points onto the image plane
        u = (Pc_filtered[:, 0] * K[0, 0] / z_filtered) + K[0, 2]
        v = (Pc_filtered[:, 1] * K[1, 1] / z_filtered) + K[1, 2]

        # Check if projected points are within image boundaries
        width, height = 640, 480
        mask_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        # Combine masks to create the final frustum mask
        final_mask = np.zeros(len(points), dtype=bool)
        final_mask[np.where(mask_z)[0]] = mask_uv
        return final_mask

    @staticmethod
    def _merge_dedup(base: np.ndarray, add: np.ndarray, radius: float) -> np.ndarray:
        if len(add) == 0:
            return base
        if len(base) == 0:
            return add

        # Use a voxel grid to deduplicate points.
        q = max(radius, 1e-6)

        # Optimized vectorized implementation
        # 1. Compute voxel keys for base points
        base_keys = np.floor(base / q).astype(np.int64)

        # 2. Compute voxel keys for add points
        add_keys = np.floor(add / q).astype(np.int64)

        # 3. Stack keys: base first, then add
        all_keys = np.vstack((base_keys, add_keys))

        # 4. Find unique voxels. np.unique with return_index=True returns the index of the first occurrence.
        # Since base keys are first, any voxel occupied by base will point to an index < len(base).
        # Any voxel occupied ONLY by add (or new in add) will have an index >= len(base).
        _, unique_indices = np.unique(all_keys, axis=0, return_index=True)

        # 5. Select indices that correspond to the 'add' array (new voxels)
        n_base = len(base)
        # Filter indices pointing to the 'add' section
        add_indices = unique_indices[unique_indices >= n_base] - n_base

        if len(add_indices) == 0:
            return base

        # 6. Sort indices to preserve relative order
        add_indices.sort()

        # 7. Stack base with the selected new points from add
        return np.vstack((base, add[add_indices]))

    def _update_core(
        self,
        pcd_world: np.ndarray,
        *,
        frustum: Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]],
    ) -> None:
        if pcd_world is None or pcd_world.size == 0:
            return None

        with self._lock:
            M = self._M

            # Determine removal region (frustum only)
            if len(M) > 0 and frustum is not None:
                K, T_wc, z_rng = frustum
                remove_mask = self._points_in_frustum_mask(M, K, T_wc, z_rng)
                if np.any(remove_mask):
                    M = M[~remove_mask]

            # Merge with deduplication for the accumulated map
            M = self._merge_dedup(M, pcd_world, self.merge_radius)

            self._M = M
            return None

    def _update_combine(self, pcd_world: np.ndarray) -> None:
        """
        Combine mode: Stack all observations together and downsample.
        No frustum removal - just accumulate and downsample to prevent density.
        """
        if pcd_world is None or pcd_world.size == 0:
            return None

        with self._lock:
            M = self._M

            # Stack the new observation with existing points
            if len(M) == 0:
                M = pcd_world
            else:
                M = np.vstack((M, pcd_world))

            # Downsample to prevent excessive density
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(M)
            pcd = pcd.voxel_down_sample(voxel_size=self.downsample_voxel_size)
            M = np.asarray(pcd.points, dtype=np.float32)

            self._M = M
            return None

    def _update_ray_casting(
        self,
        pcd_world: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        z_range: Tuple[float, float],
    ) -> None:
        """
        Ray casting update:
        1. Projects existing map M to current camera view.
        2. Removes points that are CLEARLY in front of the measured surface (free space).
           - Checks if point_depth < current_measured_depth - margin.
        3. Preserves points that are behind the measured surface (occluded) or match it.
        4. Merges new pcd_world into M.
        """
        if depth is None:
            return

        with self._lock:
            M = self._M
            K = np.asarray(K).reshape(3, 3)
            # Ensure T_wc is 4x4
            if T_wc.shape == (3, 4):
                T_wc = np.vstack((T_wc, [0, 0, 0, 1]))
            T_wc = np.asarray(T_wc).reshape(4, 4)

            # --- 1) Ray casting clearing ---
            if len(M) > 0:
                # Transform M to camera frame: P_c = (P_w - t) @ R = R.T @ (P_w - t)?
                # Standard math: P_c = R_wc.T @ (P_w - t_wc)
                # T_wc = [R | t]
                R = T_wc[:3, :3].astype(np.float32)
                t = T_wc[:3, 3].astype(np.float32)

                # Vectorized transform
                # Points are (N, 3).
                # (P - t) @ R  is equivalent to (R.T @ (P-t).T).T if R is rotation matrix
                # because (R.T @ v).T = v.T @ R
                Pc = (M - t) @ R
                zc = Pc[:, 2]

                zmin, zmax = z_range

                # Project only points with positive z to avoid div/0
                valid_proj_mask = zc > 1e-3

                # We can do a bounding box check first if M is huge, but here just process all
                # Project to UV
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                # Initialize masks
                # u, v calculation
                # We only compute for valid_proj_mask to save time if needed,
                # but numpy is fast enough for ~100k points usually.
                # Let's compute for all to keep shape aligned, mask later.
                # To avoid division by zero:
                zc_safe = np.where(valid_proj_mask, zc, 1.0)
                u = (Pc[:, 0] * fx / zc_safe) + cx
                v = (Pc[:, 1] * fy / zc_safe) + cy

                u = np.rint(u).astype(np.int32)
                v = np.rint(v).astype(np.int32)

                h, w = depth.shape
                in_img_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                # Candidates to potentially remove
                check_mask = valid_proj_mask & in_img_mask

                if np.any(check_mask):
                    idx_check = np.where(check_mask)[0]
                    u_check = u[idx_check]
                    v_check = v[idx_check]
                    z_old = zc[idx_check]

                    z_measured = depth[v_check, u_check]

                    # Valid measurement logic:
                    # If z_measured <= 0 or NaN, we treat it as unknown/max-range?
                    # If it's invalid, we shouldn't clear based on it.
                    valid_obs = (z_measured > 0) & np.isfinite(z_measured)

                    # Free space condition:
                    # Old point is CLOSER than Measured point by a margin
                    # z_old < z_measured - margin
                    # If z_old > z_measured + margin -> Occluded (Keep)
                    # If near -> Surface (Keep/Merge)
                    cutoff = z_measured - self.merge_radius
                    is_free = z_old < cutoff

                    # Points to remove
                    remove_indices_local = np.where(valid_obs & is_free)[0]

                    if len(remove_indices_local) > 0:
                        remove_indices_global = idx_check[remove_indices_local]

                        # Apply removal
                        keep_mask = np.ones(len(M), dtype=bool)
                        keep_mask[remove_indices_global] = False
                        M = M[keep_mask]

            # --- 2) Add new observations ---
            if pcd_world is not None and len(pcd_world) > 0:
                M = self._merge_dedup(M, pcd_world, self.merge_radius)

            self._M = M

    def _filter_points(
        self,
        points: np.ndarray,
        robot_filter_cb: Callable[
            [np.ndarray, dict], Tuple[List[List[float]], Optional[object]]
        ],
        joint_dict: dict,
    ) -> np.ndarray:
        if points is None or points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        pts = points.astype(np.float32)

        # Ground filter
        if self.enable_ground_filter:
            pts = pts[pts[:, 2] > self.ground_z_threshold]
            if pts.shape[0] == 0:
                return pts

        # Robot self filter with synchronized state
        if pts.shape[0] > 0:
            filtered_list, _ = robot_filter_cb(pts, joint_dict)
            try:
                pts = np.asarray(filtered_list, dtype=np.float32)
            except Exception:
                pass
            if pts.shape[0] == 0:
                return pts

        # Remove isolated noise points (more aggressive)
        if pts.shape[0] > 10:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            # pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.15)
            pts = np.asarray(pcd.points, dtype=np.float32)

        return pts
