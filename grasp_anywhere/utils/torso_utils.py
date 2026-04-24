from __future__ import annotations

import numpy as np

from grasp_anywhere.dataclass.torso_map import TorsoMap


def query_best_torso(
    torso_map: TorsoMap,
    target_pos_world: np.ndarray,
    base_config: np.ndarray,
) -> float:
    """
    Query a torso height (meters) from a target pose in WORLD frame plus a base config.

    Args:
        target_pos_world:
            Target position in WORLD frame as `[x, y, z]` (meters).
        base_config:
            Robot base pose in WORLD frame as `[x, y, theta]` where theta is yaw (radians).

    This function converts the world-frame target position into the robot/base frame
    induced by `base_config`, then performs an exact-bin torso-map lookup.

    Returns:
        best_torso_m: float (meters)
    """
    target = np.asarray(target_pos_world, dtype=np.float64).reshape(-1)
    if target.size != 3:
        raise ValueError(f"target_pos_world must be [x,y,z]. Got len={target.size}")
    x_t, y_t, z_t = float(target[0]), float(target[1]), float(target[2])

    base = np.asarray(base_config, dtype=np.float64).reshape(-1)
    if base.size != 3:
        raise ValueError(f"base_config must be [x,y,theta]. Got len={base.size}")
    x_b, y_b, theta = float(base[0]), float(base[1]), float(base[2])

    # Transform world target into base frame (2D base on ground, yaw-only).
    # rel_xy = Rz(-theta) * (p_w - p_base_w)
    dx = x_t - x_b
    dy = y_t - y_b
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_z = z_t
    pos_base = np.array([rel_x, rel_y, rel_z], dtype=np.float64)

    if not torso_map.xyz_to_index:
        raise ValueError(
            "TorsoMap.xyz_to_index is empty. This TorsoMap instance is invalid for exact-bin queries."
        )

    # Discretize to the map's xyz bin index (same convention as the map builder):
    #   ijk = round(xyz / xyz_step)
    ijk = np.rint(pos_base / float(torso_map.xyz_step)).astype(np.int32)
    key = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
    idx = torso_map.xyz_to_index.get(key)
    if idx is None:
        return 2.0

    return float(torso_map.best_torso[idx])


if __name__ == "__main__":
    import pickle

    import open3d as o3d

    torso_map = TorsoMap.from_file("resources/torso_map.pkl")

    base_cfg = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # [x,y,theta] in world
    base_xy = (float(base_cfg[0]), float(base_cfg[1]))
    base_theta = float(base_cfg[2])

    rng = np.random.default_rng(0)
    samples: list[np.ndarray] = []
    torsos: list[float] = []

    # Pick targets directly from the map (guaranteed to exist).
    # Enforce "in front": x_base > 0, and height range: 0.2 <= z_base <= 1.8.
    # Note: TorsoMap dataclass is intentionally slim and doesn't store xyz centers.
    # For visualization only, read xyz centers directly from the pickle payload.
    with open("resources/torso_map.pkl", "rb") as f:
        payload = pickle.load(f)
    xyz_base_all = np.asarray(payload["xyz"], dtype=np.float64)
    front_mask = (
        (xyz_base_all[:, 0] > 0.05)
        & (xyz_base_all[:, 2] >= 0.2)
        & (xyz_base_all[:, 2] <= 1.8)
    )
    front = xyz_base_all[front_mask]

    c = float(np.cos(base_theta))
    s = float(np.sin(base_theta))
    for _ in range(30):
        p_b = front[int(rng.integers(0, front.shape[0]))]
        # base -> world: p_w = Rz(theta)*p_b + base_xy
        x_w = c * float(p_b[0]) - s * float(p_b[1]) + base_xy[0]
        y_w = s * float(p_b[0]) + c * float(p_b[1]) + base_xy[1]
        z_w = float(p_b[2])
        p_w = np.array([x_w, y_w, z_w], dtype=np.float64)
        t = query_best_torso(torso_map, p_w, base_cfg)
        samples.append(p_w)
        torsos.append(float(t))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Torso Map Test (SPACE = next)", width=1200, height=800
    )

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.02, origin=[0, 0, 0]
    )
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.02, origin=[0, 0, 0]
    )
    base_frame.translate([base_xy[0], base_xy[1], 0.0])
    base_frame.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, base_theta)),
        center=(base_xy[0], base_xy[1], 0.0),
    )

    # Mutable state for callbacks
    state = {"i": 0, "geoms": [], "first": True}  # type: ignore[var-annotated]

    def show(i: int) -> None:
        # Clear previous geometries (except frames/base box)
        for g in state["geoms"]:
            vis.remove_geometry(g, reset_bounding_box=False)
        state["geoms"].clear()

        target = samples[i]
        torso_h = torsos[i]
        target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        target_sphere.paint_uniform_color([1.0, 0.2, 0.2])
        target_sphere.translate(target)

        p0 = np.array([base_xy[0], base_xy[1], 0.0], dtype=np.float64)
        p1 = np.array([base_xy[0], base_xy[1], float(torso_h)], dtype=np.float64)
        torso_line = o3d.geometry.LineSet()
        torso_line.points = o3d.utility.Vector3dVector(np.stack([p0, p1], axis=0))
        torso_line.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1]], dtype=np.int32)
        )
        torso_line.colors = o3d.utility.Vector3dVector(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        )

        # "Robot height" marker: a blue sphere at the torso height above base.
        height_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        height_marker.paint_uniform_color([0.2, 0.2, 1.0])
        height_marker.translate([base_xy[0], base_xy[1], torso_h])

        for g in (target_sphere, torso_line, height_marker):
            vis.add_geometry(g, reset_bounding_box=bool(state["first"]))
            state["geoms"].append(g)

        print(f"[{i+1}/{len(samples)}] target={target.tolist()} torso={torso_h:.4f}m")
        vis.poll_events()
        vis.update_renderer()

        if state["first"]:
            # Make sure the camera is set to something sensible and the scene fits.
            vis.reset_view_point(True)
            vc = vis.get_view_control()
            if vc is not None:
                vc.set_lookat([base_xy[0] + 0.6, base_xy[1], 0.6])
                vc.set_front([-1.0, 0.0, -0.3])
                vc.set_up([0.0, 0.0, 1.0])
                vc.set_zoom(0.6)
            vis.poll_events()
            vis.update_renderer()
            state["first"] = False

    def on_space(_vis: o3d.visualization.Visualizer) -> bool:
        state["i"] = (state["i"] + 1) % len(samples)
        show(state["i"])
        return False

    # Open3D key codes: space=32
    vis.register_key_callback(32, on_space)

    for g in (world_frame, base_frame):
        vis.add_geometry(g, reset_bounding_box=True)

    show(0)
    vis.run()
    vis.destroy_window()
