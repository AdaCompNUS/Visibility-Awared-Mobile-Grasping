import numpy as np

from grasp_anywhere.robot.utils.fetch_planning_backend import (
    FetchPlanningBackend,
    _resample,
    _whole_body_steps,
)

TUCK = [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0]


def _wall(x, y_min, y_max, spacing=0.02):
    """Vertical wall of points at ``x`` spanning [y_min, y_max] x [0, 1.5]."""
    y, z = np.meshgrid(
        np.arange(y_min, y_max, spacing), np.arange(0.0, 1.5, spacing)
    )
    return np.c_[np.full(y.size, x), y.ravel(), z.ravel()]


def test_resample_keeps_endpoints_and_spacing():
    q = np.zeros((1001, 11))
    q[:, 0] = np.linspace(0.0, 1.0, 1001)
    out = _resample(q, _whole_body_steps(q), 0.03)
    assert np.allclose(out[0], q[0]) and np.allclose(out[-1], q[-1])
    # Rows are picked, not interpolated: steps overshoot by at most one row.
    assert np.diff(out[:, 0]).max() <= 0.03 + 0.001 + 1e-9


def test_path_check_catches_collision_between_waypoints():
    backend = FetchPlanningBackend()
    backend.add_pointcloud(_wall(1.0, -1.0, 1.0).tolist(), 0.03)
    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert backend.validate(TUCK, base[0]) and backend.validate(TUCK, base[1])
    assert backend.path_in_collision([TUCK, TUCK], base, -1)
    backend.clear_pointclouds()
    assert not backend.path_in_collision([TUCK, TUCK], base, -1)


def test_whole_body_plan_goes_around_wall_forward_only():
    backend = FetchPlanningBackend()
    backend.add_pointcloud(_wall(1.0, -1.0, 1.0).tolist(), 0.03)
    res = backend.plan_whole_body(TUCK, TUCK, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    assert res["success"]
    base = np.asarray(res["base_configs"])
    assert np.allclose(base[0], [0.0, 0.0, 0.0]) and np.allclose(base[-1], [2.0, 0.0, 0.0])
    assert not backend.path_in_collision(res["arm_path"], base, -1)
    # Nonholonomic, forward-only base: every step moves along the heading.
    step = np.diff(base[:, :2], axis=0)
    heading = np.c_[np.cos(base[:-1, 2]), np.sin(base[:-1, 2])]
    lateral = np.abs(heading[:, 0] * step[:, 1] - heading[:, 1] * step[:, 0])
    assert lateral.max() < 1e-2
    assert (np.einsum("ij,ij->i", heading, step) > -1e-4).all()


def test_filter_robot_keeps_points_on_registered_obstacles():
    backend = FetchPlanningBackend()
    wall = _wall(1.0, -1.0, 1.0)
    backend.add_pointcloud(wall.tolist(), 0.03)
    near_robot = [[0.0, 0.0, 0.2]]  # inside the base
    kept = backend.filter_robot(wall.tolist() + near_robot, TUCK, [0.0, 0.0, 0.0], 0.03)
    assert len(kept) == len(wall)


def test_arm_plan_restores_whole_body_validation():
    backend = FetchPlanningBackend()
    goal = [0.35, 0.3, 0.5, -0.2, 1.2, 0.0, 1.0, 0.0]
    path, stats = backend.plan_arm([0.0, 0.0, 0.0], np.array(TUCK), np.array(goal))
    assert path is not None and stats["status"] == "success"
    assert np.allclose(path[0], TUCK) and np.allclose(path[-1], goal)
    assert backend.validate(TUCK, [0.0, 0.0, 0.0])
