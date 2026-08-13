from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial import cKDTree

from grasp_anywhere.benchmark.dynamic_benchmark_manager import (
    DynamicBenchmarkManager,
)


def _manager_with_obstacle(obstacle_xy):
    manager = DynamicBenchmarkManager(SimpleNamespace())
    manager.enabled = True
    manager._triggered_nav = True
    actor = SimpleNamespace(
        pose=SimpleNamespace(p=np.array([obstacle_xy[0], obstacle_xy[1], 0.875]))
    )
    manager._dynamic_actors = [{"type": "moving_pedestrian", "actor": actor}]
    manager._nav_metrics.update(
        {"obstacle_spawned": True, "spawn_route_distance_m": 0.0}
    )
    return manager


def test_point_to_route_distance_uses_segments():
    route = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    distance = DynamicBenchmarkManager._minimum_point_to_polyline_distance(
        np.array([1.0, 0.5]), route
    )
    assert np.isclose(distance, 0.5)


def test_replan_records_spatial_detour_around_obstacle():
    manager = _manager_with_obstacle([1.0, 0.0])
    previous = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    detour = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.7, 0.0],
        [1.5, 0.5, 0.0],
        [2.0, 0.0, 0.0],
    ]

    manager.record_navigation_replan(previous, detour, [0.0, 0.0])

    metrics = manager.get_task_metrics()
    assert metrics["replan_count"] == 1
    assert metrics["path_changed"] is True
    assert metrics["maximum_path_clearance_gain_m"] >= 0.4
    assert metrics["maximum_route_deviation_m"] >= 0.4


def test_same_route_is_not_counted_as_dynamic_effect():
    manager = _manager_with_obstacle([1.0, 0.0])
    route = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    manager.record_navigation_replan(route, route, [0.0, 0.0])

    metrics = manager.get_task_metrics()
    assert metrics["replan_count"] == 1
    assert metrics["path_changed"] is False


def test_connectivity_accepts_diagonal_navmesh_detour():
    manager = DynamicBenchmarkManager(SimpleNamespace())
    vertices = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    manager._navmesh_initialized = True
    manager._nav_vertices = vertices
    manager._nav_vertex_grid = vertices.astype(np.int32)
    manager._nav_mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    manager._nav_origin = np.zeros(2, dtype=np.float32)
    manager._nav_resolution = 1.0
    manager._nav_tree = cKDTree(vertices)
    manager._navigation_goal = np.array([2.0, 2.0], dtype=np.float32)

    assert manager._connectivity_safe([0.0, 2.0], [0.0, 0.0], 0.1)


def test_adaptive_scales_try_largest_and_include_minimum():
    scales = DynamicBenchmarkManager._descending_scales(1.125, 0.625)
    assert scales == [1.125, 1.0, 0.875, 0.75, 0.625]


def test_crossing_direction_is_perpendicular_to_route():
    route = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    direction = DynamicBenchmarkManager._route_crossing_direction(
        np.array([1.0, 0.0]), route
    )
    assert np.allclose(direction, [0.0, 1.0])


def test_pedestrian_advances_in_fixed_direction_not_around_robot():
    manager = DynamicBenchmarkManager(SimpleNamespace())
    yy, xx = np.mgrid[0:7, 0:7]
    vertices = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    manager._navmesh_initialized = True
    manager._nav_vertices = vertices
    manager._nav_vertex_grid = vertices.astype(np.int32)
    manager._nav_mask = np.ones((7, 7), dtype=bool)
    manager._nav_origin = np.zeros(2, dtype=np.float32)
    manager._nav_resolution = 1.0
    manager._nav_tree = cKDTree(vertices)
    manager._navigation_goal = np.array([6.0, 6.0], dtype=np.float32)

    actor = SimpleNamespace(pose=SimpleNamespace(p=np.array([3.0, 3.0, 0.875])))
    item = {
        "actor": actor,
        "travel_direction": np.array([1.0, 0.0], dtype=np.float32),
        "direction_locked": False,
        "route_xy": np.array([[3.0, 0.0], [3.0, 6.0]], dtype=np.float32),
        "previous_grid": None,
        "clearance": 0.1,
    }

    path = manager._select_motion_path(item, np.array([0.0, 0.0]))

    assert np.allclose(path, [[4.0, 3.0]])
    assert item["direction_locked"] is True
    assert np.allclose(item["travel_direction"], [1.0, 0.0])


def test_static_and_dynamic_configs_only_differ_in_benchmark_section():
    with open("grasp_anywhere/configs/maniskill_fetch.yaml") as static_file:
        static_config = yaml.safe_load(static_file)
    with open(
        "grasp_anywhere/configs/maniskill_fetch_dynamic_easy.yaml"
    ) as dynamic_file:
        dynamic_config = yaml.safe_load(dynamic_file)

    static_config.pop("benchmark")
    dynamic_config.pop("benchmark")
    assert dynamic_config == static_config
