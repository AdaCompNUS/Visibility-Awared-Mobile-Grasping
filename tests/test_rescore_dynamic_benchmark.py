from experiments.rescore_dynamic_benchmark import (
    path_changed_at_threshold,
    rescore_results,
)


def _source_result(deviation: float = 0.09):
    return {
        "config": {
            "benchmark": {
                "enable_dynamic_challenges": True,
                "require_dynamic_path_change": True,
                "nav_path_change_threshold": 0.08,
            }
        },
        "scenes": {
            "scene_0": {
                "scene_success_rate": 1.0,
                "tasks": [
                    {
                        "success": True,
                        "failure_reason": None,
                        "dynamic_path_affected": True,
                        "dynamic_metrics": {
                            "obstacle_spawned": True,
                            "spawn_route_distance_m": 0.01,
                            "maximum_route_deviation_m": deviation,
                            "path_changed": True,
                            "replan_count": 2,
                        },
                    }
                ],
            }
        },
    }


def test_path_change_uses_both_recorded_distances():
    metrics = _source_result(0.2)["scenes"]["scene_0"]["tasks"][0]["dynamic_metrics"]
    assert path_changed_at_threshold(metrics, 0.10)
    metrics["spawn_route_distance_m"] = 0.11
    assert not path_changed_at_threshold(metrics, 0.10)


def test_stricter_threshold_turns_success_into_dynamic_failure():
    rescored = rescore_results(
        _source_result(), 0.10, source_path="source.json", source_sha256="abc"
    )
    task = rescored["scenes"]["scene_0"]["tasks"][0]
    assert not task["success"]
    assert task["failure_reason"] == "dynamic_path_unaffected"
    assert rescored["summary"]["successful_tasks"] == 0
    assert rescored["summary"]["dynamic_interaction_failures"] == 1
    assert rescored["rescore"]["source_sha256"] == "abc"


def test_looser_threshold_is_rejected():
    try:
        rescore_results(
            _source_result(), 0.07, source_path="source.json", source_sha256="abc"
        )
    except ValueError as exc:
        assert "stricter" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
