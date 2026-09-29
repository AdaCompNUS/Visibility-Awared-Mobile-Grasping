from pathlib import Path

from experiments.aggregate_paper_campaign import summarize_runs, wilson_interval
from experiments.run_maniskill_benchmark import derive_task_seed
from experiments.run_paper_campaign import (
    CORE_METHOD_ORDER,
    build_run_matrix,
    config_without_benchmark,
    load_yaml,
    prepare_configs,
)


def test_task_seed_is_stable_and_task_specific():
    seed = derive_task_seed(1234, "scene_7", 9)
    assert seed == derive_task_seed(1234, "scene_7", 9)
    assert seed != derive_task_seed(1235, "scene_7", 9)
    assert seed != derive_task_seed(1234, "scene_7", 10)
    assert 0 <= seed < 2**32


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(120, 200)
    assert low < 0.60 < high
    assert 0.0 <= low <= high <= 1.0


def test_run_summary_reports_mean_std_and_pooled_interval():
    runs = [
        {"summary": {"total_tasks": 400, "successful_tasks": value}}
        for value in (280, 284, 288, 282, 286)
    ]
    summary = summarize_runs(runs)
    assert summary["runs"] == 5
    assert summary["pooled_total"] == 2000
    assert summary["pooled_successes"] == 1420
    assert summary["run_mean_success_rate"] == 0.71
    assert summary["run_std_success_rate"] > 0.0


def test_campaign_generates_all_core_static_dynamic_pairs(tmp_path):
    campaign = tmp_path / "campaign"
    configs = prepare_configs(campaign, "core")
    assert len(configs) == 2 * len(CORE_METHOD_ORDER)
    for method in CORE_METHOD_ORDER:
        static = load_yaml(configs[(method, "static")])
        dynamic = load_yaml(configs[(method, "dynamic")])
        assert not static["benchmark"]["enable_dynamic_challenges"]
        assert dynamic["benchmark"]["enable_dynamic_challenges"]
        assert config_without_benchmark(static) == config_without_benchmark(dynamic)

    matrix = build_run_matrix(configs, repeats=5, scope="core", base_seed=100)
    assert len(matrix) == 60
    assert matrix[0]["method"] == "ours"
    assert matrix[0]["condition"] == "static"
    assert matrix[0]["run_seed"] == 100
    assert matrix[4]["run_seed"] == 104
    assert all(Path(entry["config_path"]).parts[0] == "configs" for entry in matrix)
