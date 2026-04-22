from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.phase7 import (
    build_dataset_leaders,
    build_summary_tables,
    effect_size_magnitude,
    holm_adjust,
    rank_biserial_effect,
    run_phase7_pipeline,
    safe_wilcoxon,
)
from src.utils.paths import PROJECT_ROOT, RESULTS_ROOT


def test_holm_adjust_preserves_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.01, 0.03, 0.04])
    assert adjusted == [0.03, 0.06, 0.06]


def test_safe_wilcoxon_handles_equal_samples() -> None:
    statistic, p_value, n_nonzero = safe_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert statistic == 0.0
    assert p_value == 1.0
    assert n_nonzero == 0


def test_rank_biserial_effect_and_magnitude() -> None:
    effect = rank_biserial_effect([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
    assert effect > 0.9
    assert effect_size_magnitude(effect) == "large"


def test_summary_table_aggregation_and_leaders(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {
                "dataset": "split_cifar10",
                "method": "agem",
                "seed": 42,
                "run_name": "run_a",
                "config_name": "cfg.yaml",
                "model": "slim_resnet18",
                "n_epochs": 1,
                "batch_size": 32,
                "num_workers": 0,
                "fp16": False,
                "buffer_size": 200,
                "total_time_sec": 100.0,
                "log_starts": 1,
                "completed": True,
                "caveat_note": "",
                "avg_accuracy": 20.0,
                "forgetting": 40.0,
                "backward_transfer": -40.0,
                "forward_transfer": -10.0,
            },
            {
                "dataset": "split_cifar10",
                "method": "agem",
                "seed": 123,
                "run_name": "run_b",
                "config_name": "cfg.yaml",
                "model": "slim_resnet18",
                "n_epochs": 1,
                "batch_size": 32,
                "num_workers": 0,
                "fp16": False,
                "buffer_size": 200,
                "total_time_sec": 120.0,
                "log_starts": 1,
                "completed": True,
                "caveat_note": "",
                "avg_accuracy": 24.0,
                "forgetting": 38.0,
                "backward_transfer": -38.0,
                "forward_transfer": -9.0,
            },
            {
                "dataset": "split_cifar10",
                "method": "der",
                "seed": 42,
                "run_name": "run_c",
                "config_name": "cfg.yaml",
                "model": "slim_resnet18",
                "n_epochs": 1,
                "batch_size": 32,
                "num_workers": 0,
                "fp16": False,
                "buffer_size": 200,
                "total_time_sec": 90.0,
                "log_starts": 1,
                "completed": True,
                "caveat_note": "",
                "avg_accuracy": 28.0,
                "forgetting": 32.0,
                "backward_transfer": -32.0,
                "forward_transfer": -8.0,
            },
            {
                "dataset": "split_cifar10",
                "method": "der",
                "seed": 123,
                "run_name": "run_d",
                "config_name": "cfg.yaml",
                "model": "slim_resnet18",
                "n_epochs": 1,
                "batch_size": 32,
                "num_workers": 0,
                "fp16": False,
                "buffer_size": 200,
                "total_time_sec": 95.0,
                "log_starts": 1,
                "completed": True,
                "caveat_note": "",
                "avg_accuracy": 30.0,
                "forgetting": 31.0,
                "backward_transfer": -31.0,
                "forward_transfer": -7.5,
            },
        ]
    )
    summary, _, _ = build_summary_tables(master, tmp_path, "epoch_1")
    assert len(summary) == 2
    der_row = summary.loc[summary["method"].astype(str) == "der"].iloc[0]
    assert abs(float(der_row["avg_accuracy_mean"]) - 29.0) < 1e-6

    pairwise = pd.DataFrame(
        [
            {
                "dataset": "split_cifar10",
                "metric": "avg_accuracy",
                "method_a": "agem",
                "method_b": "der",
                "n_pairs": 2,
                "n_nonzero_pairs": 2,
                "mean_a": 22.0,
                "mean_b": 29.0,
                "mean_diff": -7.0,
                "statistic": 0.0,
                "p_value": 0.02,
                "holm_adjusted_p_value": 0.02,
                "reject_h0": True,
            }
        ]
    )
    leaders = build_dataset_leaders(summary, pairwise, tmp_path, "epoch_1")
    assert leaders.iloc[0]["best_method"] == "der"
    assert leaders.iloc[0]["top_cluster_methods"] == "der"


def test_phase7_pipeline_smoke_uses_existing_epoch1_outputs() -> None:
    outputs = run_phase7_pipeline(PROJECT_ROOT, RESULTS_ROOT, epoch=1)
    assert len(outputs["master"]) == 240
    assert (RESULTS_ROOT / "analysis" / "epoch_1" / "master_results.csv").exists()
    assert (RESULTS_ROOT / "analysis" / "epoch_1" / "phase7_report.md").exists()
