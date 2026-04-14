"""Phase 1 infrastructure checks for the v2 workspace.

These tests stay intentionally lightweight. They verify the rebuilt
scaffold, path conventions, logging, checkpointing, and core metric math
without requiring full dataset downloads or long model runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_experiment import load_config
from src.metrics.continual_metrics import compute_all_metrics
from src.utils import (
    RunLogger,
    cleanup_checkpoints,
    latest_checkpoint,
    load_checkpoint,
    prepare_run_config,
    save_checkpoint,
    seed_everything,
)


def test_phase1_core_imports() -> None:
    from src.datasets import get_dataset
    from src.methods import get_method
    from src.models import get_model
    from src.trainers.cl_trainer import CLTrainer
    from src.visualization.plots import plot_summary_grid

    assert get_dataset is not None
    assert get_method is not None
    assert get_model is not None
    assert CLTrainer is not None
    assert plot_summary_grid is not None


def test_seed_everything_is_deterministic() -> None:
    seed_everything(42)
    first = torch.rand(4)
    seed_everything(42)
    second = torch.rand(4)
    assert torch.allclose(first, second)


def test_metric_values() -> None:
    acc_matrix = np.array(
        [
            [90.0, np.nan, np.nan],
            [70.0, 85.0, np.nan],
            [60.0, 75.0, 88.0],
        ]
    )
    metrics = compute_all_metrics(acc_matrix, baseline_acc=[10.0, 10.0, 10.0], n_tasks=3)
    assert abs(metrics["forgetting"] - 20.0) < 0.01
    assert abs(metrics["backward_transfer"] + 20.0) < 0.01
    assert abs(metrics["avg_accuracy"] - 74.333) < 0.1
    assert isinstance(metrics["forward_transfer"], float)


def test_prepare_run_config_primary_layout() -> None:
    cfg = {
        "dataset": "split_cifar10",
        "method": "der",
        "n_epochs": 5,
        "results_root": "results",
        "data_root": "data_local",
    }
    prepared = prepare_run_config(cfg, seed=42)
    assert prepared["run_name"] == "split_cifar10_der_seed42"
    assert prepared["log_dir"].endswith(
        "results\\runs\\epoch_5\\split_cifar10\\hybrids\\der\\seed_42\\logs"
    )
    assert prepared["metrics_dir"].endswith(
        "results\\runs\\epoch_5\\split_cifar10\\hybrids\\der\\seed_42\\metrics"
    )
    assert prepared["figure_dir"].endswith(
        "results\\figures\\epoch_5\\split_cifar10\\hybrids\\der\\seed_42"
    )


def test_prepare_run_config_ablation_layout() -> None:
    cfg = {
        "dataset": "split_cifar100",
        "method": "si_der",
        "n_epochs": 1,
        "results_root": "results",
        "result_group": "ablations",
        "ablation_family": "buffer",
        "run_tag": "buffer_500",
    }
    prepared = prepare_run_config(cfg, seed=123)
    assert prepared["run_name"] == "split_cifar100_si_der_buffer_500_seed123"
    assert prepared["log_dir"].endswith(
        "results\\ablations\\epoch_1\\split_cifar100\\buffer\\si_der__buffer_500\\seed_123\\logs"
    )
    assert prepared["figure_dir"].endswith(
        "results\\figures\\ablations\\epoch_1\\split_cifar100\\buffer\\si_der__buffer_500\\seed_123"
    )


def test_logger_separates_logs_and_metrics(tmp_path: Path) -> None:
    logger = RunLogger(
        str(tmp_path / "logs"),
        "demo_run",
        metrics_dir=str(tmp_path / "metrics"),
    )
    logger.log({"task": 0, "epoch": 0, "train_loss": 1.234})
    logger.print("hello world")

    log_path = tmp_path / "logs" / "demo_run.log"
    metrics_path = tmp_path / "metrics" / "demo_run_metrics.csv"
    assert log_path.exists()
    assert metrics_path.exists()
    assert "hello world" in log_path.read_text(encoding="utf-8")
    assert "train_loss" in metrics_path.read_text(encoding="utf-8")


def test_checkpoint_roundtrip_and_cleanup(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_path = ckpt_dir / "demo_task0.pt"
    state = {"task_id": 0, "value": 123}

    save_checkpoint(state, str(ckpt_path))
    restored = load_checkpoint(str(ckpt_path), device="cuda")
    assert restored["value"] == 123
    assert latest_checkpoint(str(ckpt_dir), "demo") == str(ckpt_path)
    assert cleanup_checkpoints(str(ckpt_dir), "demo") == 1
    assert latest_checkpoint(str(ckpt_dir), "demo") is None


def test_load_config_uses_base_defaults() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "split_cifar10_der.yaml"
    )
    cfg = load_config(str(config_path))
    assert cfg["results_root"] == "results"
    assert cfg["data_root"] == "data_local"
    assert cfg["method"] == "der"
