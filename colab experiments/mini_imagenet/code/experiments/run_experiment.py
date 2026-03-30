"""CLI entry point for running a single CL experiment.

Usage::

    python experiments/run_experiment.py \\
        --config experiments/configs/split_cifar100_der.yaml \\
        --seed 42 \\
        --device cuda

Multiple seeds are batched by the companion script ``run_all_seeds.py``
or run manually with different --seed values.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# ── Make src importable when running from the project root ────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets    import get_dataset
from src.models      import get_model
from src.methods     import get_method
from src.trainers.cl_trainer import CLTrainer
from src.metrics.continual_metrics import compute_all_metrics
from src.visualization.plots import (
    plot_summary_grid,
    plot_all_metric_bars,
    plot_training_curves,
    plot_accuracy_heatmap,
)
from src.utils import seed_everything, RunLogger


# ===========================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """Merge base_config.yaml → experiment yaml (experiment overrides base)."""
    base_path = Path(__file__).parent / "configs" / "base_config.yaml"
    with open(base_path) as fh:
        base = yaml.safe_load(fh).get("base_config", {})
    with open(config_path) as fh:
        exp  = yaml.safe_load(fh)
    # Flatten nested experiment config and override base.
    merged = {**base, **exp}
    return merged


# ===========================================================================
def run(cfg: Dict[str, Any], seed: int, device: str) -> Dict[str, Any]:
    """Execute a single experiment (one seed)."""
    seed_everything(seed)

    # ── Build run name ─────────────────────────────────────────────────────
    run_tag = cfg.get("run_tag")
    run_name = f"{cfg['dataset']}_{cfg['method']}"
    if run_tag:
        run_name = f"{run_name}_{run_tag}"
    run_name = f"{run_name}_seed{seed}"
    cfg["run_name"] = run_name
    cfg["seed"]     = seed

    logger = RunLogger(cfg.get("log_dir", "results/raw"), run_name)
    logger.print(f"Starting: {run_name}")
    logger.print(f"Config: {cfg}")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.print(f"Device: {dev}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_kwargs = {
        "root": cfg.get("data_root", "data"),
        "batch_size": cfg.get("batch_size", 32),
        "num_workers": cfg.get("num_workers", 2),
    }
    if "image_size" in cfg:
        dataset_kwargs["image_size"] = cfg.get("image_size")
    if "n_tasks" in cfg:
        dataset_kwargs["n_tasks"] = cfg.get("n_tasks")
    if "perm_seed" in cfg:
        dataset_kwargs["seed"] = cfg.get("perm_seed")
    dataset = get_dataset(cfg["dataset"], **dataset_kwargs)
    logger.print(f"Dataset: {dataset}")

    # ── Model ─────────────────────────────────────────────────────────────
    in_channels = dataset.input_size[0]
    model = get_model(
        cfg["model"],
        in_channels = in_channels,
        pretrained  = cfg.get("pretrained", False),
    ).to(dev)

    # ── Method ────────────────────────────────────────────────────────────
    method = get_method(cfg["method"], model, cfg, dev)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = CLTrainer(method, dataset, cfg, logger, dev)
    results = trainer.train()

    acc_matrix = results["acc_matrix"]
    metrics    = results["metrics"]

    # ── Visualisation ─────────────────────────────────────────────────────
    if not cfg.get("disable_plots", False):
        fig_dir = cfg.get("figure_dir", "results/figures")
        run_tag = cfg.get("run_tag")
        method_label = cfg["method"]
        if run_tag:
            method_label = f"{method_label}_{run_tag}"
        method_label = f"{method_label}_seed{seed}"
        plot_summary_grid(
            acc_matrix, metrics,
            method_name  = method_label,
            dataset_name = cfg["dataset"],
            fig_dir      = fig_dir,
        )
        plot_training_curves(
            acc_matrix,
            method_name  = method_label,
            dataset_name = cfg["dataset"],
            fig_dir      = fig_dir,
        )
        plot_accuracy_heatmap(
            acc_matrix,
            method_name  = method_label,
            dataset_name = cfg["dataset"],
            fig_dir      = fig_dir,
        )

    logger.print(f"\nDone - run: {run_name}")
    logger.print(f"Metrics: {metrics}")

    return {"run_name": run_name, "metrics": metrics, "acc_matrix": acc_matrix}


# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CL experiment.")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--device", default="cuda",
                        help="'cuda', 'cpu', or 'cuda:0' etc. (default: cuda)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
