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
import sys
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
from src.visualization.plots import (
    plot_summary_grid,
    plot_training_curves,
    plot_accuracy_heatmap,
)
from src.utils import (
    seed_everything,
    RunLogger,
    cleanup_checkpoints,
    prepare_run_config,
)


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

    cfg = prepare_run_config(cfg, seed)
    run_name = cfg["run_name"]

    logger = RunLogger(
        cfg["log_dir"],
        run_name,
        metrics_dir=cfg["metrics_dir"],
        reset_log=not cfg.get("resume", False),
    )
    logger.print(f"Starting: {run_name}")
    logger.print(f"Config: {cfg}")

    if not str(device).startswith("cuda"):
        raise ValueError(
            "research_v2 is CUDA-only. Use a CUDA device string such as 'cuda' or 'cuda:0'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but research_v2 requires CUDA.")
    dev = torch.device(device)
    logger.print(f"Device: {dev}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_kwargs = {
        "root": cfg["data_root"],
        "batch_size": cfg.get("batch_size", 32),
        "num_workers": cfg.get("num_workers", 2),
    }
    if "n_tasks" in cfg:
        dataset_kwargs["n_tasks"] = cfg.get("n_tasks")
    if "perm_seed" in cfg:
        dataset_kwargs["seed"] = cfg.get("perm_seed")
    if cfg.get("dataset_image_size") is not None:
        dataset_kwargs["image_size"] = cfg.get("dataset_image_size")
    if cfg.get("dataset_out_channels") is not None:
        dataset_kwargs["out_channels"] = cfg.get("dataset_out_channels")
    dataset = get_dataset(cfg["dataset"], **dataset_kwargs)
    logger.print(f"Dataset: {dataset}")

    # ── Model ─────────────────────────────────────────────────────────────
    in_channels = dataset.input_size[0]
    model = get_model(
        cfg["model"],
        in_channels = in_channels,
        pretrained  = cfg.get("pretrained", False),
        image_size  = cfg.get("model_image_size", cfg.get("dataset_image_size")),
    ).to(dev)

    # ── Method ────────────────────────────────────────────────────────────
    method = get_method(cfg["method"], model, cfg, dev)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = CLTrainer(method, dataset, cfg, logger, dev)
    start_task = 0
    if cfg.get("resume", False):
        start_task = trainer.resume_from_latest()
        if start_task >= dataset.n_tasks:
            logger.print("Checkpoint already contains a completed run.")
    results = trainer.train(
        start_task=start_task,
        stop_after_task=cfg.get("stop_after_task"),
    )

    if results.get("stopped_early"):
        logger.print(
            "Run stopped intentionally before completion; final metrics and plots "
            "were skipped so this checkpoint can be resumed later."
        )
        return {
            "run_name": run_name,
            "stopped_early": True,
            "stopped_after_task": results.get("stopped_after_task"),
        }

    acc_matrix = results["acc_matrix"]
    metrics    = results["metrics"]

    # ── Visualisation ─────────────────────────────────────────────────────
    if not cfg.get("disable_plots", False):
        fig_dir = cfg["figure_dir"]
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

    if (
        not cfg.get("disable_checkpoints", False)
        and cfg.get("cleanup_checkpoints_on_success", False)
    ):
        deleted = cleanup_checkpoints(
            cfg["checkpoint_dir"],
            run_name,
        )
        logger.print(f"Deleted {deleted} checkpoint(s) after successful completion.")

    return {"run_name": run_name, "metrics": metrics, "acc_matrix": acc_matrix}


# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CL experiment.")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--device", default="cuda",
                        help="'cuda' or 'cuda:0' etc. (default: cuda)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest completed task checkpoint, if present.")
    parser.add_argument("--cleanup-checkpoints", action="store_true",
                        help="Delete this run's task checkpoints after a successful completed run.")
    parser.add_argument("--stop-after-task", type=int, default=None,
                        help="Intentionally stop after saving the checkpoint for this task index.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume"] = True
    if args.cleanup_checkpoints:
        cfg["cleanup_checkpoints_on_success"] = True
    if args.stop_after_task is not None:
        cfg["stop_after_task"] = args.stop_after_task
    run(cfg, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
