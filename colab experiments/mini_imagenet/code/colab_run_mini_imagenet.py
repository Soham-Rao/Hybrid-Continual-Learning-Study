#!/usr/bin/env python
"""Colab runner for Split Mini-ImageNet (all methods, progress via tqdm)."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from experiments.run_experiment import run
from src.visualization.plots import plot_all_metric_bars, plot_pareto_frontier


BASELINE_METHODS = ["fine_tune", "joint_training", "ewc", "agem", "lwf"]
HYBRID_METHODS   = ["der", "xder", "icarl", "er_ewc", "progress_compress", "agem_distill", "si_der"]
ALL_METHODS = BASELINE_METHODS + HYBRID_METHODS


def load_base_config() -> Dict:
    base_path = Path(__file__).parent / "experiments" / "configs" / "base_config.yaml"
    with open(base_path, "r") as fh:
        return yaml.safe_load(fh)["base_config"]


def build_cfg(base: Dict, args, method: str) -> Dict:
    cfg = dict(base)
    cfg.update({
        "dataset": "split_mini_imagenet",
        "model": args.model,
        "method": method,
        "data_root": args.data_root,
        "results_root": args.results_root,
        "log_dir": str(Path(args.results_root) / "raw"),
        "figure_dir": str(Path(args.results_root) / "figures"),
        "checkpoint_dir": str(Path(args.results_root) / "checkpoints"),
        "disable_tqdm": False,
        "disable_plots": not args.enable_plots,
        "disable_checkpoints": not args.enable_checkpoints,
        "fp16": args.fp16,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pretrained": args.pretrained,
        "run_tag": args.run_tag,
    })
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    return cfg


def run_all(args) -> None:
    base = load_base_config()
    methods = args.methods.split(",") if args.methods else ALL_METHODS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [42, 123, 456, 789, 1024]

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    aggregated: Dict[str, Dict[str, Dict]] = {"split_mini_imagenet": {}}

    outer = tqdm(methods, desc="Methods", position=0)
    for method in outer:
        cfg = build_cfg(base, args, method)
        method_rows = []

        inner = tqdm(seeds, desc=f"{method} seeds", position=1, leave=False)
        for seed in inner:
            run_name = f"{cfg['dataset']}_{cfg['method']}"
            if cfg.get("run_tag"):
                run_name = f"{run_name}_{cfg['run_tag']}"
            run_name = f"{run_name}_seed{seed}"
            metrics_csv = Path(cfg["log_dir"]) / f"{run_name}_metrics.csv"
            if args.skip_existing and metrics_csv.exists():
                continue
            result = run(cfg, seed=seed, device=args.device)
            row = {
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "seed": seed,
                **result["metrics"],
            }
            all_rows.append(row)
            method_rows.append(row)

        # aggregate per method
        if method_rows:
            avg_metrics = {}
            for k in method_rows[0].keys():
                if k in ("dataset", "method", "seed"):
                    continue
                avg_metrics[k] = float(np.mean([r[k] for r in method_rows]))
            aggregated["split_mini_imagenet"][method] = avg_metrics

    # Save master CSV
    df = pd.DataFrame(all_rows)
    master_path = results_root / "master_results.csv"
    df.to_csv(master_path, index=False)

    # Plot summary charts
    if args.enable_plots and aggregated["split_mini_imagenet"]:
        plot_all_metric_bars(aggregated["split_mini_imagenet"], "split_mini_imagenet", fig_dir=str(results_root / "figures"))
        plot_pareto_frontier(aggregated["split_mini_imagenet"], "split_mini_imagenet", fig_dir=str(results_root / "figures"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content/data/mini-imagenet")
    parser.add_argument("--results-root", default="/content/drive/MyDrive/colab_results/mini_imagenet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="slim_resnet18", choices=["slim_resnet18", "vit_small"])
    parser.add_argument("--run-tag", default="resnet18")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated methods; default=all baselines+hybrids")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds; default=42,123,456,789,1024")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override input image size (e.g., 224 for ViT).")

    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")

    parser.add_argument("--enable-plots", dest="enable_plots", action="store_true", default=True)
    parser.add_argument("--disable-plots", dest="enable_plots", action="store_false")

    parser.add_argument("--enable-checkpoints", dest="enable_checkpoints", action="store_true", default=True)
    parser.add_argument("--disable-checkpoints", dest="enable_checkpoints", action="store_false")

    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")

    args = parser.parse_args()

    # Defaults for ViT
    if args.model == "vit_small":
        if args.run_tag == "resnet18":
            args.run_tag = "vit_small"
        if args.image_size is None:
            args.image_size = 224
        if args.batch_size == 32:
            args.batch_size = 16
        args.pretrained = True

    run_all(args)
