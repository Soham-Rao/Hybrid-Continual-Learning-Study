"""Run all configured experiments for all seeds and aggregate results.

Usage::

    python experiments/run_all.py --device cuda
    python experiments/run_all.py --device cpu   # debug mode

Runs every .yaml in experiments/configs/ (excluding base_config.yaml)
across all seeds defined in the config, then generates a combined
multi-method Pareto chart and metric comparison bar charts.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import load_config, run
from src.visualization.plots import plot_all_metric_bars, plot_pareto_frontier, plot_compute_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--dataset", default=None,
                        help="Filter configs to this dataset name only.")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated list of method names to include (e.g. fine_tune,ewc).",
    )
    parser.add_argument(
        "--exclude-methods",
        default=None,
        help="Comma-separated list of method names to skip.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of seeds to run (overrides config).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs if results CSV already exists in log_dir.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Override results root (writes master_results.csv here).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Override per-run log/metrics CSV directory.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Override figure output directory.",
    )
    parser.add_argument(
        "--enable-plots",
        action="store_true",
        help="Force-enable plotting regardless of config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each run from the latest completed task checkpoint if present.",
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        action="store_true",
        help="Delete a run's checkpoints after it completes successfully.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Override config directory (defaults to experiments/configs).",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else (Path(__file__).parent / "configs")
    yaml_files = sorted(
        [f for f in config_dir.glob("*.yaml") if f.name != "base_config.yaml"]
    )

    dataset_filter = args.dataset

    include_methods = None
    if args.methods:
        include_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    exclude_methods = set()
    if args.exclude_methods:
        exclude_methods = {m.strip() for m in args.exclude_methods.split(",") if m.strip()}

    # {dataset → {method → {metric → mean_value}}}
    aggregated: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    task_times:  Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    all_rows:    List[dict] = []
    disable_plots = False

    seed_override = None
    results_root = None
    figure_root = None
    if args.seeds:
        seed_override = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for yaml_path in yaml_files:
        cfg = load_config(str(yaml_path))
        if args.results_root:
            cfg["results_root"] = args.results_root
        if args.log_dir:
            cfg["log_dir"] = args.log_dir
        if args.figure_dir:
            cfg["figure_dir"] = args.figure_dir
        if args.enable_plots:
            cfg["disable_plots"] = False
        if args.resume:
            cfg["resume"] = True
        if args.cleanup_checkpoints:
            cfg["cleanup_checkpoints_on_success"] = True
        disable_plots = disable_plots or cfg.get("disable_plots", False)
        if results_root is None:
            results_root = cfg.get("results_root", "results")
        if figure_root is None:
            figure_root = cfg.get("figure_dir", "results/figures")
        if dataset_filter and cfg.get("dataset") != dataset_filter:
            continue
        method_name = cfg.get("method")
        if include_methods and method_name not in include_methods:
            continue
        if method_name in exclude_methods:
            continue
        seeds = seed_override if seed_override is not None else cfg.get("seeds", [42])
        method_results = []

        run_tag = cfg.get("run_tag")
        for seed in seeds:
            run_name = f"{cfg['dataset']}_{cfg['method']}"
            if run_tag:
                run_name = f"{run_name}_{run_tag}"
            run_name = f"{run_name}_seed{seed}"
            log_dir  = cfg.get("log_dir", "results/raw")
            metrics_csv = Path(log_dir) / f"{run_name}_metrics.csv"
            if args.skip_existing and metrics_csv.exists():
                print(f"Skipping existing run: {run_name}")
                continue

            result = run(cfg, seed=seed, device=args.device)
            result["seed"] = seed
            method_results.append(result)

        if method_results:
            # Average metrics across seeds.
            metric_keys = list(method_results[0]["metrics"].keys())
            avg_metrics = {
                k: float(np.mean([r["metrics"][k] for r in method_results]))
                for k in metric_keys
            }
            avg_metrics["buffer_size"] = cfg.get("buffer_size", 0)

            ds  = cfg["dataset"]
            met = cfg["method"]
            aggregated[ds][met] = avg_metrics

            for row in method_results:
                all_rows.append({
                    "dataset": ds, "method": met,
                    "seed": row["seed"], **row["metrics"],
                })

    # Save master CSV.
    df = pd.DataFrame(all_rows)
    master_path = Path(results_root or "results") / "master_results.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(master_path, index=False)
    print(f"\nSaved -> {master_path}")

    # Generate multi-method comparison plots per dataset.
    if not disable_plots:
        for ds, methods_dict in aggregated.items():
            plot_all_metric_bars(methods_dict, ds, fig_dir=figure_root or "results/figures")
            plot_pareto_frontier(methods_dict, ds, fig_dir=figure_root or "results/figures")
            print(f"Plots saved for dataset: {ds}")
    else:
        print("Plots skipped (disable_plots=true).")


if __name__ == "__main__":
    main()
