"""Run planned stop-and-resume robustness checks for ablation configs.

This helper intentionally stops each configured run after a task boundary,
then resumes the same run from the saved checkpoint. The final outputs land
in the normal ablation result tree so they can be audited alongside other
Phase 6 artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import load_config, run
from src.utils import epoch_label, resolve_results_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stop-and-resume robustness checks.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Folder of robustness configs (defaults to experiments/configs/ablations/robustness).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of seeds to run (overrides config seeds).",
    )
    parser.add_argument(
        "--stop-after-task",
        type=int,
        default=None,
        help="Override the planned stop task for every robustness config.",
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        action="store_true",
        help="Delete checkpoints after the resumed run completes successfully.",
    )
    args = parser.parse_args()

    config_dir = (
        Path(args.config_dir)
        if args.config_dir
        else Path(__file__).parent / "configs" / "ablations" / "robustness"
    )
    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No robustness YAML files found in {config_dir}")

    seed_override = None
    if args.seeds:
        seed_override = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows: List[dict] = []
    results_root = None
    epoch_dir = None

    for yaml_path in yaml_files:
        cfg = load_config(str(yaml_path))
        seeds = seed_override if seed_override is not None else cfg.get("seeds", [42])
        stop_after_task = (
            args.stop_after_task
            if args.stop_after_task is not None
            else cfg.get("robustness_stop_task")
        )
        if stop_after_task is None:
            raise ValueError(
                f"robustness_stop_task missing for {yaml_path.name}. "
                "Set it in the config or pass --stop-after-task."
            )

        if results_root is None:
            results_root = resolve_results_root(cfg.get("results_root"))
        if epoch_dir is None:
            epoch_dir = epoch_label(cfg.get("n_epochs", 1))

        for seed in seeds:
            partial_cfg = dict(cfg)
            partial_cfg["resume"] = False
            partial_cfg["stop_after_task"] = stop_after_task
            partial_cfg["cleanup_checkpoints_on_success"] = False
            partial = run(partial_cfg, seed=seed, device=args.device)
            if not partial.get("stopped_early"):
                raise RuntimeError(
                    f"Expected planned early stop for {yaml_path.name} seed {seed}, "
                    "but the partial run completed instead."
                )

            resumed_cfg = dict(cfg)
            resumed_cfg["resume"] = True
            if args.cleanup_checkpoints:
                resumed_cfg["cleanup_checkpoints_on_success"] = True
            final = run(resumed_cfg, seed=seed, device=args.device)
            rows.append(
                {
                    "dataset": cfg["dataset"],
                    "method": cfg["method"],
                    "seed": seed,
                    "run_tag": cfg.get("run_tag"),
                    "ablation_family": cfg.get("ablation_family"),
                    "stop_after_task": stop_after_task,
                    **final.get("metrics", {}),
                }
            )

    if rows:
        output_dir = (results_root or resolve_results_root(None)) / "analysis" / "ablations" / (epoch_dir or "epoch_1")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "robustness_launch_results.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
