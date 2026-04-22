"""Rebuild ablation analysis artifacts from completed ablation runs."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import load_config
from src.utils import epoch_label, metrics_csv_path, prepare_run_config, resolve_results_root


METRIC_FIELDS = [
    "avg_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
]

RESOURCE_NOTE = (
    "Memory sensitivity is proxy-only in v2: buffer/backbone/batch settings are "
    "summarized, but peak VRAM is not instrumented yet."
)


def parse_log(log_path: Path) -> Tuple[Dict[str, float] | None, int]:
    metrics_line = None
    starts = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if "Starting:" in line:
                starts += 1
            if "Metrics:" in line:
                metrics_line = line
    if metrics_line is None:
        return None, starts
    payload = metrics_line.split("Metrics:", 1)[1].strip()
    return ast.literal_eval(payload), starts


def _empty_runtime() -> Dict[str, float]:
    return {
        "total_time_sec": 0.0,
        "mean_task_time_sec": 0.0,
        "max_task_time_sec": 0.0,
        "n_completed_tasks": 0,
    }


def parse_runtime(metrics_path: Path, log_path: Path) -> Dict[str, float]:
    task_times: List[float] = []

    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("epoch") != "END":
                    continue
                value = row.get("time_sec")
                if value in (None, ""):
                    continue
                task_times.append(float(value))

    # v2 run logs include task timing even when the metrics CSV does not.
    if not task_times and log_path.exists():
        time_pattern = re.compile(r"epoch=END\s+time_sec=([0-9]+(?:\.[0-9]+)?)")
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                match = time_pattern.search(line)
                if match:
                    task_times.append(float(match.group(1)))

    if not task_times:
        return _empty_runtime()

    return {
        "total_time_sec": round(sum(task_times), 4),
        "mean_task_time_sec": round(sum(task_times) / len(task_times), 4),
        "max_task_time_sec": round(max(task_times), 4),
        "n_completed_tasks": len(task_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ablation analysis artifacts.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch count to rebuild (default 1).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "experiments" / "configs" / "ablations"
    epoch_dir = epoch_label(args.epoch)

    configs = sorted(config_root.rglob("*.yaml"))
    seen_run_keys: set[tuple] = set()

    rows: List[Dict] = []
    audit_rows: List[Dict] = []
    runtime_rows: List[Dict] = []
    memory_rows: List[Dict] = []
    robustness_rows: List[Dict] = []

    primary_path = resolve_results_root(None) / "analysis" / epoch_dir / "current_results.csv"
    primary_df = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()
    primary_lookup = {
        (row["dataset"], row["method"], int(row["seed"])): row
        for row in primary_df.to_dict("records")
    }

    for config_path in configs:
        cfg = load_config(str(config_path))
        if not (bool(cfg.get("ablation_family")) or cfg.get("result_group") == "ablations"):
            continue
        if epoch_label(cfg.get("n_epochs", 1)) != epoch_dir:
            continue

        seeds = cfg.get("seeds", [42])
        for seed in seeds:
            run_key = (
                cfg["dataset"],
                cfg["method"],
                seed,
                cfg.get("run_tag", ""),
                cfg.get("ablation_family", ""),
                epoch_dir,
            )
            if run_key in seen_run_keys:
                continue
            seen_run_keys.add(run_key)

            run_cfg = prepare_run_config(cfg, seed)
            log_path = Path(run_cfg["log_dir"]) / f"{run_cfg['run_name']}.log"
            metrics_path = metrics_csv_path(cfg, seed)
            checkpoint_dir = Path(run_cfg["checkpoint_dir"])

            parsed_metrics, log_starts = parse_log(log_path) if log_path.exists() else (None, 0)
            runtime = parse_runtime(metrics_path, log_path)
            completed = parsed_metrics is not None

            audit_row = {
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "seed": seed,
                "run_tag": cfg.get("run_tag", ""),
                "ablation_family": cfg.get("ablation_family", ""),
                "config": config_path.name,
                "run_dir_exists": Path(run_cfg["run_root"]).exists(),
                "log_exists": log_path.exists(),
                "metrics_csv_exists": metrics_path.exists(),
                "checkpoint_dir_exists": checkpoint_dir.exists(),
                "checkpoint_files": len(list(checkpoint_dir.glob("*.pt"))) if checkpoint_dir.exists() else 0,
                "log_starts": log_starts,
                "completed": completed,
                "status": "completed" if completed else "missing",
                **runtime,
            }

            if parsed_metrics:
                for field in METRIC_FIELDS:
                    audit_row[field] = parsed_metrics[field]
                rows.append(
                    {
                        "dataset": cfg["dataset"],
                        "method": cfg["method"],
                        "seed": seed,
                        "run_tag": cfg.get("run_tag", ""),
                        "result_group": cfg.get("result_group", "ablations"),
                        "ablation_family": cfg.get("ablation_family", ""),
                        **{field: parsed_metrics[field] for field in METRIC_FIELDS},
                    }
                )
                runtime_rows.append(
                    {
                        "dataset": cfg["dataset"],
                        "method": cfg["method"],
                        "seed": seed,
                        "run_tag": cfg.get("run_tag", ""),
                        "ablation_family": cfg.get("ablation_family", ""),
                        "model": cfg.get("model", ""),
                        "batch_size": cfg.get("batch_size", ""),
                        "num_workers": cfg.get("num_workers", ""),
                        **runtime,
                    }
                )
            else:
                for field in METRIC_FIELDS:
                    audit_row[field] = ""

            memory_rows.append(
                {
                    "dataset": cfg["dataset"],
                    "method": cfg["method"],
                    "seed": seed,
                    "run_tag": cfg.get("run_tag", ""),
                    "ablation_family": cfg.get("ablation_family", ""),
                    "model": cfg.get("model", ""),
                    "dataset_image_size": cfg.get("dataset_image_size", ""),
                    "batch_size": cfg.get("batch_size", ""),
                    "num_workers": cfg.get("num_workers", ""),
                    "fp16": cfg.get("fp16", ""),
                    "buffer_size": cfg.get("buffer_size", ""),
                    "agem_mem_batch": cfg.get("agem_mem_batch", ""),
                    "fisher_samples": cfg.get("fisher_samples", ""),
                    "joint_replay_epochs": cfg.get("joint_replay_epochs", ""),
                    "n_tasks": cfg.get("n_tasks", ""),
                    "memory_note": RESOURCE_NOTE,
                }
            )

            if cfg.get("ablation_family") == "robustness":
                primary = primary_lookup.get((cfg["dataset"], cfg["method"], int(seed)))
                robustness_row = {
                    "dataset": cfg["dataset"],
                    "method": cfg["method"],
                    "seed": seed,
                    "run_tag": cfg.get("run_tag", ""),
                    "stop_after_task": cfg.get("robustness_stop_task", ""),
                    "log_starts": log_starts,
                    "completed": completed,
                    "restart_verified": log_starts >= 2,
                    **runtime,
                }
                if parsed_metrics:
                    for field in METRIC_FIELDS:
                        robustness_row[f"robust_{field}"] = parsed_metrics[field]
                if primary:
                    for field in METRIC_FIELDS:
                        robustness_row[f"primary_{field}"] = primary.get(field, "")
                        if parsed_metrics:
                            robustness_row[f"delta_{field}"] = round(
                                float(parsed_metrics[field]) - float(primary.get(field, 0.0)),
                                4,
                            )
                robustness_rows.append(robustness_row)

            audit_rows.append(audit_row)

    analysis_dir = resolve_results_root(None) / "analysis" / "ablations" / epoch_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        pd.DataFrame(rows).sort_values(
            ["dataset", "ablation_family", "method", "run_tag", "seed"]
        ).to_csv(analysis_dir / "current_results.csv", index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).sort_values(
            ["dataset", "ablation_family", "method", "run_tag", "seed"]
        ).to_csv(analysis_dir / "ablation_audit.csv", index=False)
    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        runtime_summary = (
            runtime_df.groupby(
                ["dataset", "ablation_family", "method", "run_tag", "model", "batch_size", "num_workers"],
                dropna=False,
            )
            .agg(
                seeds_completed=("seed", "count"),
                mean_total_time_sec=("total_time_sec", "mean"),
                std_total_time_sec=("total_time_sec", "std"),
                mean_task_time_sec=("mean_task_time_sec", "mean"),
                max_task_time_sec=("max_task_time_sec", "max"),
            )
            .reset_index()
        )
        runtime_summary = runtime_summary.fillna(0.0)
        runtime_summary.to_csv(analysis_dir / "runtime_sensitivity_summary.csv", index=False)
    if memory_rows:
        memory_df = pd.DataFrame(memory_rows)
        memory_summary = (
            memory_df.groupby(
                [
                    "dataset",
                    "ablation_family",
                    "method",
                    "run_tag",
                    "model",
                    "dataset_image_size",
                    "batch_size",
                    "num_workers",
                    "fp16",
                    "buffer_size",
                    "agem_mem_batch",
                    "fisher_samples",
                    "joint_replay_epochs",
                    "n_tasks",
                    "memory_note",
                ],
                dropna=False,
            )
            .agg(seeds_present=("seed", "count"))
            .reset_index()
        )
        memory_summary.to_csv(analysis_dir / "memory_sensitivity_summary.csv", index=False)
    if robustness_rows:
        pd.DataFrame(robustness_rows).sort_values(
            ["dataset", "method", "run_tag", "seed"]
        ).to_csv(analysis_dir / "robustness_summary.csv", index=False)

    note_path = analysis_dir / "resource_summary_notes.md"
    note_path.write_text(
        "# Resource Summary Notes\n\n"
        "- `runtime_sensitivity_summary.csv` is derived from per-task `time_sec` rows in run metrics CSVs.\n"
        "- `memory_sensitivity_summary.csv` is proxy-based. It summarizes memory-relevant config knobs such as backbone, "
        "batch size, and replay buffer size.\n"
        "- Peak VRAM and host RAM are not instrumented in the current v2 trainer, so those values are intentionally not claimed here.\n",
        encoding="utf-8",
    )

    print(f"Saved ablation analysis under -> {analysis_dir}")


if __name__ == "__main__":
    main()
