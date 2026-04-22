"""Rebuild epoch-1 aggregate analysis artifacts from completed run logs.

Outputs:
- results/analysis/epoch_1/current_results.csv
- results/analysis/epoch_1/epoch1_audit.csv
- results/analysis/epoch_1/epoch1_dataset_tables.md
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiments.run_experiment import load_config
from src.utils import epoch_label, metrics_csv_path, prepare_run_config, resolve_results_root


METHOD_ORDER = [
    "fine_tune",
    "joint_training",
    "ewc",
    "agem",
    "lwf",
    "der",
    "xder",
    "icarl",
    "er_ewc",
    "progress_compress",
    "agem_distill",
    "si_der",
]

METHOD_LABELS = {
    "fine_tune": "Fine-Tune",
    "joint_training": "Joint Training",
    "ewc": "EWC",
    "agem": "A-GEM",
    "lwf": "LwF",
    "der": "DER",
    "xder": "X-DER",
    "icarl": "iCaRL",
    "er_ewc": "ER+EWC",
    "progress_compress": "Progress & Compress",
    "agem_distill": "A-GEM+Distill",
    "si_der": "SI-DER",
}

DATASET_ORDER = [
    "permuted_mnist",
    "split_cifar10",
    "split_cifar100",
    "split_mini_imagenet",
]

DATASET_TITLES = {
    "permuted_mnist": "Permuted MNIST",
    "split_cifar10": "Split CIFAR-10",
    "split_cifar100": "Split CIFAR-100",
    "split_mini_imagenet": "Split Mini-ImageNet",
}

METRIC_FIELDS = [
    "avg_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
]


def parse_metrics_from_log(log_path: Path) -> Dict[str, float] | None:
    metrics_line = None
    starts = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if "Starting:" in line:
                starts += 1
            if "Metrics:" in line:
                metrics_line = line

    if metrics_line is None:
        return None

    payload = metrics_line.split("Metrics:", 1)[1].strip()
    metrics = ast.literal_eval(payload)
    metrics["log_starts"] = starts
    return metrics


def method_hparam_summary(cfg: Dict) -> str:
    method = cfg["method"]
    if method == "fine_tune":
        return "baseline only"
    if method == "joint_training":
        return f"joint_replay_epochs={cfg.get('joint_replay_epochs')}"
    if method == "ewc":
        return f"ewc_lambda={cfg.get('ewc_lambda')}, fisher_samples={cfg.get('fisher_samples')}"
    if method == "agem":
        return f"agem_mem_batch={cfg.get('agem_mem_batch')}"
    if method == "lwf":
        return f"lwf_lambda={cfg.get('lwf_lambda')}, lwf_temp={cfg.get('lwf_temp')}"
    if method == "der":
        return f"der_alpha={cfg.get('der_alpha')}"
    if method == "xder":
        return f"xder_beta={cfg.get('xder_beta')}"
    if method == "icarl":
        return (
            f"icarl_distill_w={cfg.get('icarl_distill_w')}, "
            f"icarl_temp={cfg.get('icarl_temp')}, use_nmc={cfg.get('use_nmc')}"
        )
    if method == "er_ewc":
        return f"ewc_lambda={cfg.get('ewc_lambda')}, er_replay_ratio={cfg.get('er_replay_ratio')}"
    if method == "progress_compress":
        return (
            f"pc_distill_w={cfg.get('pc_distill_w')}, pc_ewc_lambda={cfg.get('pc_ewc_lambda')}, "
            f"pc_compress_epochs={cfg.get('pc_compress_epochs')}"
        )
    if method == "agem_distill":
        return (
            f"agem_mem_batch={cfg.get('agem_mem_batch')}, distill_lambda={cfg.get('distill_lambda')}, "
            f"distill_temp={cfg.get('distill_temp')}"
        )
    if method == "si_der":
        return f"si_lambda={cfg.get('si_lambda')}, si_xi={cfg.get('si_xi')}, der_alpha={cfg.get('der_alpha')}"
    return ""


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "experiments" / "configs"

    configs = sorted(
        p for p in config_dir.glob("*.yaml")
        if p.name not in {"base_config.yaml", "smoke_test.yaml"}
    )

    rows: List[Dict] = []
    audit_rows: List[Dict] = []
    cfg_index: Dict[Tuple[str, str], Dict] = {}
    epoch_dir = epoch_label(1)

    for config_path in configs:
        cfg = load_config(str(config_path))
        dataset = cfg["dataset"]
        method = cfg["method"]
        cfg_index[(dataset, method)] = cfg
        results_root = resolve_results_root(cfg.get("results_root"))
        analysis_dir = results_root / "analysis" / epoch_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for seed in cfg.get("seeds", [42]):
            run_cfg = prepare_run_config(cfg, seed)
            log_path = Path(run_cfg["log_dir"]) / f"{run_cfg['run_name']}.log"
            checkpoint_dir = Path(run_cfg["checkpoint_dir"])
            metrics_path = metrics_csv_path(cfg, seed)

            run_dir_exists = Path(run_cfg["run_root"]).exists()
            log_exists = log_path.exists()
            metrics_exists = metrics_path.exists()
            checkpoint_dir_exists = checkpoint_dir.exists()
            checkpoint_files = (
                len(list(checkpoint_dir.glob("*.pt")))
                if checkpoint_dir_exists else 0
            )
            parsed = parse_metrics_from_log(log_path) if log_exists else None
            completed = parsed is not None

            audit_row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "config": config_path.name,
                "run_dir_exists": run_dir_exists,
                "log_exists": log_exists,
                "metrics_csv_exists": metrics_exists,
                "checkpoint_dir_exists": checkpoint_dir_exists,
                "checkpoint_files": checkpoint_files,
                "log_starts": parsed.get("log_starts", 0) if parsed else 0,
                "completed": completed,
                "status": "completed" if completed else "missing",
            }

            if parsed:
                for field in METRIC_FIELDS:
                    audit_row[field] = parsed[field]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        **{field: parsed[field] for field in METRIC_FIELDS},
                    }
                )
            else:
                for field in METRIC_FIELDS:
                    audit_row[field] = ""

            audit_rows.append(audit_row)

    if not rows:
        raise RuntimeError("No completed runs found while rebuilding epoch-1 analysis.")

    current_df = pd.DataFrame(rows).sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    analysis_dir = resolve_results_root(None) / "analysis" / epoch_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    current_path = analysis_dir / "current_results.csv"
    current_df.to_csv(current_path, index=False)

    audit_path = analysis_dir / "epoch1_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    by_dataset_method: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        by_dataset_method[(row["dataset"], row["method"])].append(row)

    lines = [
        "# Epoch-1 Dataset Summary Tables",
        "",
        "Metrics below are means across the completed seed runs listed in `current_results.csv`.",
        "",
    ]

    for dataset in DATASET_ORDER:
        lines.append(f"## {DATASET_TITLES[dataset]}")
        lines.append("")
        lines.append("| Method | Avg Accuracy | Forgetting | BWT | FWT | Backbone | Epochs | Batch | LR | FP16 | Buffer | Key Hyperparameters |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |")
        for method in METHOD_ORDER:
            values = by_dataset_method.get((dataset, method), [])
            cfg = cfg_index.get((dataset, method))
            if not values or cfg is None:
                continue

            def mean_metric(field: str) -> str:
                return f"{sum(float(v[field]) for v in values) / len(values):.4f}"

            lines.append(
                "| {method_label} | {aa} | {fg} | {bwt} | {fwt} | {model} | {epochs} | {batch} | {lr} | {fp16} | {buffer} | {key_hparams} |".format(
                    method_label=METHOD_LABELS.get(method, method),
                    aa=mean_metric("avg_accuracy"),
                    fg=mean_metric("forgetting"),
                    bwt=mean_metric("backward_transfer"),
                    fwt=mean_metric("forward_transfer"),
                    model=cfg.get("model", ""),
                    epochs=cfg.get("n_epochs", ""),
                    batch=cfg.get("batch_size", ""),
                    lr=cfg.get("lr", ""),
                    fp16=cfg.get("fp16", ""),
                    buffer=cfg.get("buffer_size", ""),
                    key_hparams=method_hparam_summary(cfg),
                )
            )
        lines.append("")

    tables_path = analysis_dir / "epoch1_dataset_tables.md"
    tables_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved -> {current_path}")
    print(f"Saved -> {audit_path}")
    print(f"Saved -> {tables_path}")


if __name__ == "__main__":
    main()
