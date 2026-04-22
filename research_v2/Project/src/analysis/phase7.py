"""Phase 7 analysis pipeline for the finalized v2 epoch-1 study."""

from __future__ import annotations

import ast
import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import yaml

from src.recommendation.engine import estimate_memory_mb, method_traits
from src.utils.paths import epoch_label, metrics_csv_path, prepare_run_config


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

PRIMARY_METRIC = "avg_accuracy"
SECONDARY_METRIC = "forgetting"
STAT_METRICS = [PRIMARY_METRIC, SECONDARY_METRIC]
ALL_METRICS = [
    "avg_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
]
ALPHA = 0.05

PAIRWISE_COLUMNS = [
    "dataset",
    "metric",
    "method_a",
    "method_b",
    "n_pairs",
    "n_nonzero_pairs",
    "mean_a",
    "mean_b",
    "mean_diff",
    "statistic",
    "p_value",
    "holm_adjusted_p_value",
    "reject_h0",
]

EFFECT_COLUMNS = [
    "dataset",
    "metric",
    "method_a",
    "method_b",
    "n_pairs",
    "rank_biserial",
    "magnitude",
]

TIME_PATTERN = re.compile(r"epoch=END\s+time_sec=([0-9]+(?:\.[0-9]+)?)")
METRICS_PATTERN = re.compile(r"Metrics:\s*(\{.*?\})")

sns.set_theme(style="whitegrid", font_scale=1.0)


def _phase7_analysis_dir(results_root: Path, epoch_dir: str) -> Path:
    return results_root / "analysis" / epoch_dir


def _phase7_figure_dir(results_root: Path, epoch_dir: str) -> Path:
    return results_root / "figures" / epoch_dir / "analysis"


def load_experiment_config(config_path: str) -> Dict[str, object]:
    """Merge base_config.yaml into one experiment config without importing the trainer stack."""
    config_file = Path(config_path)
    base_path = config_file.parent / "base_config.yaml"
    with base_path.open(encoding="utf-8") as fh:
        base = yaml.safe_load(fh).get("base_config", {})
    with config_file.open(encoding="utf-8") as fh:
        exp = yaml.safe_load(fh)
    return {**base, **exp}


def _parse_log_metrics(log_path: Path) -> Tuple[Optional[Dict[str, float]], int]:
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


def _parse_runtime_seconds(log_path: Path, metrics_path: Path) -> float:
    task_times: List[float] = []
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if "time_sec" in metrics_df.columns:
            end_rows = metrics_df[metrics_df["epoch"].astype(str) == "END"]
            values = pd.to_numeric(end_rows["time_sec"], errors="coerce").dropna().tolist()
            task_times.extend(float(value) for value in values)
    if not task_times and log_path.exists():
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                match = TIME_PATTERN.search(line)
                if match:
                    task_times.append(float(match.group(1)))
    return round(sum(task_times), 4)


def _caveat_note(log_starts: int) -> str:
    notes: List[str] = []
    if log_starts > 1:
        notes.append(
            "Run log contains restarts; final metrics were read from the last completed metrics block."
        )
    return " ".join(notes)


def _primary_configs(config_dir: Path, epoch_dir: str) -> Iterable[Path]:
    for config_path in sorted(config_dir.glob("*.yaml")):
        if config_path.name in {"base_config.yaml", "smoke_test.yaml"}:
            continue
        cfg = load_experiment_config(str(config_path))
        if cfg.get("run_tag") or cfg.get("ablation_family") or cfg.get("result_group") == "ablations":
            continue
        if epoch_label(cfg.get("n_epochs", 1)) != epoch_dir:
            continue
        yield config_path


def build_master_results(project_root: Path, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    """Collect one row per finalized primary run seed."""
    config_dir = project_root / "experiments" / "configs"
    rows: List[Dict[str, object]] = []

    for config_path in _primary_configs(config_dir, epoch_dir):
        cfg = load_experiment_config(str(config_path))
        for seed in cfg.get("seeds", [42]):
            run_cfg = prepare_run_config(cfg, seed)
            log_path = Path(run_cfg["log_dir"]) / f"{run_cfg['run_name']}.log"
            metrics_path = metrics_csv_path(cfg, seed)
            parsed_metrics, log_starts = parse_log_metrics_with_starts(log_path)
            total_time_sec = _parse_runtime_seconds(log_path, metrics_path)
            completed = parsed_metrics is not None
            row: Dict[str, object] = {
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "seed": int(seed),
                "run_name": run_cfg["run_name"],
                "config_name": config_path.name,
                "model": cfg.get("model", ""),
                "n_epochs": int(cfg.get("n_epochs", 1)),
                "batch_size": int(cfg.get("batch_size", 0) or 0),
                "num_workers": int(cfg.get("num_workers", 0) or 0),
                "fp16": bool(cfg.get("fp16", False)),
                "buffer_size": int(cfg.get("buffer_size", 0) or 0),
                "total_time_sec": total_time_sec,
                "log_starts": int(log_starts),
                "completed": bool(completed),
                "caveat_note": _caveat_note(log_starts),
            }
            if parsed_metrics:
                for field in ALL_METRICS:
                    row[field] = float(parsed_metrics[field])
            else:
                for field in ALL_METRICS:
                    row[field] = float("nan")
            rows.append(row)

    master = pd.DataFrame(rows)
    if master.empty:
        raise RuntimeError("No completed primary epoch-1 runs were found for Phase 7.")

    master = master.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    master_path = _phase7_analysis_dir(results_root, epoch_dir) / "master_results.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(master_path, index=False)
    return master


def parse_log_metrics_with_starts(log_path: Path) -> Tuple[Optional[Dict[str, float]], int]:
    if not log_path.exists():
        return None, 0
    return _parse_log_metrics(log_path)


def _aggregate_caveat(notes: Sequence[str]) -> str:
    unique = [note for note in dict.fromkeys(note for note in notes if note)]
    return " | ".join(unique)


def build_summary_tables(master_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_dir = _phase7_analysis_dir(results_root, epoch_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        master_df.groupby(
            [
                "dataset",
                "method",
                "model",
                "n_epochs",
                "batch_size",
                "num_workers",
                "fp16",
                "buffer_size",
            ],
            dropna=False,
        )
        .agg(
            seeds=("seed", "count"),
            total_time_sec_mean=("total_time_sec", "mean"),
            total_time_sec_std=("total_time_sec", "std"),
            caveat_note=("caveat_note", _aggregate_caveat),
            avg_accuracy_mean=("avg_accuracy", "mean"),
            avg_accuracy_std=("avg_accuracy", "std"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            backward_transfer_mean=("backward_transfer", "mean"),
            backward_transfer_std=("backward_transfer", "std"),
            forward_transfer_mean=("forward_transfer", "mean"),
            forward_transfer_std=("forward_transfer", "std"),
        )
        .reset_index()
    )

    summary["runtime_hours_mean"] = summary["total_time_sec_mean"] / 3600.0
    summary["runtime_hours_std"] = summary["total_time_sec_std"].fillna(0.0) / 3600.0
    summary["estimated_memory_mb"] = summary.apply(
        lambda row: estimate_memory_mb(str(row["dataset"]), str(row["method"]), float(row["buffer_size"])),
        axis=1,
    )
    summary["method_family"] = summary["method"].map(lambda item: str(method_traits(item).get("family", "other")))
    summary["dataset"] = pd.Categorical(summary["dataset"], categories=DATASET_ORDER, ordered=True)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["dataset", "method"]).reset_index(drop=True)

    summary_metrics = pd.DataFrame(
        [
            {
                "dataset": row.dataset,
                "method": row.method,
                "metric": metric,
                "mean": getattr(row, f"{metric}_mean"),
                "std": 0.0 if pd.isna(getattr(row, f"{metric}_std")) else getattr(row, f"{metric}_std"),
            }
            for row in summary.itertuples(index=False)
            for metric in ALL_METRICS
        ]
    )

    pretty = summary[
        [
            "dataset",
            "method",
            "method_family",
            "model",
            "seeds",
            "batch_size",
            "num_workers",
            "fp16",
            "buffer_size",
            "estimated_memory_mb",
            "runtime_hours_mean",
            "runtime_hours_std",
            "avg_accuracy_mean",
            "avg_accuracy_std",
            "forgetting_mean",
            "forgetting_std",
            "backward_transfer_mean",
            "backward_transfer_std",
            "forward_transfer_mean",
            "forward_transfer_std",
            "caveat_note",
        ]
    ].copy()

    for metric in ALL_METRICS:
        pretty[f"{metric}_mean_std"] = pretty.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} +/- {(0.0 if pd.isna(row[f'{metric}_std']) else row[f'{metric}_std']):.4f}",
            axis=1,
        )
    pretty["runtime_hours_mean_std"] = pretty.apply(
        lambda row: f"{row['runtime_hours_mean']:.4f} +/- {row['runtime_hours_std']:.4f}",
        axis=1,
    )

    summary.to_csv(analysis_dir / "paper_ready_summary.csv", index=False)
    pretty.to_csv(analysis_dir / "paper_ready_summary_pretty.csv", index=False)
    summary_metrics.to_csv(analysis_dir / "summary_metrics.csv", index=False)
    return summary, pretty, summary_metrics


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm-adjusted p-values in the original order."""
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted_sorted: List[float] = []
    running_max = 0.0
    for rank, (_, p_value) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * float(p_value))
        running_max = max(running_max, adjusted)
        adjusted_sorted.append(running_max)
    out = [1.0] * m
    for adjusted, (original_idx, _) in zip(adjusted_sorted, indexed):
        out[original_idx] = adjusted
    return out


def safe_wilcoxon(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[float, float, int]:
    """Run a conservative paired Wilcoxon test with stable fallback behavior."""
    diffs = [float(a) - float(b) for a, b in zip(sample_a, sample_b)]
    nonzero = [diff for diff in diffs if not math.isclose(diff, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 0.0, 1.0, 0
    try:
        result = wilcoxon(sample_a, sample_b, zero_method="wilcox", alternative="two-sided")
    except TypeError:
        result = wilcoxon(sample_a, sample_b, zero_method="wilcox")
    except ValueError:
        return 0.0, 1.0, len(nonzero)
    return float(result.statistic), float(result.pvalue), len(nonzero)


def rank_biserial_effect(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    diffs = pd.Series([float(a) - float(b) for a, b in zip(sample_a, sample_b)])
    diffs = diffs[~diffs.map(lambda value: math.isclose(value, 0.0, abs_tol=1e-12))]
    if diffs.empty:
        return 0.0
    ranks = rankdata(diffs.abs().to_numpy(), method="average")
    positive = float(sum(rank for rank, diff in zip(ranks, diffs) if diff > 0))
    negative = float(sum(rank for rank, diff in zip(ranks, diffs) if diff < 0))
    denom = positive + negative
    if math.isclose(denom, 0.0, abs_tol=1e-12):
        return 0.0
    return round((positive - negative) / denom, 6)


def effect_size_magnitude(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "medium"
    return "large"


def build_pairwise_tests(master_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_df = master_df[master_df["dataset"] == dataset].copy()
        if dataset_df.empty:
            continue
        for metric in STAT_METRICS:
            metric_rows: List[Dict[str, object]] = []
            for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
                subset_a = dataset_df[dataset_df["method"] == method_a][["seed", metric]]
                subset_b = dataset_df[dataset_df["method"] == method_b][["seed", metric]]
                merged = subset_a.merge(subset_b, on="seed", suffixes=("_a", "_b")).sort_values("seed")
                if merged.empty:
                    continue
                sample_a = merged[f"{metric}_a"].tolist()
                sample_b = merged[f"{metric}_b"].tolist()
                statistic, p_value, n_nonzero = safe_wilcoxon(sample_a, sample_b)
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": int(len(merged)),
                        "n_nonzero_pairs": int(n_nonzero),
                        "mean_a": float(pd.Series(sample_a).mean()),
                        "mean_b": float(pd.Series(sample_b).mean()),
                        "mean_diff": float(pd.Series(sample_a).mean() - pd.Series(sample_b).mean()),
                        "statistic": statistic,
                        "p_value": p_value,
                    }
                )
            adjusted = holm_adjust([row["p_value"] for row in metric_rows])
            for row, adjusted_p in zip(metric_rows, adjusted):
                row["holm_adjusted_p_value"] = adjusted_p
                row["reject_h0"] = bool(adjusted_p < ALPHA)
                rows.append(row)
    pairwise = pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
    pairwise.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "pairwise_tests.csv", index=False)
    return pairwise


def build_effect_sizes(master_df: pd.DataFrame, pairwise_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if pairwise_df.empty:
        effect_sizes = pd.DataFrame(columns=EFFECT_COLUMNS)
        effect_sizes.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "effect_sizes.csv", index=False)
        return effect_sizes

    for row in pairwise_df.itertuples(index=False):
        dataset_df = master_df[master_df["dataset"] == row.dataset]
        subset_a = dataset_df[dataset_df["method"] == row.method_a][["seed", row.metric]]
        subset_b = dataset_df[dataset_df["method"] == row.method_b][["seed", row.metric]]
        merged = subset_a.merge(subset_b, on="seed", suffixes=("_a", "_b")).sort_values("seed")
        sample_a = merged[f"{row.metric}_a"].tolist()
        sample_b = merged[f"{row.metric}_b"].tolist()
        effect = rank_biserial_effect(sample_a, sample_b)
        rows.append(
            {
                "dataset": row.dataset,
                "metric": row.metric,
                "method_a": row.method_a,
                "method_b": row.method_b,
                "n_pairs": int(len(merged)),
                "rank_biserial": effect,
                "magnitude": effect_size_magnitude(effect),
            }
        )
    effect_sizes = pd.DataFrame(rows, columns=EFFECT_COLUMNS)
    effect_sizes.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "effect_sizes.csv", index=False)
    return effect_sizes


def build_dataset_leaders(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_summary = summary_df[summary_df["dataset"].astype(str) == dataset].sort_values(
            "avg_accuracy_mean",
            ascending=False,
        )
        if dataset_summary.empty:
            continue
        leader = dataset_summary.iloc[0]
        top_cluster = [str(leader["method"])]
        dataset_pairwise = pairwise_df[
            (pairwise_df["dataset"] == dataset) & (pairwise_df["metric"] == PRIMARY_METRIC)
        ]
        for contender in dataset_summary["method"].astype(str).tolist()[1:]:
            mask = (
                ((dataset_pairwise["method_a"] == leader["method"]) & (dataset_pairwise["method_b"] == contender))
                | ((dataset_pairwise["method_a"] == contender) & (dataset_pairwise["method_b"] == leader["method"]))
            )
            comparison = dataset_pairwise[mask]
            if comparison.empty:
                top_cluster.append(contender)
                continue
            comp = comparison.iloc[0]
            contender_mean = float(
                dataset_summary.loc[dataset_summary["method"].astype(str) == contender, "avg_accuracy_mean"].iloc[0]
            )
            significantly_worse = bool(
                float(comp["holm_adjusted_p_value"]) < ALPHA and float(leader["avg_accuracy_mean"]) > contender_mean
            )
            if not significantly_worse:
                top_cluster.append(contender)
        rows.append(
            {
                "dataset": dataset,
                "best_method": leader["method"],
                "best_avg_accuracy_mean": leader["avg_accuracy_mean"],
                "best_forgetting_mean": leader["forgetting_mean"],
                "best_runtime_hours_mean": leader["runtime_hours_mean"],
                "best_estimated_memory_mb": leader["estimated_memory_mb"],
                "top_cluster_methods": "|".join(top_cluster),
                "top_cluster_size": len(top_cluster),
                "caveat_note": leader["caveat_note"],
            }
        )
    leaders = pd.DataFrame(rows)
    leaders.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "dataset_leaders.csv", index=False)
    return leaders


def build_friedman_table(summary_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    rank_rows: List[Dict[str, object]] = []
    per_dataset = []
    for dataset in DATASET_ORDER:
        dataset_summary = summary_df[summary_df["dataset"].astype(str) == dataset].copy()
        if dataset_summary.empty:
            continue
        dataset_summary["rank"] = dataset_summary["avg_accuracy_mean"].rank(
            ascending=False,
            method="average",
        )
        per_dataset.append(dataset_summary[["method", "rank"]].set_index("method")["rank"])
    if len(per_dataset) < 2:
        friedman = pd.DataFrame(columns=["method", "average_rank", "dataset_count", "friedman_statistic", "friedman_p_value"])
        friedman.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "friedman_avg_accuracy.csv", index=False)
        return friedman

    rank_matrix = pd.concat(per_dataset, axis=1).reindex(METHOD_ORDER).dropna()
    statistic, p_value = friedmanchisquare(*[rank_matrix.loc[method].tolist() for method in rank_matrix.index])
    average_ranks = rank_matrix.mean(axis=1)
    for method, average_rank in average_ranks.sort_values().items():
        rank_rows.append(
            {
                "method": method,
                "average_rank": float(average_rank),
                "dataset_count": int(rank_matrix.shape[1]),
                "friedman_statistic": float(statistic),
                "friedman_p_value": float(p_value),
            }
        )
    friedman = pd.DataFrame(rank_rows)
    friedman.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "friedman_avg_accuracy.csv", index=False)
    return friedman


def _is_non_dominated(target: Tuple[float, float, float], pool: Iterable[Tuple[float, float, float]]) -> bool:
    acc, runtime, memory = target
    for other_acc, other_runtime, other_memory in pool:
        if other_acc >= acc and other_runtime <= runtime and other_memory <= memory:
            if other_acc > acc or other_runtime < runtime or other_memory < memory:
                return False
    return True


def build_pareto_candidates(summary_df: pd.DataFrame, leaders_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> pd.DataFrame:
    leader_lookup = {row.dataset: row.best_method for row in leaders_df.itertuples(index=False)}
    cluster_lookup = {row.dataset: row.top_cluster_methods.split("|") for row in leaders_df.itertuples(index=False)}
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_summary = summary_df[
            (summary_df["dataset"].astype(str) == dataset)
            & (summary_df["method"].astype(str) != "joint_training")
        ].copy()
        if dataset_summary.empty:
            continue
        points = [
            (float(row.avg_accuracy_mean), float(row.runtime_hours_mean), float(row.estimated_memory_mb), row)
            for row in dataset_summary.itertuples(index=False)
        ]
        plain_pool = [(acc, runtime, memory) for acc, runtime, memory, _ in points]
        for acc, runtime, memory, row in points:
            if not _is_non_dominated((acc, runtime, memory), plain_pool):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": row.method,
                    "avg_accuracy_mean": acc,
                    "forgetting_mean": float(row.forgetting_mean),
                    "runtime_hours_mean": runtime,
                    "estimated_memory_mb": memory,
                    "leader_flag": bool(leader_lookup.get(dataset) == row.method),
                    "top_cluster_flag": bool(row.method in cluster_lookup.get(dataset, [])),
                }
            )
    pareto = pd.DataFrame(rows).sort_values(["dataset", "avg_accuracy_mean"], ascending=[True, False])
    pareto.to_csv(_phase7_analysis_dir(results_root, epoch_dir) / "pareto_frontier_candidates.csv", index=False)
    return pareto


def _save_heatmap(summary_df: pd.DataFrame, metric: str, figure_dir: Path) -> None:
    pivot = summary_df.pivot(index="dataset", columns="method", values=f"{metric}_mean")
    pivot = pivot.reindex(DATASET_ORDER)
    pivot = pivot.reindex(columns=METHOD_ORDER)
    if pivot.dropna(how="all").empty:
        return
    plt.figure(figsize=(12, 4.8))
    cmap = "RdYlGn" if metric == "avg_accuracy" else "RdYlGn_r"
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap)
    plt.title(f"Epoch-1 {metric.replace('_', ' ').title()} Heatmap")
    plt.xlabel("Method")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(figure_dir / f"phase7_{metric}_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()


def _save_tradeoff_plot(
    dataset_summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if dataset_summary.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=dataset_summary,
        x=x_col,
        y=y_col,
        hue="method",
        s=110,
    )
    for row in dataset_summary.itertuples(index=False):
        plt.text(getattr(row, x_col), getattr(row, y_col), str(row.method), fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_phase7_figures(summary_df: pd.DataFrame, results_root: Path, epoch_dir: str) -> None:
    figure_dir = _phase7_figure_dir(results_root, epoch_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_heatmap(summary_df, "avg_accuracy", figure_dir)
    _save_heatmap(summary_df, "forgetting", figure_dir)
    for dataset in DATASET_ORDER:
        dataset_summary = summary_df[summary_df["dataset"].astype(str) == dataset].copy()
        if dataset_summary.empty:
            continue
        title = DATASET_TITLES.get(dataset, dataset)
        _save_tradeoff_plot(
            dataset_summary,
            "forgetting_mean",
            "avg_accuracy_mean",
            "Forgetting (%)",
            "Average Accuracy (%)",
            f"{title}: Accuracy vs Forgetting",
            figure_dir / f"{dataset}_accuracy_vs_forgetting.png",
        )
        _save_tradeoff_plot(
            dataset_summary,
            "runtime_hours_mean",
            "avg_accuracy_mean",
            "Runtime (hours)",
            "Average Accuracy (%)",
            f"{title}: Accuracy vs Runtime",
            figure_dir / f"{dataset}_accuracy_vs_runtime.png",
        )
        _save_tradeoff_plot(
            dataset_summary,
            "estimated_memory_mb",
            "avg_accuracy_mean",
            "Estimated Memory (MB, proxy)",
            "Average Accuracy (%)",
            f"{title}: Accuracy vs Estimated Memory",
            figure_dir / f"{dataset}_accuracy_vs_memory.png",
        )


def write_phase7_report(
    summary_df: pd.DataFrame,
    leaders_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    ablation_analysis_dir: Path,
    results_root: Path,
    epoch_dir: str,
) -> Path:
    lines = [
        "# Phase 7 Analysis Report",
        "",
        "This report is generated from the finalized v2 epoch-1 primary matrix only.",
        "Primary inferential claims use matched-seed per-dataset tests on average accuracy and forgetting.",
        "",
        "## Dataset Leaders",
        "",
    ]
    for row in leaders_df.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: leader `{row.best_method}` with AA {row.best_avg_accuracy_mean:.4f}, "
            f"forgetting {row.best_forgetting_mean:.4f}, runtime {row.best_runtime_hours_mean:.4f} h, "
            f"memory proxy {row.best_estimated_memory_mb:.2f} MB. "
            f"Top cluster: {row.top_cluster_methods}."
        )
    lines.extend(
        [
            "",
            "## Statistical Caveats",
            "",
            "- `avg_accuracy` is the primary claim metric; `forgetting` is the main secondary claim metric.",
            "- Wilcoxon signed-rank tests are matched by seed within each dataset and corrected with Holm at alpha 0.05.",
            "- `backward_transfer`, `forward_transfer`, runtime, and memory are descriptive only in Phase 7.",
            "- The cross-dataset Friedman result is secondary because it is based on only four datasets.",
            "",
        ]
    )

    if not friedman_df.empty:
        first = friedman_df.iloc[0]
        lines.append(
            f"- Friedman average-rank summary: statistic={first['friedman_statistic']:.4f}, "
            f"p={first['friedman_p_value']:.6f} across {int(first['dataset_count'])} datasets."
        )
        lines.append("")

    lines.extend(["## Trade-Off Highlights", ""])
    for dataset in DATASET_ORDER:
        subset = pareto_df[pareto_df["dataset"] == dataset]
        if subset.empty:
            continue
        methods = ", ".join(subset["method"].astype(str).tolist())
        lines.append(f"- `{dataset}` non-joint Pareto candidates: {methods}.")
    lines.append("")

    lines.extend(["## Phase 6 Context", ""])
    if ablation_analysis_dir.exists():
        ablation_current = ablation_analysis_dir / "current_results.csv"
        robustness_summary = ablation_analysis_dir / "robustness_summary.csv"
        if ablation_current.exists():
            ablation_df = pd.read_csv(ablation_current)
            counts = ablation_df.groupby("ablation_family").size().to_dict()
            lines.append(
                "- Completed ablation families: "
                + ", ".join(f"{family}={count}" for family, count in sorted(counts.items()))
                + "."
            )
        if robustness_summary.exists():
            robustness_df = pd.read_csv(robustness_summary)
            lines.append(
                f"- Resume/restart robustness checks completed: {len(robustness_df)} representative runs."
            )
    else:
        lines.append("- No Phase 6 ablation context was found under the v2 ablation analysis directory.")
    lines.append("")

    report_path = _phase7_analysis_dir(results_root, epoch_dir) / "phase7_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_phase7_pipeline(project_root: Path, results_root: Path, epoch: int = 1) -> Dict[str, pd.DataFrame]:
    epoch_dir = epoch_label(epoch)
    analysis_dir = _phase7_analysis_dir(results_root, epoch_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    master = build_master_results(project_root, results_root, epoch_dir)
    summary, pretty, summary_metrics = build_summary_tables(master, results_root, epoch_dir)
    pairwise = build_pairwise_tests(master, results_root, epoch_dir)
    effect_sizes = build_effect_sizes(master, pairwise, results_root, epoch_dir)
    leaders = build_dataset_leaders(summary, pairwise, results_root, epoch_dir)
    friedman = build_friedman_table(summary, results_root, epoch_dir)
    pareto = build_pareto_candidates(summary, leaders, results_root, epoch_dir)
    generate_phase7_figures(summary, results_root, epoch_dir)
    report_path = write_phase7_report(
        summary,
        leaders,
        pairwise,
        friedman,
        pareto,
        results_root / "analysis" / "ablations" / epoch_dir,
        results_root,
        epoch_dir,
    )

    return {
        "master": master,
        "summary": summary,
        "pretty": pretty,
        "summary_metrics": summary_metrics,
        "pairwise": pairwise,
        "effect_sizes": effect_sizes,
        "leaders": leaders,
        "friedman": friedman,
        "pareto": pareto,
        "report": pd.DataFrame([{"path": str(report_path)}]),
    }
