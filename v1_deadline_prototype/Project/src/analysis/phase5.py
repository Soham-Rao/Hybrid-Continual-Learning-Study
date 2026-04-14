"""Phase 5 analysis pipeline: aggregation, trade-off analysis, and reporting."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.recommendation.engine import RecommendationEngine, RecommendationRequest

try:  # pragma: no cover - optional dependency in runtime env
    from scipy.stats import friedmanchisquare, ttest_ind
except Exception:  # pragma: no cover
    friedmanchisquare = None
    ttest_ind = None


METRIC_KEYS = ["avg_accuracy", "forgetting", "backward_transfer", "forward_transfer"]
DATASET_ORDER = ["permuted_mnist", "split_cifar10", "split_cifar100", "split_mini_imagenet"]
MAIN_SOURCE_GROUPS = {"baseline_fwt", "hybrid_fwt", "hybrid_fwt_sider_fix", "phase4_local_mini"}
SOURCE_PRIORITY = {
    "interaction_ablations": 0,
    "hybrid_fwt_sider_fix": 1,
    "phase4_local_mini": 2,
    "hybrid_fwt": 3,
    "baseline_fwt": 4,
    "unknown": 9,
}
QUALITY_NOTES = {
    ("permuted_mnist", "agem"): "Historical epoch-1 archive currently contains 3 seeds.",
    ("permuted_mnist", "ewc"): "Historical epoch-1 archive currently contains 3 seeds.",
    ("permuted_mnist", "joint_training"): "Historical epoch-1 archive currently contains 3 seeds.",
    ("split_cifar100", "si_der"): "Current fixed Split CIFAR-100 SI-DER archive contains 4 seeds.",
    ("split_cifar100", "xder"): "Pre-fix Split CIFAR-100 X-DER results may reflect historical replay-logit instability.",
    ("split_cifar10", "agem"): "Epoch-1 AGEM-family results predate the later AGEM gradient/AMP fix.",
    ("split_cifar10", "agem_distill"): "Epoch-1 AGEM-family results predate the later AGEM gradient/AMP fix.",
    ("split_cifar100", "agem"): "Epoch-1 AGEM-family results predate the later AGEM gradient/AMP fix.",
    ("split_cifar100", "agem_distill"): "Epoch-1 AGEM-family results predate the later AGEM gradient/AMP fix.",
}
LOG_PATTERNS = [("epoch_1", "epoch_1/**/*.log"), ("phase4", "phase4/local_mini/raw/*.log")]


@dataclass
class RunRecord:
    dataset: str
    method: str
    method_variant: str
    run_tag: str
    is_primary_run: bool
    seed: int
    run_name: str
    phase: str
    source_group: str
    model: str
    n_epochs: int
    fp16: bool
    buffer_size: int
    total_time_sec: float
    log_path: str
    metrics_csv_path: str
    avg_accuracy: float
    forgetting: float
    backward_transfer: float
    forward_transfer: float
    data_quality_note: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_config(text: str) -> Dict:
    match = re.search(r"Config:\s*(\{.*?\})\s*\n", text, flags=re.S)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return {}


def _parse_metrics(text: str) -> Optional[Dict[str, float]]:
    match = re.search(r"Metrics:\s*(\{.*?\})", text)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    return {key: float(value) for key, value in parsed.items() if key in METRIC_KEYS}


def _parse_total_time(metrics_csv_path: Path, text: str) -> float:
    if metrics_csv_path.exists():
        df = pd.read_csv(metrics_csv_path)
        if "time_sec" in df.columns:
            end_rows = df[df["epoch"].astype(str) == "END"]
            values = pd.to_numeric(end_rows["time_sec"], errors="coerce").dropna()
            if not values.empty:
                return float(values.sum())
    matches = re.findall(r"time_sec=([0-9.]+)", text)
    return float(sum(float(item) for item in matches))


def _infer_dataset_and_method(config: Dict, path: Path) -> Tuple[str, str, int]:
    dataset = str(config.get("dataset", "unknown"))
    method = str(config.get("method", "unknown"))
    seed = int(config.get("seed", -1))
    if dataset != "unknown" and method != "unknown" and seed != -1:
        return dataset, method, seed

    stem = path.stem
    match = re.search(r"_seed(\d+)$", stem)
    if match:
        seed = int(match.group(1))
        prefix = stem[: match.start()]
    else:
        prefix = stem
    parts = prefix.split("_")
    for idx in range(1, len(parts)):
        candidate_dataset = "_".join(parts[:idx])
        candidate_method = "_".join(parts[idx:])
        if candidate_dataset in DATASET_ORDER or candidate_dataset == "seq_tiny_imagenet":
            return candidate_dataset, candidate_method, seed
    return dataset, method, seed


def _extract_run_tag(run_name: str, dataset: str, method: str, seed: int, config: Dict) -> str:
    run_tag = str(config.get("run_tag", "") or "")
    if run_tag:
        return run_tag
    prefix = f"{dataset}_{method}"
    suffix = f"_seed{seed}"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        middle = run_name[len(prefix):-len(suffix)]
        return middle[1:] if middle.startswith("_") else middle
    return ""


def _infer_source_group(path: Path, run_tag: str) -> str:
    normalized = path.as_posix()
    if "phase4/local_mini/raw" in normalized:
        return "phase4_local_mini"
    if run_tag or "/ablations/" in normalized:
        return "interaction_ablations"
    if "/baselines/" in normalized:
        return "baseline_fwt"
    if "/hybrids/" in normalized:
        if "sider_fix" in normalized:
            return "hybrid_fwt_sider_fix"
        return "hybrid_fwt"
    return "unknown"


def _quality_note(dataset: str, method: str) -> str:
    return QUALITY_NOTES.get((dataset, method), "")


def collect_run_records(results_root: Path) -> pd.DataFrame:
    records: List[RunRecord] = []
    for phase_name, pattern in LOG_PATTERNS:
        for log_path in sorted(results_root.glob(pattern)):
            text = _read_text(log_path)
            metrics = _parse_metrics(text)
            if not metrics:
                continue
            config = _parse_config(text)
            dataset, method, seed = _infer_dataset_and_method(config, log_path)
            run_name = str(config.get("run_name", log_path.stem))
            run_tag = _extract_run_tag(run_name, dataset, method, seed, config)
            method_variant = method if not run_tag else f"{method}_{run_tag}"
            metrics_csv_path = log_path.with_name(f"{log_path.stem}_metrics.csv")
            records.append(
                RunRecord(
                    dataset=dataset,
                    method=method,
                    method_variant=method_variant,
                    run_tag=run_tag,
                    is_primary_run=(run_tag == ""),
                    seed=seed,
                    run_name=run_name,
                    phase=phase_name,
                    source_group=_infer_source_group(log_path, run_tag),
                    model=str(config.get("model", "unknown")),
                    n_epochs=int(config.get("n_epochs", 1)),
                    fp16=bool(config.get("fp16", False)),
                    buffer_size=int(config.get("buffer_size", 0) or 0),
                    total_time_sec=round(_parse_total_time(metrics_csv_path, text), 4),
                    log_path=str(log_path),
                    metrics_csv_path=str(metrics_csv_path),
                    avg_accuracy=float(metrics["avg_accuracy"]),
                    forgetting=float(metrics["forgetting"]),
                    backward_transfer=float(metrics["backward_transfer"]),
                    forward_transfer=float(metrics["forward_transfer"]),
                    data_quality_note=_quality_note(dataset, method),
                )
            )

    df = pd.DataFrame([record.__dict__ for record in records])
    if df.empty:
        return df
    df["source_priority"] = df["source_group"].map(SOURCE_PRIORITY).fillna(99)
    df = df.sort_values(["run_name", "source_priority", "log_path"]).drop_duplicates(subset=["run_name"], keep="first")
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "method_variant", "seed"]).reset_index(drop=True)


def build_master_results(results_root: Path, output_dir: Path) -> pd.DataFrame:
    master = collect_run_records(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_dir / "master_results.csv", index=False)
    master.to_csv(results_root / "master_results.csv", index=False)
    return master


def build_summary_tables(master_df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "dataset",
        "method",
        "method_variant",
        "run_tag",
        "is_primary_run",
        "source_group",
        "phase",
        "model",
        "n_epochs",
    ]
    summary = master_df.groupby(group_cols, dropna=False).agg(
        seeds=("seed", "count"),
        avg_accuracy_mean=("avg_accuracy", "mean"),
        avg_accuracy_std=("avg_accuracy", "std"),
        forgetting_mean=("forgetting", "mean"),
        forgetting_std=("forgetting", "std"),
        backward_transfer_mean=("backward_transfer", "mean"),
        backward_transfer_std=("backward_transfer", "std"),
        forward_transfer_mean=("forward_transfer", "mean"),
        forward_transfer_std=("forward_transfer", "std"),
        total_time_sec_mean=("total_time_sec", "mean"),
        total_time_sec_std=("total_time_sec", "std"),
        buffer_size_mean=("buffer_size", "mean"),
        fp16_any=("fp16", "max"),
        data_quality_note=("data_quality_note", lambda s: " | ".join(sorted({item for item in s if item}))),
    ).reset_index()

    for metric in METRIC_KEYS:
        summary[f"{metric}_mean_std"] = summary.apply(
            lambda row: f"{row[f'{metric}_mean']:.2f} ± {0.0 if pd.isna(row[f'{metric}_std']) else row[f'{metric}_std']:.2f}",
            axis=1,
        )

    summary["runtime_hours_mean"] = summary["total_time_sec_mean"] / 3600.0
    summary["dataset"] = pd.Categorical(summary["dataset"], categories=DATASET_ORDER, ordered=True)
    summary = summary.sort_values(["dataset", "is_primary_run", "avg_accuracy_mean"], ascending=[True, False, False]).reset_index(drop=True)

    long_rows: List[Dict[str, object]] = []
    for row in summary.itertuples(index=False):
        for metric in METRIC_KEYS:
            long_rows.append(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "method_variant": row.method_variant,
                    "source_group": row.source_group,
                    "phase": row.phase,
                    "run_tag": row.run_tag,
                    "is_primary_run": row.is_primary_run,
                    "metric": metric,
                    "n": row.seeds,
                    "mean": getattr(row, f"{metric}_mean"),
                    "std": getattr(row, f"{metric}_std"),
                    "data_quality_note": row.data_quality_note,
                }
            )
    summary_metrics = pd.DataFrame(long_rows)

    pretty = summary[
        [
            "dataset",
            "method",
            "method_variant",
            "run_tag",
            "is_primary_run",
            "source_group",
            "phase",
            "model",
            "seeds",
            "avg_accuracy_mean_std",
            "forgetting_mean_std",
            "backward_transfer_mean_std",
            "forward_transfer_mean_std",
            "runtime_hours_mean",
            "buffer_size_mean",
            "data_quality_note",
        ]
    ].copy()
    pretty["runtime_hours_mean"] = pretty["runtime_hours_mean"].map(lambda x: f"{x:.2f}")
    pretty["buffer_size_mean"] = pretty["buffer_size_mean"].map(lambda x: f"{x:.0f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    summary.to_csv(output_dir / "paper_ready_summary.csv", index=False)
    pretty.to_csv(output_dir / "paper_ready_summary_pretty.csv", index=False)
    return summary, pretty, summary_metrics


def _primary_main_master(master_df: pd.DataFrame) -> pd.DataFrame:
    return master_df[(master_df["is_primary_run"] == True) & (master_df["source_group"].isin(MAIN_SOURCE_GROUPS))].copy()


def _primary_main_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[(summary_df["is_primary_run"] == True) & (summary_df["source_group"].isin(MAIN_SOURCE_GROUPS))].copy()


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan")
    mean_diff = float(np.mean(sample_a) - np.mean(sample_b))
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled_num = (len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b
    pooled_den = len(sample_a) + len(sample_b) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_std = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return mean_diff / pooled_std if pooled_std > 0 else float("nan")


def build_pairwise_tests(master_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    main_df = _primary_main_master(master_df)
    rows: List[Dict[str, object]] = []
    for dataset, dataset_df in main_df.groupby("dataset"):
        methods = sorted(dataset_df["method"].unique())
        for idx, method_a in enumerate(methods):
            for method_b in methods[idx + 1 :]:
                sample_a = dataset_df.loc[dataset_df["method"] == method_a, "avg_accuracy"].to_numpy(dtype=float)
                sample_b = dataset_df.loc[dataset_df["method"] == method_b, "avg_accuracy"].to_numpy(dtype=float)
                stat = float("nan")
                pvalue = float("nan")
                if ttest_ind is not None and len(sample_a) >= 2 and len(sample_b) >= 2:
                    result = ttest_ind(sample_a, sample_b, equal_var=False)
                    stat = float(result.statistic)
                    pvalue = float(result.pvalue)
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": "avg_accuracy",
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_a": len(sample_a),
                        "n_b": len(sample_b),
                        "mean_a": float(np.mean(sample_a)),
                        "mean_b": float(np.mean(sample_b)),
                        "mean_diff": float(np.mean(sample_a) - np.mean(sample_b)),
                        "t_stat": stat,
                        "p_value": pvalue,
                        "cohens_d": _cohens_d(sample_a, sample_b),
                    }
                )
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(output_dir / "pairwise_tests.csv", index=False)
    return pairwise


def build_friedman_table(master_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if friedmanchisquare is None:
        friedman = pd.DataFrame(rows)
        friedman.to_csv(output_dir / "friedman_avg_accuracy.csv", index=False)
        return friedman

    main_df = _primary_main_master(master_df)
    for dataset, dataset_df in main_df.groupby("dataset"):
        method_seeds = dataset_df.groupby("method")["seed"].apply(lambda s: set(int(v) for v in s)).to_dict()
        if not method_seeds:
            continue
        common_seeds = sorted(set.intersection(*method_seeds.values())) if len(method_seeds) > 1 else sorted(next(iter(method_seeds.values())))
        if len(common_seeds) < 2:
            continue
        arrays: List[List[float]] = []
        final_methods: List[str] = []
        for method in sorted(method_seeds):
            values = []
            method_df = dataset_df[dataset_df["method"] == method]
            for seed in common_seeds:
                match = method_df.loc[method_df["seed"] == seed, "avg_accuracy"]
                if match.empty:
                    values = []
                    break
                values.append(float(match.iloc[0]))
            if values:
                arrays.append(values)
                final_methods.append(method)
        if len(arrays) < 3:
            continue
        stat = friedmanchisquare(*arrays)
        rows.append(
            {
                "dataset": dataset,
                "metric": "avg_accuracy",
                "n_methods": len(arrays),
                "n_blocks": len(common_seeds),
                "methods": ",".join(final_methods),
                "statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
            }
        )
    friedman = pd.DataFrame(rows)
    friedman.to_csv(output_dir / "friedman_avg_accuracy.csv", index=False)
    return friedman


def _is_non_dominated(acc: float, cost_a: float, cost_b: float, pool: Iterable[Tuple[float, float, float]]) -> bool:
    for other_acc, other_cost_a, other_cost_b in pool:
        if other_acc >= acc and other_cost_a <= cost_a and other_cost_b <= cost_b:
            if other_acc > acc or other_cost_a < cost_a or other_cost_b < cost_b:
                return False
    return True


def build_pareto_tables(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    main_df = _primary_main_summary(summary_df)
    rows: List[Dict[str, object]] = []
    for dataset, dataset_df in main_df.groupby("dataset"):
        points = [(float(row.avg_accuracy_mean), float(row.buffer_size_mean), float(row.total_time_sec_mean), row.method) for row in dataset_df.itertuples(index=False)]
        simple_pool = [(acc, mem, runtime) for acc, mem, runtime, _ in points]
        for acc, mem, runtime, method in points:
            if _is_non_dominated(acc, mem, runtime, simple_pool):
                rows.append({"dataset": dataset, "method": method, "avg_accuracy_mean": acc, "buffer_size_mean": mem, "runtime_hours_mean": runtime / 3600.0})
    pareto = pd.DataFrame(rows).sort_values(["dataset", "avg_accuracy_mean"], ascending=[True, False])
    pareto.to_csv(output_dir / "pareto_frontier_candidates.csv", index=False)
    return pareto


def _ensure_fig_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save_heatmap(summary_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    fig_dir = _ensure_fig_dir(output_dir)
    pivot = _primary_main_summary(summary_df).pivot_table(index="dataset", columns="method", values=f"{metric}_mean")
    if pivot.empty:
        return
    pivot = pivot.reindex(DATASET_ORDER).dropna(how="all")
    plt.figure(figsize=(12, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn" if metric == "avg_accuracy" else "RdYlGn_r")
    plt.title(f"Phase 5 {metric.replace('_', ' ').title()} Heatmap")
    plt.xlabel("Method")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(fig_dir / f"phase5_{metric}_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()


def _save_tradeoff_scatter(summary_df: pd.DataFrame, output_dir: Path, dataset: str, x_col: str, filename: str, xlabel: str) -> None:
    fig_dir = _ensure_fig_dir(output_dir)
    subset = _primary_main_summary(summary_df)
    subset = subset[subset["dataset"] == dataset].copy()
    if subset.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=subset, x=x_col, y="avg_accuracy_mean", hue="method", s=110)
    for row in subset.itertuples(index=False):
        plt.text(getattr(row, x_col), row.avg_accuracy_mean, row.method, fontsize=8)
    plt.title(f"{dataset}: accuracy trade-off")
    plt.xlabel(xlabel)
    plt.ylabel("Average Accuracy (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=160, bbox_inches="tight")
    plt.close()


def save_phase5_figures(summary_df: pd.DataFrame, output_dir: Path) -> None:
    _save_heatmap(summary_df, "avg_accuracy", output_dir)
    _save_heatmap(summary_df, "forgetting", output_dir)
    datasets = [item for item in DATASET_ORDER if item in set(summary_df["dataset"].astype(str))]
    for dataset in datasets:
        _save_tradeoff_scatter(summary_df, output_dir, dataset, "buffer_size_mean", f"{dataset}_accuracy_vs_buffer.png", "Replay Buffer Size (samples)")
        _save_tradeoff_scatter(summary_df, output_dir, dataset, "runtime_hours_mean", f"{dataset}_accuracy_vs_runtime.png", "Runtime (hours)")


def build_case_studies(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    engine = RecommendationEngine(summary_df)
    requests = [
        RecommendationRequest(dataset="split_mini_imagenet", memory_budget_mb=250.0, compute_budget="medium", acceptable_forgetting=20.0, task_similarity="medium", joint_retraining_allowed=False),
        RecommendationRequest(dataset="split_mini_imagenet", memory_budget_mb=1200.0, compute_budget="high", acceptable_forgetting=15.0, task_similarity="low", joint_retraining_allowed=False),
        RecommendationRequest(dataset="split_mini_imagenet", memory_budget_mb=4096.0, compute_budget="high", acceptable_forgetting=12.0, task_similarity="medium", joint_retraining_allowed=True),
        RecommendationRequest(dataset="split_cifar10", memory_budget_mb=96.0, compute_budget="low", acceptable_forgetting=40.0, task_similarity="high", joint_retraining_allowed=False),
        RecommendationRequest(dataset="permuted_mnist", memory_budget_mb=32.0, compute_budget="low", acceptable_forgetting=40.0, task_similarity="high", joint_retraining_allowed=False),
    ]
    rows: List[Dict[str, object]] = []
    for idx, request in enumerate(requests, start=1):
        result = engine.recommend(request, top_k=3)
        for rank, candidate in enumerate(result["shortlist"], start=1):
            rows.append(
                {
                    "case_id": idx,
                    "dataset": request.dataset,
                    "memory_budget_mb": request.memory_budget_mb,
                    "compute_budget": request.compute_budget,
                    "acceptable_forgetting": request.acceptable_forgetting,
                    "task_similarity": request.task_similarity,
                    "joint_retraining_allowed": request.joint_retraining_allowed,
                    "rank": rank,
                    "method": candidate["method"],
                    "score": candidate["score"],
                    "avg_accuracy_mean": candidate["avg_accuracy_mean"],
                    "forgetting_mean": candidate["forgetting_mean"],
                    "runtime_hours_mean": candidate["runtime_hours_mean"],
                    "estimated_memory_mb": candidate["estimated_memory_mb"],
                    "rationale": " | ".join(candidate["reasons"]),
                }
            )
    cases = pd.DataFrame(rows)
    cases.to_csv(output_dir / "recommendation_cases.csv", index=False)
    return cases


def write_phase5_report(summary_df: pd.DataFrame, pareto_df: pd.DataFrame, case_df: pd.DataFrame, output_dir: Path) -> None:
    main_df = _primary_main_summary(summary_df)
    lines = [
        "# Phase 5 Analysis Report",
        "",
        "This file is generated from the current repository result archives.",
        "It combines Phase 1-3 epoch-1 results with the completed Phase 4 Mini-ImageNet local sweep.",
        "",
        "## Dataset Leaders",
        "",
    ]
    for dataset in DATASET_ORDER:
        subset = main_df[main_df["dataset"] == dataset]
        if subset.empty:
            continue
        top3 = subset.sort_values("avg_accuracy_mean", ascending=False).head(3)
        lines.append(f"### {dataset}")
        for row in top3.itertuples(index=False):
            lines.append(f"- `{row.method}`: {row.avg_accuracy_mean:.2f} ± {0.0 if pd.isna(row.avg_accuracy_std) else row.avg_accuracy_std:.2f} AA, forgetting {row.forgetting_mean:.2f}, runtime {row.runtime_hours_mean:.2f} h.")
        lines.append("")

    lines.extend(["## Pareto Candidates", ""])
    for dataset in DATASET_ORDER:
        subset = pareto_df[pareto_df["dataset"] == dataset]
        if subset.empty:
            continue
        lines.append(f"- `{dataset}`: {', '.join(subset['method'].tolist())}")
    lines.append("")

    lines.extend(["## Recommendation Cases", ""])
    for case_id in sorted(case_df["case_id"].unique()):
        subset = case_df[case_df["case_id"] == case_id].sort_values("rank")
        if subset.empty:
            continue
        head = subset.iloc[0]
        lines.append(f"- Case {case_id} on `{head['dataset']}` -> recommend `{head['method']}` (memory {head['memory_budget_mb']} MB, compute {head['compute_budget']}, forgetting <= {head['acceptable_forgetting']}).")
    lines.append("")

    cautions = [note for note in summary_df["data_quality_note"].dropna().astype(str).unique() if note]
    if cautions:
        lines.extend(["## Cautions", ""])
        for note in sorted(cautions):
            lines.append(f"- {note}")
        lines.append("")

    (output_dir / "phase5_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase5_pipeline(results_root: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    master = build_master_results(results_root, output_dir)
    if master.empty:
        raise RuntimeError("No runnable study logs were found for Phase 5 aggregation.")
    summary, pretty, summary_metrics = build_summary_tables(master, output_dir)
    pairwise = build_pairwise_tests(master, output_dir)
    friedman = build_friedman_table(master, output_dir)
    pareto = build_pareto_tables(summary, output_dir)
    save_phase5_figures(summary, output_dir)
    cases = build_case_studies(summary, output_dir)
    write_phase5_report(summary, pareto, cases, output_dir)
    return {
        "master": master,
        "summary": summary,
        "pretty": pretty,
        "summary_metrics": summary_metrics,
        "pairwise": pairwise,
        "friedman": friedman,
        "pareto": pareto,
        "cases": cases,
    }
