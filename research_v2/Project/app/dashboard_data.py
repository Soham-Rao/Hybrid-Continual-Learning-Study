"""Artifact loading and dataframe shaping helpers for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from src.recommendation.engine import RecommendationEngine, RecommendationRequest
from src.utils.paths import RESULTS_ROOT


PRIMARY_ANALYSIS_DIR = RESULTS_ROOT / "analysis" / "epoch_1"
ABLATION_ANALYSIS_DIR = RESULTS_ROOT / "analysis" / "ablations" / "epoch_1"
FIGURE_DIR = RESULTS_ROOT / "figures" / "epoch_1" / "analysis"

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

DATASET_LABELS = {
    "permuted_mnist": "Permuted MNIST",
    "split_cifar10": "Split CIFAR-10",
    "split_cifar100": "Split CIFAR-100",
    "split_mini_imagenet": "Split Mini-ImageNet",
}

MEMORY_BUCKETS = [
    {"key": "mem_tight", "label": "Tight Memory (<=64 MB)", "min": 0.0, "max": 64.0, "center": 32.0},
    {"key": "mem_balanced", "label": "Balanced Memory (65-256 MB)", "min": 65.0, "max": 256.0, "center": 128.0},
    {"key": "mem_roomy", "label": "Roomy Memory (257-1024 MB)", "min": 257.0, "max": 1024.0, "center": 512.0},
    {"key": "mem_wide", "label": "Wide Memory (>1024 MB)", "min": 1025.0, "max": None, "center": 2048.0},
]

FORGETTING_BUCKETS = [
    {"key": "forgetting_strict", "label": "Strict Retention (<=15)", "min": 0.0, "max": 15.0, "center": 10.0},
    {"key": "forgetting_balanced", "label": "Balanced Retention (16-35)", "min": 16.0, "max": 35.0, "center": 25.0},
    {"key": "forgetting_flexible", "label": "Flexible Retention (>35)", "min": 36.0, "max": None, "center": 50.0},
]

COMPUTE_BUCKETS = [
    {"key": "low", "label": "Low Compute"},
    {"key": "medium", "label": "Medium Compute"},
    {"key": "high", "label": "High Compute"},
]

SIMILARITY_BUCKETS = [
    {"key": "low", "label": "Low Similarity"},
    {"key": "medium", "label": "Medium Similarity"},
    {"key": "high", "label": "High Similarity"},
]

JOINT_BUCKETS = [
    {"key": "joint_off", "label": "Joint Retraining Off", "value": False},
    {"key": "joint_on", "label": "Joint Retraining On", "value": True},
]

TREE_LEVELS = [
    ("dataset_label", "Dataset"),
    ("memory_bucket_label", "Memory"),
    ("compute_bucket_label", "Compute"),
    ("forgetting_bucket_label", "Retention"),
    ("similarity_bucket_label", "Task Similarity"),
    ("joint_bucket_label", "Joint Retraining"),
    ("method_label", "Recommended Method"),
]

PRIMARY_ARTIFACTS = {
    "summary": PRIMARY_ANALYSIS_DIR / "paper_ready_summary.csv",
    "summary_pretty": PRIMARY_ANALYSIS_DIR / "paper_ready_summary_pretty.csv",
    "leaders": PRIMARY_ANALYSIS_DIR / "dataset_leaders.csv",
    "pairwise": PRIMARY_ANALYSIS_DIR / "pairwise_tests.csv",
    "effect_sizes": PRIMARY_ANALYSIS_DIR / "effect_sizes.csv",
    "friedman": PRIMARY_ANALYSIS_DIR / "friedman_avg_accuracy.csv",
    "pareto": PRIMARY_ANALYSIS_DIR / "pareto_frontier_candidates.csv",
    "report": PRIMARY_ANALYSIS_DIR / "phase7_report.md",
    "recommendation_profiles": PRIMARY_ANALYSIS_DIR / "recommendation_profiles.csv",
    "recommendation_cases": PRIMARY_ANALYSIS_DIR / "recommendation_cases.csv",
    "recommendation_notes": PRIMARY_ANALYSIS_DIR / "phase8_recommendation_notes.md",
}

SECONDARY_ARTIFACTS = {
    "ablation_current": ABLATION_ANALYSIS_DIR / "current_results.csv",
    "ablation_runtime": ABLATION_ANALYSIS_DIR / "runtime_sensitivity_summary.csv",
    "ablation_memory": ABLATION_ANALYSIS_DIR / "memory_sensitivity_summary.csv",
    "ablation_robustness": ABLATION_ANALYSIS_DIR / "robustness_summary.csv",
    "ablation_notes": ABLATION_ANALYSIS_DIR / "resource_summary_notes.md",
}

STATIC_FIGURES = {
    "avg_accuracy_heatmap": FIGURE_DIR / "phase7_avg_accuracy_heatmap.png",
    "forgetting_heatmap": FIGURE_DIR / "phase7_forgetting_heatmap.png",
}

ARTIFACT_LABELS = {
    "summary": "Study Summary",
    "summary_pretty": "Presentation Summary",
    "leaders": "Dataset Leaders",
    "pairwise": "Pairwise Tests",
    "effect_sizes": "Effect Sizes",
    "friedman": "Cross-Dataset Ranking",
    "pareto": "Pareto Candidates",
    "report": "Analysis Report",
    "recommendation_profiles": "Recommendation Profiles",
    "recommendation_cases": "Recommendation Cases",
    "recommendation_notes": "Recommendation Notes",
    "ablation_current": "Ablation Results",
    "ablation_runtime": "Ablation Runtime Summary",
    "ablation_memory": "Ablation Memory Summary",
    "ablation_robustness": "Resume Robustness Summary",
    "ablation_notes": "Ablation Resource Notes",
}

NUMERIC_COLUMNS = {
    "n_epochs",
    "batch_size",
    "num_workers",
    "buffer_size",
    "seeds",
    "total_time_sec_mean",
    "total_time_sec_std",
    "avg_accuracy_mean",
    "avg_accuracy_std",
    "forgetting_mean",
    "forgetting_std",
    "backward_transfer_mean",
    "backward_transfer_std",
    "forward_transfer_mean",
    "forward_transfer_std",
    "runtime_hours_mean",
    "runtime_hours_std",
    "estimated_memory_mb",
    "best_avg_accuracy_mean",
    "best_forgetting_mean",
    "best_runtime_hours_mean",
    "best_estimated_memory_mb",
    "top_cluster_size",
    "n_pairs",
    "n_nonzero_pairs",
    "mean_a",
    "mean_b",
    "mean_diff",
    "statistic",
    "p_value",
    "holm_adjusted_p_value",
    "rank_biserial",
    "average_rank",
    "dataset_count",
    "friedman_statistic",
    "friedman_p_value",
    "score",
    "memory_budget_mb",
    "acceptable_forgetting",
    "rank",
    "case_id",
    "mean_total_time_sec",
    "std_total_time_sec",
    "mean_task_time_sec",
    "max_task_time_sec",
    "dataset_image_size",
    "agem_mem_batch",
    "fisher_samples",
    "joint_replay_epochs",
    "n_tasks",
    "seeds_present",
    "seeds_completed",
    "seed",
    "stop_after_task",
    "log_starts",
    "total_time_sec",
    "max_task_time_sec",
    "n_completed_tasks",
    "robust_avg_accuracy",
    "robust_forgetting",
    "robust_backward_transfer",
    "robust_forward_transfer",
    "primary_avg_accuracy",
    "delta_avg_accuracy",
    "primary_forgetting",
    "delta_forgetting",
    "primary_backward_transfer",
    "delta_backward_transfer",
    "primary_forward_transfer",
    "delta_forward_transfer",
    "avg_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
}

BOOL_COLUMNS = {
    "fp16",
    "leader_flag",
    "top_cluster_flag",
    "replay",
    "distill",
    "regularize",
    "joint",
    "reject_h0",
    "joint_retraining_allowed",
    "completed",
    "restart_verified",
}


@dataclass
class DashboardBundle:
    summary: pd.DataFrame
    summary_pretty: pd.DataFrame
    leaders: pd.DataFrame
    pairwise: pd.DataFrame
    effect_sizes: pd.DataFrame
    friedman: pd.DataFrame
    pareto: pd.DataFrame
    recommendation_profiles: pd.DataFrame
    recommendation_cases: pd.DataFrame
    report_text: str
    recommendation_notes: str
    ablation_current: pd.DataFrame
    ablation_runtime: pd.DataFrame
    ablation_memory: pd.DataFrame
    ablation_robustness: pd.DataFrame
    ablation_notes: str
    static_figures: Dict[str, Path]
    missing_primary: List[str]
    missing_secondary: List[str]


def sanitize_user_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "v2": "",
        "V2": "",
        "Phase 7 Analysis Report": "Analysis Report",
        "Phase 8 Recommendation Notes": "Recommendation Notes",
        "Phase 7 report": "Analysis Report",
        "Phase 8 notes": "Recommendation Notes",
        "Phase 6 Context": "Ablation Context",
        "Phase 6 context": "Ablation Context",
        "Phase 7": "",
        "Phase 8": "",
        "Phase 6": "",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = out.replace("  ", " ").replace("`", "`")
    return out


def _bucket_label(value: float, buckets: Sequence[Dict[str, object]], default_label: str) -> str:
    numeric = float(value)
    for bucket in buckets:
        lower = float(bucket["min"])
        upper = bucket["max"]
        if upper is None and numeric >= lower:
            return str(bucket["label"])
        if upper is not None and lower <= numeric <= float(upper):
            return str(bucket["label"])
    return default_label


def _bucket_key(value: float, buckets: Sequence[Dict[str, object]], default_key: str) -> str:
    numeric = float(value)
    for bucket in buckets:
        lower = float(bucket["min"])
        upper = bucket["max"]
        if upper is None and numeric >= lower:
            return str(bucket["key"])
        if upper is not None and lower <= numeric <= float(upper):
            return str(bucket["key"])
    return default_key


def memory_bucket_label(value: float) -> str:
    return _bucket_label(value, MEMORY_BUCKETS, str(MEMORY_BUCKETS[-1]["label"]))


def memory_bucket_key(value: float) -> str:
    return _bucket_key(value, MEMORY_BUCKETS, str(MEMORY_BUCKETS[-1]["key"]))


def forgetting_bucket_label(value: float) -> str:
    return _bucket_label(value, FORGETTING_BUCKETS, str(FORGETTING_BUCKETS[-1]["label"]))


def forgetting_bucket_key(value: float) -> str:
    return _bucket_key(value, FORGETTING_BUCKETS, str(FORGETTING_BUCKETS[-1]["key"]))


def representative_memory_budget(bucket_key: str) -> float:
    for bucket in MEMORY_BUCKETS:
        if str(bucket["key"]) == str(bucket_key):
            return float(bucket["center"])
    return float(MEMORY_BUCKETS[1]["center"])


def representative_acceptable_forgetting(bucket_key: str) -> float:
    for bucket in FORGETTING_BUCKETS:
        if str(bucket["key"]) == str(bucket_key):
            return float(bucket["center"])
    return float(FORGETTING_BUCKETS[1]["center"])


def request_bucket_state(request: RecommendationRequest) -> Dict[str, object]:
    return {
        "dataset": str(request.dataset),
        "dataset_label": DATASET_LABELS.get(str(request.dataset), str(request.dataset)),
        "memory_bucket_key": memory_bucket_key(float(request.memory_budget_mb)),
        "memory_bucket_label": memory_bucket_label(float(request.memory_budget_mb)),
        "compute_bucket_key": str(request.compute_budget),
        "compute_bucket_label": next(
            (str(bucket["label"]) for bucket in COMPUTE_BUCKETS if str(bucket["key"]) == str(request.compute_budget)),
            str(request.compute_budget),
        ),
        "forgetting_bucket_key": forgetting_bucket_key(float(request.acceptable_forgetting or 25.0)),
        "forgetting_bucket_label": forgetting_bucket_label(float(request.acceptable_forgetting or 25.0)),
        "similarity_bucket_key": str(request.task_similarity),
        "similarity_bucket_label": next(
            (str(bucket["label"]) for bucket in SIMILARITY_BUCKETS if str(bucket["key"]) == str(request.task_similarity)),
            str(request.task_similarity),
        ),
        "joint_bucket_key": "joint_on" if bool(request.joint_retraining_allowed) else "joint_off",
        "joint_bucket_label": "Joint Retraining On" if bool(request.joint_retraining_allowed) else "Joint Retraining Off",
    }


def build_decision_tree_rows(profiles_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame()
    dataset_profiles = profiles_df[profiles_df["dataset"].astype(str) == str(dataset)].copy()
    if dataset_profiles.empty:
        return pd.DataFrame()

    engine = RecommendationEngine(dataset_profiles)
    rows: List[Dict[str, object]] = []
    case_id = 0
    for memory_bucket in MEMORY_BUCKETS:
        for compute_bucket in COMPUTE_BUCKETS:
            for retention_bucket in FORGETTING_BUCKETS:
                for similarity_bucket in SIMILARITY_BUCKETS:
                    for joint_bucket in JOINT_BUCKETS:
                        request = RecommendationRequest(
                            dataset=str(dataset),
                            memory_budget_mb=float(memory_bucket["center"]),
                            compute_budget=str(compute_bucket["key"]),
                            acceptable_forgetting=float(retention_bucket["center"]),
                            task_similarity=str(similarity_bucket["key"]),
                            joint_retraining_allowed=bool(joint_bucket["value"]),
                        )
                        result = engine.recommend(request, top_k=3)
                        best = result["shortlist"][0]
                        case_id += 1
                        rows.append(
                            {
                                "case_id": case_id,
                                "dataset": str(dataset),
                                "dataset_label": DATASET_LABELS.get(str(dataset), str(dataset)),
                                "memory_bucket_key": str(memory_bucket["key"]),
                                "memory_bucket_label": str(memory_bucket["label"]),
                                "compute_bucket_key": str(compute_bucket["key"]),
                                "compute_bucket_label": str(compute_bucket["label"]),
                                "forgetting_bucket_key": str(retention_bucket["key"]),
                                "forgetting_bucket_label": str(retention_bucket["label"]),
                                "similarity_bucket_key": str(similarity_bucket["key"]),
                                "similarity_bucket_label": str(similarity_bucket["label"]),
                                "joint_bucket_key": str(joint_bucket["key"]),
                                "joint_bucket_label": str(joint_bucket["label"]),
                                "joint_retraining_allowed": bool(joint_bucket["value"]),
                                "recommended_method": str(result["recommended_method"]),
                                "method_label": METHOD_LABELS.get(str(result["recommended_method"]), str(result["recommended_method"])),
                                "score": float(best["score"]),
                                "avg_accuracy_mean": float(best["avg_accuracy_mean"]),
                                "forgetting_mean": float(best["forgetting_mean"]),
                                "runtime_hours_mean": float(best["runtime_hours_mean"]),
                                "estimated_memory_mb": float(best["estimated_memory_mb"]),
                                "leader_flag": bool(best["leader_flag"]),
                                "top_cluster_flag": bool(best["top_cluster_flag"]),
                                "rationale": str(result["rationale"]),
                            }
                        )
    tree_df = pd.DataFrame(rows)
    return tree_df.sort_values(
        [
            "memory_bucket_key",
            "compute_bucket_key",
            "forgetting_bucket_key",
            "similarity_bucket_key",
            "joint_bucket_key",
            "recommended_method",
        ]
    ).reset_index(drop=True)


def filter_decision_tree_rows(tree_df: pd.DataFrame, request: RecommendationRequest, focus_level: int) -> pd.DataFrame:
    if tree_df.empty:
        return tree_df.copy()
    active = request_bucket_state(request)
    filters = [
        ("dataset", active["dataset"]),
        ("memory_bucket_key", active["memory_bucket_key"]),
        ("compute_bucket_key", active["compute_bucket_key"]),
        ("forgetting_bucket_key", active["forgetting_bucket_key"]),
        ("similarity_bucket_key", active["similarity_bucket_key"]),
        ("joint_bucket_key", active["joint_bucket_key"]),
    ]
    subset = tree_df.copy()
    for idx, (column, value) in enumerate(filters):
        if idx >= int(focus_level):
            break
        subset = subset[subset[column].astype(str) == str(value)].copy()
    return subset.reset_index(drop=True)


def visible_tree_choices(tree_df: pd.DataFrame, request: RecommendationRequest, focus_level: int) -> pd.DataFrame:
    subset = filter_decision_tree_rows(tree_df, request, focus_level)
    if subset.empty:
        return subset
    next_level_columns = [
        "memory_bucket_label",
        "compute_bucket_label",
        "forgetting_bucket_label",
        "similarity_bucket_label",
        "joint_bucket_label",
        "method_label",
    ]
    next_key_columns = [
        "memory_bucket_key",
        "compute_bucket_key",
        "forgetting_bucket_key",
        "similarity_bucket_key",
        "joint_bucket_key",
        "recommended_method",
    ]
    if int(focus_level) >= len(next_level_columns):
        return pd.DataFrame()
    label_col = next_level_columns[int(focus_level)]
    key_col = next_key_columns[int(focus_level)]
    grouped = (
        subset.groupby([key_col, label_col], dropna=False)
        .agg(
            cases=("case_id", "count"),
            mean_score=("score", "mean"),
            mean_accuracy=("avg_accuracy_mean", "mean"),
            mean_forgetting=("forgetting_mean", "mean"),
        )
        .reset_index()
        .sort_values(["mean_score", "mean_accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )
    grouped = grouped.rename(columns={key_col: "choice_key", label_col: "choice_label"})
    return grouped


def strip_markdown_section(text: str, heading: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out: List[str] = []
    skipping = False
    target = heading.strip().lower()
    for line in lines:
        normalized = line.strip().lower()
        if normalized == target:
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def _to_bool(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False}).fillna(False)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in out.columns:
        if column in NUMERIC_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        elif column in BOOL_COLUMNS:
            out[column] = _to_bool(out[column])
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _coerce_types(pd.read_csv(path))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_dashboard_bundle() -> DashboardBundle:
    missing_primary = [str(path) for path in PRIMARY_ARTIFACTS.values() if not path.exists()]
    missing_secondary = [str(path) for path in SECONDARY_ARTIFACTS.values() if not path.exists()]
    return DashboardBundle(
        summary=_read_csv(PRIMARY_ARTIFACTS["summary"]),
        summary_pretty=_read_csv(PRIMARY_ARTIFACTS["summary_pretty"]),
        leaders=_read_csv(PRIMARY_ARTIFACTS["leaders"]),
        pairwise=_read_csv(PRIMARY_ARTIFACTS["pairwise"]),
        effect_sizes=_read_csv(PRIMARY_ARTIFACTS["effect_sizes"]),
        friedman=_read_csv(PRIMARY_ARTIFACTS["friedman"]),
        pareto=_read_csv(PRIMARY_ARTIFACTS["pareto"]),
        recommendation_profiles=_read_csv(PRIMARY_ARTIFACTS["recommendation_profiles"]),
        recommendation_cases=_read_csv(PRIMARY_ARTIFACTS["recommendation_cases"]),
        report_text=sanitize_user_text(_read_text(PRIMARY_ARTIFACTS["report"])),
        recommendation_notes=sanitize_user_text(_read_text(PRIMARY_ARTIFACTS["recommendation_notes"])),
        ablation_current=_read_csv(SECONDARY_ARTIFACTS["ablation_current"]),
        ablation_runtime=_read_csv(SECONDARY_ARTIFACTS["ablation_runtime"]),
        ablation_memory=_read_csv(SECONDARY_ARTIFACTS["ablation_memory"]),
        ablation_robustness=_read_csv(SECONDARY_ARTIFACTS["ablation_robustness"]),
        ablation_notes=sanitize_user_text(_read_text(SECONDARY_ARTIFACTS["ablation_notes"])),
        static_figures=STATIC_FIGURES,
        missing_primary=missing_primary,
        missing_secondary=missing_secondary,
    )


def dataset_options(df: pd.DataFrame) -> List[str]:
    if df.empty or "dataset" not in df.columns:
        return DATASET_ORDER
    values = df["dataset"].dropna().astype(str).unique().tolist()
    ordered = [item for item in DATASET_ORDER if item in values]
    ordered.extend([item for item in values if item not in ordered])
    return ordered


def filter_profiles(
    profiles_df: pd.DataFrame,
    dataset: str,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    include_joint: bool = True,
    top_cluster_only: bool = False,
) -> pd.DataFrame:
    if profiles_df.empty:
        return profiles_df.copy()
    subset = profiles_df[profiles_df["dataset"].astype(str) == str(dataset)].copy()
    if not include_joint:
        subset = subset[subset["method"].astype(str) != "joint_training"].copy()
    if families:
        subset = subset[subset["method_family"].astype(str).isin([str(item) for item in families])].copy()
    if methods:
        subset = subset[subset["method"].astype(str).isin([str(item) for item in methods])].copy()
    if top_cluster_only and "top_cluster_flag" in subset.columns:
        subset = subset[subset["top_cluster_flag"] == True].copy()  # noqa: E712
    subset["method_label"] = subset["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item)))
    subset["dataset_label"] = subset["dataset"].map(lambda item: DATASET_LABELS.get(str(item), str(item)))
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    subset["method_order"] = subset["method"].map(lambda item: order.get(str(item), 999))
    return subset.sort_values(["method_order", "avg_accuracy_mean"], ascending=[True, False]).reset_index(drop=True)


def comparison_table(
    profiles_df: pd.DataFrame,
    dataset: str,
    families: Sequence[str] | None,
    include_joint: bool,
    top_cluster_only: bool,
    sort_by: str,
) -> pd.DataFrame:
    subset = filter_profiles(
        profiles_df,
        dataset=dataset,
        families=families,
        include_joint=include_joint,
        top_cluster_only=top_cluster_only,
    )
    if subset.empty:
        return subset
    display = subset[
        [
            "method_label",
            "method_family",
            "avg_accuracy_mean",
            "forgetting_mean",
            "backward_transfer_mean",
            "forward_transfer_mean",
            "runtime_hours_mean",
            "estimated_memory_mb",
            "leader_flag",
            "top_cluster_flag",
            "caveat_note",
        ]
    ].copy()
    display = display.rename(
        columns={
            "method_label": "Method",
            "method_family": "Family",
            "avg_accuracy_mean": "Avg Accuracy",
            "forgetting_mean": "Forgetting",
            "backward_transfer_mean": "BWT",
            "forward_transfer_mean": "FWT",
            "runtime_hours_mean": "Runtime (h)",
            "estimated_memory_mb": "Memory Proxy (MB)",
            "leader_flag": "Leader",
            "top_cluster_flag": "Top Cluster",
            "caveat_note": "Notes",
        }
    )
    ascending = sort_by in {"Forgetting", "Runtime (h)", "Memory Proxy (MB)"}
    return display.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def build_pairwise_matrix(pairwise_df: pd.DataFrame, dataset: str, metric: str, value_col: str) -> pd.DataFrame:
    methods = METHOD_ORDER
    matrix = pd.DataFrame(0.0, index=methods, columns=methods)
    if value_col == "holm_adjusted_p_value":
        matrix.loc[:, :] = 1.0
        for method in methods:
            matrix.loc[method, method] = 0.0
    if pairwise_df.empty:
        return matrix
    subset = pairwise_df[
        (pairwise_df["dataset"].astype(str) == str(dataset))
        & (pairwise_df["metric"].astype(str) == str(metric))
    ]
    for row in subset.itertuples(index=False):
        value = float(getattr(row, value_col))
        matrix.loc[str(row.method_a), str(row.method_b)] = value
        matrix.loc[str(row.method_b), str(row.method_a)] = value
    return matrix.rename(index=METHOD_LABELS, columns=METHOD_LABELS)


def build_effect_matrix(effect_df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    methods = METHOD_ORDER
    matrix = pd.DataFrame(0.0, index=methods, columns=methods)
    if effect_df.empty:
        return matrix.rename(index=METHOD_LABELS, columns=METHOD_LABELS)
    subset = effect_df[
        (effect_df["dataset"].astype(str) == str(dataset))
        & (effect_df["metric"].astype(str) == str(metric))
    ]
    for row in subset.itertuples(index=False):
        value = float(row.rank_biserial)
        matrix.loc[str(row.method_a), str(row.method_b)] = value
        matrix.loc[str(row.method_b), str(row.method_a)] = -value
    return matrix.rename(index=METHOD_LABELS, columns=METHOD_LABELS)


def build_rank_dataframe(profiles_df: pd.DataFrame, include_joint: bool) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame()
    source = profiles_df.copy()
    if not include_joint:
        source = source[source["method"].astype(str) != "joint_training"].copy()
    rows: List[Dict[str, object]] = []
    for dataset in dataset_options(source):
        subset = source[source["dataset"].astype(str) == dataset].copy()
        if subset.empty:
            continue
        subset["rank"] = subset["avg_accuracy_mean"].rank(ascending=False, method="average")
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "dataset": row.dataset,
                    "dataset_label": DATASET_LABELS.get(str(row.dataset), str(row.dataset)),
                    "method": row.method,
                    "method_label": METHOD_LABELS.get(str(row.method), str(row.method)),
                    "rank": float(row.rank),
                    "avg_accuracy_mean": float(row.avg_accuracy_mean),
                    "top_cluster_flag": bool(row.top_cluster_flag),
                }
            )
    return pd.DataFrame(rows)


def build_top_cluster_membership(profiles_df: pd.DataFrame, include_joint: bool) -> pd.DataFrame:
    ranks = build_rank_dataframe(profiles_df, include_joint=include_joint)
    if ranks.empty:
        return ranks
    membership = (
        ranks.groupby(["method", "method_label"], dropna=False)
        .agg(
            datasets_in_top_cluster=("top_cluster_flag", "sum"),
            mean_rank=("rank", "mean"),
            mean_avg_accuracy=("avg_accuracy_mean", "mean"),
        )
        .reset_index()
        .sort_values(["datasets_in_top_cluster", "mean_rank"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return membership


def dataset_snapshot(profiles_df: pd.DataFrame, leaders_df: pd.DataFrame, dataset: str, include_joint: bool) -> Dict[str, object]:
    subset = filter_profiles(profiles_df, dataset=dataset, include_joint=include_joint)
    leader = subset.sort_values(["avg_accuracy_mean", "forgetting_mean"], ascending=[False, True]).iloc[0] if not subset.empty else None
    top_cluster = subset[subset["top_cluster_flag"] == True].copy() if not subset.empty else subset
    if top_cluster.empty and leader is not None:
        top_cluster = subset[subset["method"].astype(str) == str(leader["method"])].copy()
    fastest = top_cluster.sort_values("runtime_hours_mean").iloc[0] if not top_cluster.empty else None
    lightest = top_cluster.sort_values("estimated_memory_mb").iloc[0] if not top_cluster.empty else None
    return {
        "best_method": None if leader is None else str(leader["method"]),
        "best_method_label": None if leader is None else METHOD_LABELS.get(str(leader["method"]), str(leader["method"])),
        "top_cluster_size": 0 if top_cluster.empty else int(len(top_cluster)),
        "fastest_top_cluster": None if fastest is None else str(fastest["method"]),
        "fastest_top_cluster_label": None if fastest is None else METHOD_LABELS.get(str(fastest["method"]), str(fastest["method"])),
        "lowest_memory_top_cluster": None if lightest is None else str(lightest["method"]),
        "lowest_memory_top_cluster_label": None if lightest is None else METHOD_LABELS.get(str(lightest["method"]), str(lightest["method"])),
    }


def dataset_leader_rows(profiles_df: pd.DataFrame, include_joint: bool) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for dataset in dataset_options(profiles_df):
        subset = filter_profiles(profiles_df, dataset=dataset, include_joint=include_joint)
        if subset.empty:
            continue
        leader = subset.sort_values(["avg_accuracy_mean", "forgetting_mean"], ascending=[False, True]).iloc[0]
        cluster = subset[subset["top_cluster_flag"] == True].copy()
        if cluster.empty:
            cluster = subset[subset["method"].astype(str) == str(leader["method"])].copy()
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(str(dataset), str(dataset)),
                "best_method": str(leader["method"]),
                "best_method_label": METHOD_LABELS.get(str(leader["method"]), str(leader["method"])),
                "best_avg_accuracy_mean": float(leader["avg_accuracy_mean"]),
                "best_forgetting_mean": float(leader["forgetting_mean"]),
                "best_runtime_hours_mean": float(leader["runtime_hours_mean"]),
                "best_estimated_memory_mb": float(leader["estimated_memory_mb"]),
                "top_cluster_methods": "|".join(cluster["method"].astype(str).tolist()),
                "top_cluster_methods_label": "|".join(
                    METHOD_LABELS.get(str(item), str(item)) for item in cluster["method"].astype(str).tolist()
                ),
                "top_cluster_size": int(len(cluster)),
            }
        )
    return pd.DataFrame(rows)


def case_study_options(cases_df: pd.DataFrame, dataset: str) -> List[int]:
    if cases_df.empty:
        return []
    subset = cases_df[cases_df["dataset"].astype(str) == str(dataset)]
    return sorted(pd.to_numeric(subset["case_id"], errors="coerce").dropna().astype(int).unique().tolist())


def filter_ablation_family(current_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if current_df.empty:
        return current_df
    return current_df[current_df["dataset"].astype(str) == str(dataset)].copy()


def artifact_status_rows(bundle: DashboardBundle) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for name, path in PRIMARY_ARTIFACTS.items():
        rows.append({"group": "primary", "artifact": ARTIFACT_LABELS.get(name, name), "status": "present" if path.exists() else "missing"})
    for name, path in SECONDARY_ARTIFACTS.items():
        rows.append({"group": "secondary", "artifact": ARTIFACT_LABELS.get(name, name), "status": "present" if path.exists() else "missing"})
    return pd.DataFrame(rows)


def available_ablation_datasets(bundle: DashboardBundle) -> List[str]:
    frames = [bundle.ablation_current, bundle.ablation_runtime, bundle.ablation_memory, bundle.ablation_robustness]
    values: List[str] = []
    for frame in frames:
        if frame.empty or "dataset" not in frame.columns:
            continue
        for dataset in frame["dataset"].dropna().astype(str).tolist():
            if dataset not in values:
                values.append(dataset)
    ordered = [item for item in DATASET_ORDER if item in values]
    ordered.extend([item for item in values if item not in ordered])
    return ordered


def artifact_library_entries(bundle: DashboardBundle) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for key, path in PRIMARY_ARTIFACTS.items():
        entries.append({"key": key, "label": ARTIFACT_LABELS.get(key, key), "group": "Study Evidence", "path": path})
    for key, path in SECONDARY_ARTIFACTS.items():
        entries.append({"key": key, "label": ARTIFACT_LABELS.get(key, key), "group": "Ablation Context", "path": path})
    return entries


def static_dataset_figures(dataset: str) -> Dict[str, Path]:
    return {
        "accuracy_vs_forgetting": FIGURE_DIR / f"{dataset}_accuracy_vs_forgetting.png",
        "accuracy_vs_runtime": FIGURE_DIR / f"{dataset}_accuracy_vs_runtime.png",
        "accuracy_vs_memory": FIGURE_DIR / f"{dataset}_accuracy_vs_memory.png",
    }
