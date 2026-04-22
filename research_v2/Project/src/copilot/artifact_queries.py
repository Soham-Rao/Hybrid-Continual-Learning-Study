"""Structured access to the finalized study artifacts for copilot retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import WORKSPACE_ROOT


ANALYSIS_ROOT = WORKSPACE_ROOT / "results" / "analysis" / "epoch_1"
ABLATION_ANALYSIS_ROOT = WORKSPACE_ROOT / "results" / "analysis" / "ablations" / "epoch_1"


@dataclass(frozen=True)
class ArtifactRow:
    source_name: str
    source_path: Path
    label: str
    title: str
    content: str
    metadata: dict[str, Any]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_primary_artifacts() -> dict[str, Any]:
    return {
        "summary": _safe_read_csv(ANALYSIS_ROOT / "paper_ready_summary.csv"),
        "leaders": _safe_read_csv(ANALYSIS_ROOT / "dataset_leaders.csv"),
        "profiles": _safe_read_csv(ANALYSIS_ROOT / "recommendation_profiles.csv"),
        "pairwise": _safe_read_csv(ANALYSIS_ROOT / "pairwise_tests.csv"),
        "effect_sizes": _safe_read_csv(ANALYSIS_ROOT / "effect_sizes.csv"),
        "cases": _safe_read_csv(ANALYSIS_ROOT / "recommendation_cases.csv"),
        "report_text": (ANALYSIS_ROOT / "phase7_report.md").read_text(encoding="utf-8")
        if (ANALYSIS_ROOT / "phase7_report.md").exists()
        else "",
        "recommendation_notes": (ANALYSIS_ROOT / "phase8_recommendation_notes.md").read_text(encoding="utf-8")
        if (ANALYSIS_ROOT / "phase8_recommendation_notes.md").exists()
        else "",
    }


@lru_cache(maxsize=1)
def load_ablation_artifacts() -> dict[str, Any]:
    return {
        "current_results": _safe_read_csv(ABLATION_ANALYSIS_ROOT / "current_results.csv"),
        "runtime_summary": _safe_read_csv(ABLATION_ANALYSIS_ROOT / "runtime_sensitivity_summary.csv"),
        "memory_summary": _safe_read_csv(ABLATION_ANALYSIS_ROOT / "memory_sensitivity_summary.csv"),
        "robustness_summary": _safe_read_csv(ABLATION_ANALYSIS_ROOT / "robustness_summary.csv"),
        "notes_text": (ABLATION_ANALYSIS_ROOT / "resource_summary_notes.md").read_text(encoding="utf-8")
        if (ABLATION_ANALYSIS_ROOT / "resource_summary_notes.md").exists()
        else "",
    }


def dataset_method_profile_rows(dataset: str, method: str | None = None, limit: int = 5) -> list[ArtifactRow]:
    artifacts = load_primary_artifacts()
    df = artifacts["profiles"]
    if df.empty:
        return []
    subset = df[df["dataset"].astype(str) == str(dataset)].copy()
    if method is not None:
        subset = subset[subset["method"].astype(str) == str(method)].copy()
    subset = subset.head(limit)
    rows: list[ArtifactRow] = []
    for row in subset.itertuples(index=False):
        rows.append(
            ArtifactRow(
                source_name="recommendation_profiles",
                source_path=ANALYSIS_ROOT / "recommendation_profiles.csv",
                label="empirical_result",
                title=f"Profile for {row.dataset} / {row.method}",
                content=(
                    f"Method `{row.method}` on `{row.dataset}` has mean accuracy {row.avg_accuracy_mean:.4f}, "
                    f"mean forgetting {row.forgetting_mean:.4f}, runtime {row.runtime_hours_mean:.4f} hours, "
                    f"estimated memory {row.estimated_memory_mb:.4f} MB, family `{row.method_family}`."
                ),
                metadata={
                    "dataset": str(row.dataset),
                    "method": str(row.method),
                    "leader_flag": bool(row.leader_flag),
                    "top_cluster_flag": bool(row.top_cluster_flag),
                    "source_type": "structured_artifact",
                },
            )
        )
    return rows


def dataset_leader_row(dataset: str) -> ArtifactRow | None:
    artifacts = load_primary_artifacts()
    df = artifacts["leaders"]
    if df.empty:
        return None
    subset = df[df["dataset"].astype(str) == str(dataset)]
    if subset.empty:
        return None
    row = subset.iloc[0]
    return ArtifactRow(
        source_name="dataset_leaders",
        source_path=ANALYSIS_ROOT / "dataset_leaders.csv",
        label="empirical_result",
        title=f"Leader summary for {dataset}",
        content=(
            f"Dataset `{dataset}` is currently led by `{row['best_method']}` with mean accuracy "
            f"{float(row['best_avg_accuracy_mean']):.4f}, mean forgetting {float(row['best_forgetting_mean']):.4f}, "
            f"runtime {float(row['best_runtime_hours_mean']):.4f} hours, and "
            f"{int(row['top_cluster_size'])} methods in the top non-inferior cluster."
        ),
        metadata={
            "dataset": str(dataset),
            "best_method": str(row["best_method"]),
            "top_cluster_methods": str(row["top_cluster_methods"]),
            "source_type": "structured_artifact",
        },
    )


def recommendation_case_rows(dataset: str, limit: int = 3) -> list[ArtifactRow]:
    artifacts = load_primary_artifacts()
    df = artifacts["cases"]
    if df.empty:
        return []
    subset = df[df["dataset"].astype(str) == str(dataset)].head(limit)
    rows: list[ArtifactRow] = []
    for row in subset.itertuples(index=False):
        rows.append(
            ArtifactRow(
                source_name="recommendation_cases",
                source_path=ANALYSIS_ROOT / "recommendation_cases.csv",
                label="empirical_result",
                title=f"Recommendation case {row.case_id} for {row.dataset}",
                content=(
                    f"Case {row.case_id} ranked `{row.method}` at position {int(row.rank)} with score {float(row.score):.4f}; "
                    f"recommended winner was `{row.recommended_method}` under memory {float(row.memory_budget_mb):.1f} MB, "
                    f"compute `{row.compute_budget}`, acceptable forgetting {float(row.acceptable_forgetting):.1f}, "
                    f"task similarity `{row.task_similarity}`, joint allowed={bool(row.joint_retraining_allowed)}."
                ),
                metadata={
                    "dataset": str(row.dataset),
                    "method": str(row.method),
                    "recommended_method": str(row.recommended_method),
                    "rank": int(row.rank),
                    "source_type": "structured_artifact",
                },
            )
        )
    return rows


def report_snippets(query: str, limit: int = 3) -> list[ArtifactRow]:
    artifacts = load_primary_artifacts()
    report_text = str(artifacts["report_text"] or "")
    if not report_text.strip():
        return []
    snippets: list[ArtifactRow] = []
    lowered = query.lower()
    for block in report_text.split("\n## "):
        block_text = block.strip()
        if not block_text:
            continue
        if lowered and lowered not in block_text.lower():
            continue
        header = block_text.splitlines()[0]
        content = "\n".join(block_text.splitlines()[1:]).strip()
        snippets.append(
            ArtifactRow(
                source_name="phase7_report",
                source_path=ANALYSIS_ROOT / "phase7_report.md",
                label="empirical_result",
                title=f"Report section: {header}",
                content=content[:1200],
                metadata={"query": query, "source_type": "markdown_report"},
            )
        )
        if len(snippets) >= limit:
            break
    return snippets


def ablation_rows(dataset: str, limit: int = 4) -> list[ArtifactRow]:
    artifacts = load_ablation_artifacts()
    rows: list[ArtifactRow] = []
    for key, label in [
        ("runtime_summary", "runtime sensitivity"),
        ("memory_summary", "memory sensitivity"),
        ("robustness_summary", "robustness"),
    ]:
        df = artifacts[key]
        if df.empty:
            continue
        dataset_columns = [col for col in df.columns if "dataset" in str(col).lower()]
        subset = df
        if dataset_columns:
            col = dataset_columns[0]
            subset = df[df[col].astype(str) == str(dataset)]
        if subset.empty:
            continue
        sample = subset.head(1).to_dict(orient="records")[0]
        rows.append(
            ArtifactRow(
                source_name=key,
                source_path=ABLATION_ANALYSIS_ROOT / f"{key.replace('_summary', '_summary')}.csv",
                label="empirical_result",
                title=f"Ablation {label} context for {dataset}",
                content=str(sample),
                metadata={"dataset": dataset, "source_type": "ablation_summary"},
            )
        )
        if len(rows) >= limit:
            break
    return rows
