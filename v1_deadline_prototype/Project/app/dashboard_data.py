"""Data loading helpers for the Phase 6 Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
PHASE5_ROOT = REPO_ROOT / "results" / "analysis" / "phase5"
FIGURES_ROOT = PHASE5_ROOT / "figures"


REQUIRED_ARTIFACTS = {
    "summary": PHASE5_ROOT / "paper_ready_summary.csv",
    "cases": PHASE5_ROOT / "recommendation_cases.csv",
    "pareto": PHASE5_ROOT / "pareto_frontier_candidates.csv",
    "report": PHASE5_ROOT / "phase5_report.md",
}


GLOBAL_FIGURES = {
    "avg_accuracy_heatmap": FIGURES_ROOT / "phase5_avg_accuracy_heatmap.png",
    "forgetting_heatmap": FIGURES_ROOT / "phase5_forgetting_heatmap.png",
}


def dataset_figure_map(dataset: str) -> Dict[str, Path]:
    return {
        "accuracy_vs_buffer": FIGURES_ROOT / f"{dataset}_accuracy_vs_buffer.png",
        "accuracy_vs_runtime": FIGURES_ROOT / f"{dataset}_accuracy_vs_runtime.png",
    }


def missing_artifacts() -> List[str]:
    missing = [str(path) for path in REQUIRED_ARTIFACTS.values() if not path.exists()]
    for path in GLOBAL_FIGURES.values():
        if not path.exists():
            missing.append(str(path))
    return missing


def available_sections() -> Dict[str, bool]:
    return {name: path.exists() for name, path in REQUIRED_ARTIFACTS.items()}


def load_summary() -> pd.DataFrame:
    path = REQUIRED_ARTIFACTS["summary"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_cases() -> pd.DataFrame:
    path = REQUIRED_ARTIFACTS["cases"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_pareto() -> pd.DataFrame:
    path = REQUIRED_ARTIFACTS["pareto"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_report_text() -> str:
    path = REQUIRED_ARTIFACTS["report"]
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def datasets_from_summary(summary_df: pd.DataFrame) -> List[str]:
    if summary_df.empty or "dataset" not in summary_df.columns:
        return []
    values = summary_df["dataset"].dropna().astype(str).unique().tolist()
    preferred_order = [
        "split_mini_imagenet",
        "split_cifar100",
        "split_cifar10",
        "permuted_mnist",
    ]
    ordered = [item for item in preferred_order if item in values]
    ordered.extend([item for item in values if item not in ordered])
    return ordered


def primary_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    if "is_primary_run" not in summary_df.columns:
        return summary_df
    return summary_df[summary_df["is_primary_run"] == True].copy()  # noqa: E712


def comparison_table(
    summary_df: pd.DataFrame,
    dataset: str,
    primary_only: bool = True,
) -> pd.DataFrame:
    df = primary_summary(summary_df) if primary_only else summary_df.copy()
    if df.empty:
        return df

    subset = df[df["dataset"].astype(str) == dataset].copy()
    if subset.empty:
        return subset

    display = subset[
        [
            "method",
            "method_variant",
            "source_group",
            "model",
            "seeds",
            "avg_accuracy_mean",
            "forgetting_mean",
            "backward_transfer_mean",
            "forward_transfer_mean",
            "runtime_hours_mean",
            "buffer_size_mean",
            "data_quality_note",
        ]
    ].copy()
    display = display.rename(
        columns={
            "method": "Method",
            "method_variant": "Variant",
            "source_group": "Source",
            "model": "Model",
            "seeds": "Seeds",
            "avg_accuracy_mean": "Avg Accuracy",
            "forgetting_mean": "Forgetting",
            "backward_transfer_mean": "BWT",
            "forward_transfer_mean": "FWT",
            "runtime_hours_mean": "Runtime (h)",
            "buffer_size_mean": "Buffer Size",
            "data_quality_note": "Notes",
        }
    )
    return display.sort_values(["Avg Accuracy", "Forgetting"], ascending=[False, True]).reset_index(drop=True)


def top_findings(summary_df: pd.DataFrame) -> List[Dict[str, str]]:
    df = primary_summary(summary_df)
    if df.empty:
        return []

    findings: List[Dict[str, str]] = []
    joint_rows = df[df["method"] == "joint_training"].sort_values("avg_accuracy_mean", ascending=False)
    if not joint_rows.empty:
        row = joint_rows.iloc[0]
        findings.append(
            {
                "title": "Upper Bound",
                "body": f"{row['dataset']}: joint_training reaches {row['avg_accuracy_mean']:.2f}% average accuracy.",
            }
        )

    mini_icarl = df[
        (df["dataset"].astype(str) == "split_mini_imagenet") & (df["method"] == "icarl")
    ]
    if not mini_icarl.empty:
        row = mini_icarl.iloc[0]
        findings.append(
            {
                "title": "Mini-ImageNet Pick",
                "body": f"iCaRL is the strongest non-joint option at {row['avg_accuracy_mean']:.2f}% AA with {row['forgetting_mean']:.2f} forgetting.",
            }
        )

    notes = caution_notes(df)
    if notes:
        findings.append(
            {
                "title": "Caution",
                "body": notes[0],
            }
        )
    return findings


def caution_notes(summary_df: pd.DataFrame) -> List[str]:
    if summary_df.empty or "data_quality_note" not in summary_df.columns:
        return []
    notes = [
        note.strip()
        for note in summary_df["data_quality_note"].fillna("").astype(str).tolist()
        if note.strip()
    ]
    deduped: List[str] = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return deduped

