from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dashboard_data import (
    DashboardBundle,
    artifact_status_rows,
    build_effect_matrix,
    build_pairwise_matrix,
    build_rank_dataframe,
    build_top_cluster_membership,
    comparison_table,
    dataset_leader_rows,
    dataset_snapshot,
    filter_profiles,
    load_dashboard_bundle,
    strip_markdown_section,
)
from app.dashboard_charts import (
    build_cross_dataset_heatmap,
    build_grouped_metric_bars,
    build_recommendation_breakdown,
)
from app.main import main as dashboard_main
from src.recommendation.engine import RecommendationEngine, RecommendationRequest


def test_dashboard_bundle_loads_existing_v2_artifacts() -> None:
    bundle = load_dashboard_bundle()
    assert not bundle.summary.empty
    assert not bundle.recommendation_profiles.empty
    assert not bundle.leaders.empty
    assert not bundle.pairwise.empty
    assert not bundle.effect_sizes.empty


def test_pairwise_and_effect_matrices_build_for_real_dataset() -> None:
    bundle = load_dashboard_bundle()
    pairwise = build_pairwise_matrix(bundle.pairwise, "split_cifar10", "avg_accuracy", "holm_adjusted_p_value")
    effect = build_effect_matrix(bundle.effect_sizes, "split_cifar10", "avg_accuracy")
    assert pairwise.shape[0] == pairwise.shape[1]
    assert effect.shape[0] == effect.shape[1]
    assert "Joint Training" in pairwise.columns
    assert "Joint Training" in effect.columns


def test_filtered_profiles_and_snapshot_respect_joint_toggle() -> None:
    bundle = load_dashboard_bundle()
    subset = filter_profiles(bundle.recommendation_profiles, dataset="split_cifar100", include_joint=False)
    assert "joint_training" not in subset["method"].astype(str).tolist()

    snapshot = dataset_snapshot(bundle.recommendation_profiles, bundle.leaders, "split_cifar100", include_joint=False)
    assert snapshot["best_method"] != "joint_training"
    assert snapshot["top_cluster_size"] >= 1


def test_comparison_table_and_membership_use_dashboard_flags() -> None:
    bundle = load_dashboard_bundle()
    table = comparison_table(
        bundle.recommendation_profiles,
        dataset="permuted_mnist",
        families=["baseline", "hybrid"],
        include_joint=False,
        top_cluster_only=True,
        sort_by="Avg Accuracy",
    )
    assert not table.empty
    assert "Joint Training" not in table["Method"].astype(str).tolist()

    ranks = build_rank_dataframe(bundle.recommendation_profiles, include_joint=False)
    membership = build_top_cluster_membership(bundle.recommendation_profiles, include_joint=False)
    assert not ranks.empty
    assert not membership.empty


def test_dynamic_dataset_leaders_and_report_strip_follow_current_view() -> None:
    bundle = load_dashboard_bundle()
    leaders_without_joint = dataset_leader_rows(bundle.recommendation_profiles, include_joint=False)
    assert not leaders_without_joint.empty
    assert "joint_training" not in leaders_without_joint["best_method"].astype(str).tolist()

    stripped = strip_markdown_section(bundle.report_text, "## Dataset Leaders")
    assert "## Dataset Leaders" not in stripped


def test_artifact_status_rows_and_main_entrypoint_import_cleanly() -> None:
    bundle = load_dashboard_bundle()
    status = artifact_status_rows(bundle)
    assert not status.empty
    assert {"primary", "secondary"} <= set(status["group"].astype(str).unique().tolist())
    assert callable(dashboard_main)


def test_chart_builders_return_plotly_figures_for_real_artifacts() -> None:
    bundle = load_dashboard_bundle()
    subset = filter_profiles(bundle.recommendation_profiles, dataset="split_cifar10", include_joint=False)
    engine = RecommendationEngine(bundle.recommendation_profiles)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=128.0,
            compute_budget="low",
            acceptable_forgetting=35.0,
            task_similarity="high",
            joint_retraining_allowed=False,
        )
    )

    fig1 = build_cross_dataset_heatmap(bundle.recommendation_profiles, "avg_accuracy_mean")
    fig2 = build_grouped_metric_bars(subset, ["avg_accuracy_mean", "forgetting_mean"])
    fig3 = build_recommendation_breakdown(result["shortlist"][0])

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
