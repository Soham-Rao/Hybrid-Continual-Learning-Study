from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.engine import (
    RecommendationEngine,
    RecommendationRequest,
    build_recommendation_profiles,
    generate_phase8_artifacts,
)


def _toy_summary() -> pd.DataFrame:
    rows = []
    for dataset in ["permuted_mnist", "split_cifar10", "split_cifar100", "split_mini_imagenet"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "method": "joint_training",
                    "model": "slim_resnet18",
                    "n_epochs": 1,
                    "batch_size": 32,
                    "num_workers": 2,
                    "fp16": False,
                    "buffer_size": 200,
                    "seeds": 5,
                    "total_time_sec_mean": 7200.0,
                    "total_time_sec_std": 10.0,
                    "caveat_note": "",
                    "avg_accuracy_mean": 90.0,
                    "avg_accuracy_std": 1.0,
                    "forgetting_mean": 5.0,
                    "forgetting_std": 0.5,
                    "backward_transfer_mean": -1.0,
                    "backward_transfer_std": 0.1,
                    "forward_transfer_mean": 1.0,
                    "forward_transfer_std": 0.1,
                    "runtime_hours_mean": 2.0,
                    "runtime_hours_std": 0.01,
                    "estimated_memory_mb": 2048.0,
                    "method_family": "baseline",
                },
                {
                    "dataset": dataset,
                    "method": "icarl",
                    "model": "slim_resnet18",
                    "n_epochs": 1,
                    "batch_size": 32,
                    "num_workers": 2,
                    "fp16": False,
                    "buffer_size": 500,
                    "seeds": 5,
                    "total_time_sec_mean": 1800.0,
                    "total_time_sec_std": 10.0,
                    "caveat_note": "",
                    "avg_accuracy_mean": 60.0,
                    "avg_accuracy_std": 1.0,
                    "forgetting_mean": 18.0,
                    "forgetting_std": 0.5,
                    "backward_transfer_mean": -10.0,
                    "backward_transfer_std": 0.1,
                    "forward_transfer_mean": 0.5,
                    "forward_transfer_std": 0.1,
                    "runtime_hours_mean": 0.5,
                    "runtime_hours_std": 0.01,
                    "estimated_memory_mb": 6.0,
                    "method_family": "hybrid",
                },
                {
                    "dataset": dataset,
                    "method": "ewc",
                    "model": "slim_resnet18",
                    "n_epochs": 1,
                    "batch_size": 32,
                    "num_workers": 2,
                    "fp16": False,
                    "buffer_size": 200,
                    "seeds": 5,
                    "total_time_sec_mean": 2400.0,
                    "total_time_sec_std": 10.0,
                    "caveat_note": "",
                    "avg_accuracy_mean": 52.0,
                    "avg_accuracy_std": 1.0,
                    "forgetting_mean": 22.0,
                    "forgetting_std": 0.5,
                    "backward_transfer_mean": -12.0,
                    "backward_transfer_std": 0.1,
                    "forward_transfer_mean": 0.4,
                    "forward_transfer_std": 0.1,
                    "runtime_hours_mean": 0.66,
                    "runtime_hours_std": 0.01,
                    "estimated_memory_mb": 16.0,
                    "method_family": "baseline",
                },
            ]
        )
    return pd.DataFrame(rows)


def _toy_leaders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "best_method": "joint_training",
                "best_avg_accuracy_mean": 90.0,
                "best_forgetting_mean": 5.0,
                "best_runtime_hours_mean": 2.0,
                "best_estimated_memory_mb": 2048.0,
                "top_cluster_methods": "joint_training|icarl",
                "top_cluster_size": 2,
                "caveat_note": "",
            }
            for dataset in ["permuted_mnist", "split_cifar10", "split_cifar100", "split_mini_imagenet"]
        ]
    )


def test_recommendation_engine_prefers_joint_when_allowed() -> None:
    profiles = build_recommendation_profiles(_toy_summary(), _toy_leaders())
    engine = RecommendationEngine(profiles)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=4096.0,
            compute_budget="high",
            acceptable_forgetting=20.0,
            task_similarity="low",
            joint_retraining_allowed=True,
        )
    )
    assert result["recommended_method"] == "joint_training"


def test_recommendation_engine_excludes_joint_when_not_allowed() -> None:
    profiles = build_recommendation_profiles(_toy_summary(), _toy_leaders())
    engine = RecommendationEngine(profiles)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=4096.0,
            compute_budget="high",
            acceptable_forgetting=20.0,
            task_similarity="low",
            joint_retraining_allowed=False,
        )
    )
    assert result["recommended_method"] == "icarl"


def test_recommendation_engine_penalizes_memory_and_runtime() -> None:
    profiles = build_recommendation_profiles(_toy_summary(), _toy_leaders())
    engine = RecommendationEngine(profiles)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=32.0,
            compute_budget="low",
            acceptable_forgetting=35.0,
            task_similarity="high",
            joint_retraining_allowed=True,
        )
    )
    shortlist = result["shortlist"]
    assert shortlist[0]["method"] != "joint_training"
    assert "memory budget" in " ".join(shortlist[0]["reasons"])


def test_recommendation_shortlist_is_ranked_and_has_rationales() -> None:
    profiles = build_recommendation_profiles(_toy_summary(), _toy_leaders())
    engine = RecommendationEngine(profiles)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_cifar100",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=25.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        )
    )
    scores = [item["score"] for item in result["shortlist"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["reasons"] for item in result["shortlist"])


def test_non_joint_ranking_is_stable_when_joint_toggle_changes() -> None:
    profiles = build_recommendation_profiles(_toy_summary(), _toy_leaders())
    engine = RecommendationEngine(profiles)

    allowed = engine.recommend(
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=32.0,
            compute_budget="low",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=True,
        )
    )
    disallowed = engine.recommend(
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=32.0,
            compute_budget="low",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        )
    )
    assert allowed["recommended_method"] == disallowed["recommended_method"]


def test_phase8_artifact_generation_smoke(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis" / "epoch_1"
    analysis_dir.mkdir(parents=True)
    _toy_summary().to_csv(analysis_dir / "paper_ready_summary.csv", index=False)
    _toy_leaders().to_csv(analysis_dir / "dataset_leaders.csv", index=False)

    outputs = generate_phase8_artifacts(analysis_dir)
    assert len(outputs["profiles"]) == 12
    assert sorted(outputs["cases"]["case_id"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert (analysis_dir / "recommendation_profiles.csv").exists()
    assert (analysis_dir / "phase8_recommendation_notes.md").exists()
