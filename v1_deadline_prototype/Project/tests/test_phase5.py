import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.phase5 import collect_run_records
from src.recommendation.engine import RecommendationEngine, RecommendationRequest


def test_collect_run_records_finds_phase4_results() -> None:
    df = collect_run_records(Path("results"))
    assert not df.empty
    assert "split_mini_imagenet" in set(df["dataset"].astype(str))
    assert "phase4_local_mini" in set(df["source_group"].astype(str))


def test_recommendation_engine_prefers_joint_when_allowed() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "split_mini_imagenet",
                "method": "joint_training",
                "source_group": "phase4_local_mini",
                "avg_accuracy_mean": 10.6,
                "forgetting_mean": 5.0,
                "runtime_hours_mean": 2.5,
                "buffer_size_mean": 10000.0,
            },
            {
                "dataset": "split_mini_imagenet",
                "method": "icarl",
                "source_group": "phase4_local_mini",
                "avg_accuracy_mean": 3.7,
                "forgetting_mean": 11.0,
                "runtime_hours_mean": 0.5,
                "buffer_size_mean": 2000.0,
            },
        ]
    )
    engine = RecommendationEngine(summary)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=4096.0,
            compute_budget="high",
            acceptable_forgetting=20.0,
            joint_retraining_allowed=True,
        )
    )
    assert result["recommended_method"] == "joint_training"


def test_recommendation_engine_excludes_joint_when_not_allowed() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "split_mini_imagenet",
                "method": "joint_training",
                "source_group": "phase4_local_mini",
                "avg_accuracy_mean": 10.6,
                "forgetting_mean": 5.0,
                "runtime_hours_mean": 2.5,
                "buffer_size_mean": 10000.0,
            },
            {
                "dataset": "split_mini_imagenet",
                "method": "icarl",
                "source_group": "phase4_local_mini",
                "avg_accuracy_mean": 3.7,
                "forgetting_mean": 11.0,
                "runtime_hours_mean": 0.5,
                "buffer_size_mean": 2000.0,
            },
        ]
    )
    engine = RecommendationEngine(summary)
    result = engine.recommend(
        RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=4096.0,
            compute_budget="high",
            acceptable_forgetting=20.0,
            joint_retraining_allowed=False,
        )
    )
    assert result["recommended_method"] == "icarl"
