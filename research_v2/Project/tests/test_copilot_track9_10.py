from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.copilot_ui import _apply_inferred_settings
from src.copilot.actions import infer_settings_from_text
from src.copilot.context_builder import CopilotContextBuilder
from src.copilot.knowledge_base import ChartExplanationFacts, chart_explanation_draft, recommendation_explanation_draft
from src.recommendation.engine import RecommendationRequest


def _profiles() -> pd.DataFrame:
    return pd.read_csv(Path("research_v2/results/analysis/epoch_1/recommendation_profiles.csv"))


def test_apply_inferred_settings_defers_dashboard_mutation_until_sidebar_phase() -> None:
    st.session_state.clear()
    result = infer_settings_from_text(
        "I have a GT210.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    st.session_state["copilot_last_inferred"] = result
    st.session_state["dashboard_dataset"] = "split_mini_imagenet"
    _apply_inferred_settings()
    assert st.session_state["dashboard_dataset"] == "split_mini_imagenet"
    pending = st.session_state.get("copilot_pending_dashboard_request")
    assert pending is not None
    assert pending.dataset == "permuted_mnist"
    assert pending.compute_budget == "low"


def test_recommendation_explanation_draft_stays_grounded_to_current_context() -> None:
    builder = CopilotContextBuilder()
    ctx = builder.build_recommendation_context(
        _profiles(),
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=128.0,
            compute_budget="low",
            acceptable_forgetting=35.0,
            task_similarity="high",
            joint_retraining_allowed=False,
        ),
        query="Explain recommendation clearly",
    )
    summary, explanation = recommendation_explanation_draft(ctx)
    assert ctx.recommended_method in summary or ctx.method_card.label in summary
    assert "deterministic recommendation engine" in explanation.lower()
    assert f"mean accuracy {ctx.shortlist[0]['avg_accuracy_mean']:.2f}" in explanation
    assert ctx.method_card.mechanism.split(".")[0] in explanation


def test_chart_explanation_draft_uses_chart_specific_template() -> None:
    facts = ChartExplanationFacts(
        chart_focus="accuracy_forgetting",
        dataset="permuted_mnist",
        winner={"method": "si_der", "avg_accuracy_mean": 70.86, "forgetting_mean": 29.26},
        best_accuracy={"method": "joint_training", "avg_accuracy_mean": 95.37},
        lowest_forgetting={"method": "joint_training", "forgetting_mean": 1.19},
        fastest={"method": "lwf", "runtime_hours_mean": 0.15},
        smallest_memory={"method": "agem", "estimated_memory_mb": 0.8},
        shortlist_summary="xder (acc 71.17, forget 28.97), der (acc 70.96, forget 29.24)",
    )
    text = chart_explanation_draft(facts)
    assert "accuracy-vs-forgetting chart" in text
    assert "highest-accuracy method" in text
    assert "lowest-forgetting method" in text
    assert "xder" in text and "der" in text
