from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.copilot_ui import _chart_focus_from_text, _format_inferred_settings, infer_copilot_intent, init_copilot_state
from src.copilot import infer_settings_from_text
from src.recommendation.engine import RecommendationRequest


def test_infer_copilot_intent_routes_settings_like_prompts() -> None:
    assert infer_copilot_intent("I have 4 GB VRAM and do not want retraining.") == "infer_settings"
    assert infer_copilot_intent("i have a gt210 what do you think i should do") == "infer_settings"
    assert infer_copilot_intent("Why was this method recommended?") == "explain_recommendation"
    assert infer_copilot_intent("What trade-off is this chart showing right now?") == "interpret_chart"


def test_chart_focus_detects_named_tradeoff_views() -> None:
    assert _chart_focus_from_text("explain accuracy vs forgetting chart") == "accuracy_forgetting"
    assert _chart_focus_from_text("explain accuracy vs estimated memory chart") == "accuracy_memory"
    assert _chart_focus_from_text("show me the score breakdown") == "score_breakdown"


def test_format_inferred_settings_includes_assumptions_and_scope_notes() -> None:
    result = infer_settings_from_text(
        "I have a GTX 1650 and want to use Fashion MNIST with a ViT.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    text = _format_inferred_settings(result)
    assert "Assumptions:" in text
    assert "Scope notes:" in text
    assert "Apply inferred settings" in text


def test_init_copilot_state_preserves_cross_tab_state_shape() -> None:
    st.session_state.clear()
    init_copilot_state()
    assert "copilot_open" in st.session_state
    assert "copilot_messages" in st.session_state
    assert isinstance(st.session_state["copilot_messages"], list)
