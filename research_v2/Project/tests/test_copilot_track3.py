from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot import CopilotSettings, OllamaClient
from src.copilot.context_builder import CopilotContextBuilder
from src.copilot.engine import CopilotEngine
from src.copilot.method_cards import get_method_card
from src.copilot.prompts import (
    build_explain_recommendation_system_prompt,
    build_explain_recommendation_user_prompt,
)
from src.recommendation.engine import RecommendationRequest


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _profiles() -> pd.DataFrame:
    return pd.read_csv(Path("research_v2/results/analysis/epoch_1/recommendation_profiles.csv"))


def test_method_cards_cover_core_methods() -> None:
    card = get_method_card("si_der")
    assert card.label == "SI-DER"
    assert "replay" in card.mechanism.lower() or "distillation" in card.mechanism.lower()


def test_context_builder_collects_shortlist_and_evidence() -> None:
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
        query="Explain this recommendation",
    )
    assert ctx.recommended_method
    assert len(ctx.shortlist) >= 1
    assert ctx.evidence.items


def test_prompts_enforce_grounding_policy() -> None:
    builder = CopilotContextBuilder()
    ctx = builder.build_recommendation_context(
        _profiles(),
        RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
        query="Why does this fit?",
    )
    system_prompt = build_explain_recommendation_system_prompt()
    user_prompt = build_explain_recommendation_user_prompt(ctx)
    assert "source of truth" in system_prompt
    assert "empirical evidence" in user_prompt.lower()
    assert ctx.recommended_method in user_prompt


def test_copilot_engine_uses_ollama_when_available() -> None:
    def transport(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return _FakeResponse({"models": [{"name": "qwen2.5:7b-instruct"}]})
        return _FakeResponse({"response": "Grounded explanation from local model."})

    engine = CopilotEngine(
        settings=CopilotSettings(),
        ollama_client=OllamaClient(CopilotSettings(), transport=transport),
    )
    result = engine.explain_recommendation(
        _profiles(),
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=128.0,
            compute_budget="low",
            acceptable_forgetting=35.0,
            task_similarity="high",
            joint_retraining_allowed=False,
        ),
        query="Explain why this was chosen",
    )
    assert result.mode == "ollama"
    assert result.used_model == "qwen2.5:7b-instruct"
    assert "Grounded explanation" in result.explanation


def test_copilot_engine_falls_back_when_ollama_is_unavailable() -> None:
    def transport(request, timeout):
        raise OSError("offline")

    engine = CopilotEngine(
        settings=CopilotSettings(),
        ollama_client=OllamaClient(CopilotSettings(), transport=transport),
    )
    result = engine.explain_recommendation(
        _profiles(),
        RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
        query="Explain recommendation",
    )
    assert result.mode == "fallback"
    assert result.used_model is None
    assert result.recommended_method in result.explanation
