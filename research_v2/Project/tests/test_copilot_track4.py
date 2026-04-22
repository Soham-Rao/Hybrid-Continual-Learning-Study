from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot import CopilotEngine, CopilotSettings, OllamaClient, infer_settings_from_text
from src.recommendation.engine import RecommendationRequest


def test_infer_settings_maps_gt210_description_to_low_compute_and_1gb_budget() -> None:
    result = infer_settings_from_text(
        "I have a GT210 and I want something lightweight.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    assert result.request.memory_budget_mb == 1024.0
    assert result.request.compute_budget == "low"
    assert result.requires_confirmation is True
    assert any("GT 210" in item or "gt 210" in item.lower() for item in result.assumptions)


def test_infer_settings_preserves_current_values_when_text_is_vague() -> None:
    current = RecommendationRequest(
        dataset="split_cifar10",
        memory_budget_mb=128.0,
        compute_budget="medium",
        acceptable_forgetting=25.0,
        task_similarity="medium",
        joint_retraining_allowed=False,
    )
    result = infer_settings_from_text("I want a good option.", current_request=current)
    assert result.request == current
    assert len(result.assumptions) >= 3


def test_infer_settings_reads_dataset_retraining_and_retention_language() -> None:
    result = infer_settings_from_text(
        "For split cifar100, I can retrain overnight and retention matters a lot.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    assert result.request.dataset == "split_cifar100"
    assert result.request.joint_retraining_allowed is True
    assert result.request.acceptable_forgetting == 10.0


def test_copilot_engine_exposes_infer_settings_api() -> None:
    engine = CopilotEngine(
        settings=CopilotSettings(),
        ollama_client=OllamaClient(CopilotSettings(), transport=lambda request, timeout: (_ for _ in ()).throw(OSError("offline"))),
    )
    result = engine.infer_settings(
        "I have an old laptop and I don't want retraining.",
        current_request=RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=512.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=True,
        ),
    )
    assert result.request.compute_budget == "low"
    assert result.request.joint_retraining_allowed is False
    assert result.mode == "heuristic"
