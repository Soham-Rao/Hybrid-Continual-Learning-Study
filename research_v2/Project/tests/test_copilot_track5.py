from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot import infer_settings_from_text
from src.recommendation.engine import RecommendationRequest


def test_infer_settings_supports_broader_hardware_descriptions() -> None:
    result = infer_settings_from_text(
        "I have a GTX 1650 laptop with 16 GB RAM.",
        current_request=RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=128.0,
            compute_budget="medium",
            acceptable_forgetting=25.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    assert result.request.memory_budget_mb == 4096.0
    assert result.request.compute_budget == "low"
    assert any("heuristic" in note.lower() for note in result.scope_notes)


def test_infer_settings_marks_out_of_scope_dataset_and_model_mentions() -> None:
    result = infer_settings_from_text(
        "I want to use Fashion MNIST with a ViT backbone.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    assert result.scope_notes
    assert any("outside the exact evaluated scope" in note.lower() or "optional backbone scope" in note.lower() for note in result.scope_notes)


def test_infer_settings_returns_targeted_clarification_questions_when_ambiguous() -> None:
    result = infer_settings_from_text("I want something good.", current_request=None)
    assert result.needs_clarification is True
    assert 1 <= len(result.clarification_questions) <= 3
    assert any("benchmark" in question.lower() or "hardware budget" in question.lower() for question in result.clarification_questions)


def test_infer_settings_limits_follow_up_questions_to_high_value_gaps() -> None:
    result = infer_settings_from_text(
        "For split cifar10 with 4 GB VRAM, I care a lot about forgetting and I do not want retraining.",
        current_request=RecommendationRequest(
            dataset="split_mini_imagenet",
            memory_budget_mb=512.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=True,
        ),
    )
    assert len(result.clarification_questions) <= 2
    assert result.request.dataset == "split_cifar10"
    assert result.request.joint_retraining_allowed is False
