from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.copilot_prompt_templates import build_prompt_templates
from src.recommendation.engine import RecommendationRequest


def test_prompt_templates_are_context_aware_and_hover_ready() -> None:
    templates = build_prompt_templates(
        RecommendationRequest(
            dataset="split_cifar10",
            memory_budget_mb=128.0,
            compute_budget="low",
            acceptable_forgetting=35.0,
            task_similarity="high",
            joint_retraining_allowed=False,
        ),
        recommended_method="si_der",
    )
    assert len(templates) >= 5
    labels = {template.label for template in templates}
    assert "Why this fits" in labels
    assert any("Split CIFAR-10" in template.prompt for template in templates)
    assert all(template.help_text for template in templates)
