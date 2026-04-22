from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot import CopilotSettings, OllamaClient
from src.copilot.actions import infer_settings_from_text
from src.copilot.engine import CopilotEngine
from src.copilot.policies import build_safety_envelope
from src.copilot.retrieval import EvidenceItem
from src.recommendation.engine import RecommendationRequest


def _profiles() -> pd.DataFrame:
    return pd.read_csv(Path("research_v2/results/analysis/epoch_1/recommendation_profiles.csv"))


def test_safety_envelope_labels_local_evidence_and_uncertainty() -> None:
    items = (
        EvidenceItem(
            label="empirical_result",
            title="Profile for split_cifar10 / si_der",
            content="Mean accuracy 80.0 and forgetting 10.0.",
            source_name="recommendation_profiles",
            source_kind="structured_artifact",
            metadata={},
        ),
        EvidenceItem(
            label="literature_note",
            title="Method note",
            content="Replay hybrids often improve retention.",
            source_name="notes.md",
            source_kind="local_document",
            metadata={},
        ),
    )
    envelope = build_safety_envelope(
        "This method is competitive and conceptually stable.",
        items,
        external_policy_note="External lookup is disabled by default.",
    )
    assert "deterministic recommendation engine" in envelope.recommendation_source_note.lower()
    assert "local study evidence" in envelope.source_disclosure.lower()
    assert "uncertainty" in envelope.uncertainty_note.lower()
    assert envelope.evidence_snippets


def test_safety_envelope_flags_unsupported_significance_language() -> None:
    items = (
        EvidenceItem(
            label="empirical_result",
            title="Profile for permuted_mnist / xder",
            content="Mean accuracy 97.0 and forgetting 3.0.",
            source_name="recommendation_profiles",
            source_kind="structured_artifact",
            metadata={},
        ),
    )
    envelope = build_safety_envelope(
        "This method is statistically significant and proves the best trade-off.",
        items,
        external_policy_note="External lookup is disabled by default.",
    )
    assert envelope.claim_guardrail_note is not None
    assert "descriptive only" in envelope.claim_guardrail_note.lower()


def test_copilot_engine_returns_safety_fields() -> None:
    def transport(request, timeout):
        raise OSError("offline")

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
        query="Explain recommendation safely",
    )
    assert result.recommendation_source_note
    assert "evidence basis" in result.source_disclosure.lower()
    assert "uncertainty" in result.uncertainty_note.lower()
    assert result.evidence_snippets


def test_inferred_settings_scope_note_stays_explicit_for_out_of_scope_requests() -> None:
    result = infer_settings_from_text(
        "I want Fashion MNIST on a ViT with a GTX 1050.",
        current_request=RecommendationRequest(
            dataset="permuted_mnist",
            memory_budget_mb=256.0,
            compute_budget="medium",
            acceptable_forgetting=20.0,
            task_similarity="medium",
            joint_retraining_allowed=False,
        ),
    )
    assert any("outside the exact evaluated scope" in note.lower() for note in result.scope_notes)
    assert any("heuristic" in note.lower() for note in result.scope_notes)
