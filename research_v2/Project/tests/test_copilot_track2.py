from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot.artifact_queries import dataset_leader_row, dataset_method_profile_rows, recommendation_case_rows
from src.copilot.external_sources import default_external_lookup_policy
from src.copilot.local_docs_index import build_local_docs_index, search_local_docs
from src.copilot.retrieval import CopilotRetriever


def test_artifact_queries_return_primary_study_rows() -> None:
    leader = dataset_leader_row("permuted_mnist")
    profiles = dataset_method_profile_rows("permuted_mnist", method="si_der", limit=1)
    cases = recommendation_case_rows("split_cifar10", limit=1)

    assert leader is not None
    assert leader.label == "empirical_result"
    assert profiles and profiles[0].metadata["dataset"] == "permuted_mnist"
    assert cases and cases[0].label == "empirical_result"


def test_local_docs_index_covers_v1_and_v2_and_labels_docs() -> None:
    docs = build_local_docs_index()
    assert docs
    workspaces = {doc.workspace for doc in docs}
    labels = {doc.label for doc in docs}
    assert "research_v2" in workspaces
    assert "v1_deadline_prototype" in workspaces
    assert "design_note" in labels or "literature_note" in labels


def test_search_local_docs_finds_relevant_material() -> None:
    docs = search_local_docs("catastrophic forgetting methods", limit=5)
    assert docs
    assert any(doc.label in {"design_note", "literature_note"} for doc in docs)


def test_retriever_returns_labeled_bundle_with_policy_note() -> None:
    retriever = CopilotRetriever()
    bundle = retriever.retrieve_for_recommendation(
        dataset="permuted_mnist",
        method="agem_distill",
        query="Why does this method fit moderate memory settings?",
    )
    assert bundle.items
    assert any(item.label == "empirical_result" for item in bundle.items)
    assert any(item.label in {"design_note", "literature_note"} for item in bundle.items)
    assert "External lookups are optional" in bundle.external_policy_note


def test_default_external_policy_disallows_project_claims() -> None:
    policy = default_external_lookup_policy()
    assert policy.enabled is False
    assert policy.allowed_for_project_specific_claims is False
