"""Grounded evidence retrieval for the dashboard copilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifact_queries import (
    ArtifactRow,
    ablation_rows,
    dataset_leader_row,
    dataset_method_profile_rows,
    recommendation_case_rows,
    report_snippets,
)
from .external_sources import ExternalLookupPolicy, default_external_lookup_policy
from .local_docs_index import search_local_docs


@dataclass(frozen=True)
class EvidenceItem:
    label: str
    title: str
    content: str
    source_name: str
    source_kind: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalBundle:
    query: str
    items: tuple[EvidenceItem, ...]
    external_policy_note: str


def _artifact_to_item(row: ArtifactRow) -> EvidenceItem:
    return EvidenceItem(
        label=row.label,
        title=row.title,
        content=row.content,
        source_name=row.source_name,
        source_kind=str(row.metadata.get("source_type", "artifact")),
        metadata={**row.metadata, "source_path": str(row.source_path)},
    )


class CopilotRetriever:
    """Compose structured artifacts, local docs, and external-source policy."""

    def __init__(self, external_policy: ExternalLookupPolicy | None = None) -> None:
        self.external_policy = external_policy or default_external_lookup_policy()

    def retrieve_for_recommendation(
        self,
        *,
        dataset: str,
        method: str,
        query: str,
        include_ablation_context: bool = True,
        include_local_docs: bool = True,
        local_doc_limit: int = 3,
    ) -> RetrievalBundle:
        items: list[EvidenceItem] = []

        leader = dataset_leader_row(dataset)
        if leader is not None:
            items.append(_artifact_to_item(leader))

        items.extend(_artifact_to_item(row) for row in dataset_method_profile_rows(dataset, method=method, limit=2))
        items.extend(_artifact_to_item(row) for row in recommendation_case_rows(dataset, limit=2))
        items.extend(_artifact_to_item(row) for row in report_snippets(dataset, limit=2))

        if include_ablation_context:
            items.extend(_artifact_to_item(row) for row in ablation_rows(dataset, limit=3))

        if include_local_docs:
            doc_query = f"{dataset} {method} {query}".strip()
            for doc in search_local_docs(doc_query, limit=local_doc_limit):
                items.append(
                    EvidenceItem(
                        label=doc.label,
                        title=doc.title,
                        content=doc.text[:1200],
                        source_name=doc.path.name,
                        source_kind="local_document",
                        metadata={
                            "workspace": doc.workspace,
                            "source_path": str(doc.path),
                        },
                    )
                )

        return RetrievalBundle(
            query=query,
            items=tuple(items),
            external_policy_note=self.external_policy.policy_note,
        )

    def retrieve_for_query(
        self,
        query: str,
        *,
        dataset: str | None = None,
        method: str | None = None,
        limit_docs: int = 5,
    ) -> RetrievalBundle:
        items: list[EvidenceItem] = []
        if dataset and method:
            base = self.retrieve_for_recommendation(
                dataset=dataset,
                method=method,
                query=query,
                include_ablation_context=True,
                include_local_docs=False,
            )
            items.extend(base.items)
        for doc in search_local_docs(" ".join(part for part in [query, dataset or "", method or ""] if part), limit=limit_docs):
            items.append(
                EvidenceItem(
                    label=doc.label,
                    title=doc.title,
                    content=doc.text[:1200],
                    source_name=doc.path.name,
                    source_kind="local_document",
                    metadata={"workspace": doc.workspace, "source_path": str(doc.path)},
                )
            )
        return RetrievalBundle(query=query, items=tuple(items), external_policy_note=self.external_policy.policy_note)
