"""Safety, grounding, and claim-control helpers for the copilot."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import EvidenceItem


_CLAIM_TERMS = (
    "statistically significant",
    "significant",
    "p-value",
    "effect size",
    "proves",
    "state of the art",
    "sota",
)

_BACKED_STAT_TERMS = (
    "pairwise",
    "holm",
    "p-value",
    "effect size",
    "wilcoxon",
    "friedman",
)


@dataclass(frozen=True)
class SourceDisclosure:
    empirical_count: int
    local_document_count: int
    external_count: int
    summary: str


@dataclass(frozen=True)
class SafetyEnvelope:
    recommendation_source_note: str
    source_disclosure: str
    uncertainty_note: str
    claim_guardrail_note: str | None
    evidence_snippets: tuple[str, ...]


def _looks_empirical(item: EvidenceItem) -> bool:
    return item.label == "empirical_result" or item.source_kind in {
        "structured_artifact",
        "markdown_report",
        "ablation_summary",
    }


def _looks_external(item: EvidenceItem) -> bool:
    return item.label == "external_source" or item.source_kind == "external_source"


def build_source_disclosure(items: tuple[EvidenceItem, ...], *, external_policy_note: str) -> SourceDisclosure:
    empirical_count = sum(1 for item in items if _looks_empirical(item))
    external_count = sum(1 for item in items if _looks_external(item))
    local_document_count = max(0, len(items) - empirical_count - external_count)

    parts: list[str] = []
    if empirical_count:
        parts.append(f"{empirical_count} local study evidence item{'s' if empirical_count != 1 else ''}")
    if local_document_count:
        parts.append(f"{local_document_count} local note/document item{'s' if local_document_count != 1 else ''}")
    if external_count:
        parts.append(f"{external_count} external source item{'s' if external_count != 1 else ''}")
    if not parts:
        parts.append("no retrieved evidence items")

    summary = f"Evidence basis: {', '.join(parts)}."
    if external_count:
        summary += " External material was included and should be read as supporting background rather than study evidence."
    else:
        summary += f" No external internet material was used for this answer. {external_policy_note}".strip()
    return SourceDisclosure(
        empirical_count=empirical_count,
        local_document_count=local_document_count,
        external_count=external_count,
        summary=summary,
    )


def build_uncertainty_note(items: tuple[EvidenceItem, ...]) -> str:
    empirical_count = sum(1 for item in items if _looks_empirical(item))
    external_count = sum(1 for item in items if _looks_external(item))
    local_document_count = max(0, len(items) - empirical_count - external_count)

    if empirical_count == 0 and local_document_count == 0:
        return "Uncertainty: this answer is lightly grounded because no local study evidence was retrieved."
    if empirical_count == 0:
        return "Uncertainty: this answer relies on local notes or background material rather than direct study results."
    if empirical_count <= 2 and local_document_count > empirical_count:
        return "Uncertainty: the explanation is partially grounded in the study, but much of the detail comes from local notes and method descriptions."
    if empirical_count <= 2:
        return "Uncertainty: only a small amount of direct study evidence was retrieved, so treat the explanation as directional rather than exhaustive."
    return "Uncertainty: the recommendation itself comes from the deterministic engine, while the surrounding explanation mixes retrieved evidence with reasoned interpretation."


def build_claim_guardrail_note(explanation: str, items: tuple[EvidenceItem, ...]) -> str | None:
    lowered = explanation.lower()
    if not any(term in lowered for term in _CLAIM_TERMS):
        return None

    evidence_text = " ".join(f"{item.title} {item.content}" for item in items).lower()
    if any(term in evidence_text for term in _BACKED_STAT_TERMS):
        return None

    return (
        "Claim guardrail: treat any wording that sounds statistically significant or literature-authoritative as descriptive only; "
        "no dedicated significance or external-literature evidence was retrieved for this answer."
    )


def build_evidence_snippets(items: tuple[EvidenceItem, ...], *, max_items: int = 3, max_chars: int = 180) -> tuple[str, ...]:
    snippets: list[str] = []
    for item in items[:max_items]:
        source_label = item.label.replace("_", " ").title()
        excerpt = " ".join(item.content.strip().split())
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 3].rstrip() + "..."
        snippets.append(f"[{source_label}] {item.title}: {excerpt}")
    return tuple(snippets)


def build_safety_envelope(
    explanation: str,
    items: tuple[EvidenceItem, ...],
    *,
    external_policy_note: str,
) -> SafetyEnvelope:
    disclosure = build_source_disclosure(items, external_policy_note=external_policy_note)
    return SafetyEnvelope(
        recommendation_source_note=(
            "Recommendation source: the deterministic recommendation engine chose the method; the copilot is only explaining that choice."
        ),
        source_disclosure=disclosure.summary,
        uncertainty_note=build_uncertainty_note(items),
        claim_guardrail_note=build_claim_guardrail_note(explanation, items),
        evidence_snippets=build_evidence_snippets(items),
    )
