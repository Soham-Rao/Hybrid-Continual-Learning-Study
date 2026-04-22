"""Policies for optional external-source retrieval in the copilot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalLookupPolicy:
    enabled: bool = False
    allowed_for_literature: bool = True
    allowed_for_project_specific_claims: bool = False
    source_label: str = "external_source"
    policy_note: str = (
        "External lookups are optional and should be clearly labeled as external. "
        "They may support literature/background explanation, but current-study claims "
        "must prefer local artifacts and project documents."
    )


def default_external_lookup_policy() -> ExternalLookupPolicy:
    return ExternalLookupPolicy()
