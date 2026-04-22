"""Copilot foundation modules."""

from .config import CopilotSettings, load_copilot_settings
from .external_sources import ExternalLookupPolicy, default_external_lookup_policy
from .ollama_client import OllamaClient, OllamaGenerateResult, OllamaStatus
from .retrieval import CopilotRetriever, EvidenceItem, RetrievalBundle

__all__ = [
    "CopilotSettings",
    "CopilotRetriever",
    "EvidenceItem",
    "ExternalLookupPolicy",
    "OllamaClient",
    "OllamaGenerateResult",
    "OllamaStatus",
    "RetrievalBundle",
    "default_external_lookup_policy",
    "load_copilot_settings",
]
