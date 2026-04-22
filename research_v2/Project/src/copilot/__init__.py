"""Copilot foundation modules."""

from .actions import InferredSettingsResult, infer_settings_from_text
from .config import CopilotSettings, load_copilot_settings
from .context_builder import CopilotContextBuilder, RecommendationExplanationContext
from .engine import CopilotEngine, CopilotExplanationResult
from .external_sources import ExternalLookupPolicy, default_external_lookup_policy
from .method_cards import MethodCard, get_method_card
from .ollama_client import OllamaClient, OllamaGenerateResult, OllamaStatus
from .prompts import build_explain_recommendation_system_prompt, build_explain_recommendation_user_prompt
from .retrieval import CopilotRetriever, EvidenceItem, RetrievalBundle

__all__ = [
    "CopilotSettings",
    "CopilotContextBuilder",
    "CopilotEngine",
    "CopilotExplanationResult",
    "CopilotRetriever",
    "EvidenceItem",
    "ExternalLookupPolicy",
    "InferredSettingsResult",
    "MethodCard",
    "OllamaClient",
    "OllamaGenerateResult",
    "OllamaStatus",
    "RecommendationExplanationContext",
    "RetrievalBundle",
    "build_explain_recommendation_system_prompt",
    "build_explain_recommendation_user_prompt",
    "default_external_lookup_policy",
    "get_method_card",
    "infer_settings_from_text",
    "load_copilot_settings",
]
