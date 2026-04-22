"""Copilot explanation engine for grounded recommendation explanations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.recommendation.engine import RecommendationRequest

from .actions import InferredSettingsResult, infer_settings_from_text
from .config import CopilotSettings
from .context_builder import CopilotContextBuilder, RecommendationExplanationContext
from .method_cards import get_method_card
from .ollama_client import OllamaClient, OllamaError
from .prompts import build_explain_recommendation_system_prompt, build_explain_recommendation_user_prompt
from .retrieval import CopilotRetriever, EvidenceItem


@dataclass(frozen=True)
class CopilotExplanationResult:
    recommended_method: str
    summary: str
    explanation: str
    evidence_items: tuple[EvidenceItem, ...]
    used_model: str | None
    mode: str


def _fallback_explanation(ctx: RecommendationExplanationContext) -> tuple[str, str]:
    best = ctx.shortlist[0]
    alternatives = [item["method"] for item in ctx.shortlist[1:3]]
    alt_text = ", ".join(alternatives) if alternatives else "nearby alternatives"
    empirical = (
        f"In the current study artifacts, `{ctx.recommended_method}` is competitive on `{ctx.request.dataset}` with "
        f"mean accuracy {best['avg_accuracy_mean']:.2f}, forgetting {best['forgetting_mean']:.2f}, "
        f"runtime {best['runtime_hours_mean']:.2f} hours, and estimated memory {best['estimated_memory_mb']:.2f} MB."
    )
    conceptual = (
        f"Conceptually, this fits because {ctx.method_card.mechanism.lower()} "
        f"It is especially attractive when {ctx.method_card.works_well_when[0].lower()}."
    )
    comparison = (
        f"The engine placed it ahead of {alt_text} because its constraint-adjusted trade-off was stronger for the current "
        "memory, runtime, and retention target."
    )
    summary = f"{ctx.method_card.label} fits this request because it offers the strongest grounded trade-off under the current constraints."
    explanation = f"{summary}\n\nEmpirical evidence: {empirical}\n\nConceptual interpretation: {conceptual}\n\nComparison: {comparison}"
    return summary, explanation


class CopilotEngine:
    """High-level copilot service for Track 3 explanation mode."""

    def __init__(
        self,
        *,
        settings: CopilotSettings,
        ollama_client: OllamaClient,
        retriever: CopilotRetriever | None = None,
        context_builder: CopilotContextBuilder | None = None,
    ) -> None:
        self.settings = settings
        self.ollama_client = ollama_client
        self.retriever = retriever or CopilotRetriever()
        self.context_builder = context_builder or CopilotContextBuilder(self.retriever)

    def explain_recommendation(
        self,
        profiles_df: pd.DataFrame,
        request: RecommendationRequest,
        *,
        query: str = "",
    ) -> CopilotExplanationResult:
        ctx = self.context_builder.build_recommendation_context(profiles_df, request, query=query)
        try:
            generated = self.ollama_client.generate(
                build_explain_recommendation_user_prompt(ctx),
                system=build_explain_recommendation_system_prompt(),
                model=self.settings.default_model,
                temperature=0.2,
            )
            summary = f"{get_method_card(ctx.recommended_method).label} is recommended because it best matches the current empirical trade-off and constraints."
            return CopilotExplanationResult(
                recommended_method=ctx.recommended_method,
                summary=summary,
                explanation=generated.text,
                evidence_items=ctx.evidence.items,
                used_model=generated.model,
                mode="ollama",
            )
        except OllamaError:
            summary, explanation = _fallback_explanation(ctx)
            return CopilotExplanationResult(
                recommended_method=ctx.recommended_method,
                summary=summary,
                explanation=explanation,
                evidence_items=ctx.evidence.items,
                used_model=None,
                mode="fallback",
            )

    def infer_settings(
        self,
        text: str,
        *,
        current_request: RecommendationRequest | None = None,
    ) -> InferredSettingsResult:
        return infer_settings_from_text(text, current_request=current_request)
