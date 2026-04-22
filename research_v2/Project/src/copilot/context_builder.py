"""Context assembly helpers for grounded copilot explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.recommendation.engine import RecommendationEngine, RecommendationRequest

from .method_cards import MethodCard, get_method_card
from .retrieval import CopilotRetriever, RetrievalBundle


@dataclass(frozen=True)
class RecommendationExplanationContext:
    request: RecommendationRequest
    recommended_method: str
    shortlist: tuple[dict[str, Any], ...]
    rationale: str
    method_card: MethodCard
    evidence: RetrievalBundle


class CopilotContextBuilder:
    """Build grounded context objects for explanation-oriented copilot tasks."""

    def __init__(self, retriever: CopilotRetriever | None = None) -> None:
        self.retriever = retriever or CopilotRetriever()

    def build_recommendation_context(
        self,
        profiles_df: pd.DataFrame,
        request: RecommendationRequest,
        *,
        query: str = "",
        top_k: int = 3,
    ) -> RecommendationExplanationContext:
        engine = RecommendationEngine(profiles_df)
        result = engine.recommend(request, top_k=top_k)
        recommended_method = str(result["recommended_method"])
        method_card = get_method_card(recommended_method)
        evidence = self.retriever.retrieve_for_recommendation(
            dataset=str(request.dataset),
            method=recommended_method,
            query=query or f"Explain why {recommended_method} fits {request.dataset}",
        )
        return RecommendationExplanationContext(
            request=request,
            recommended_method=recommended_method,
            shortlist=tuple(result["shortlist"]),
            rationale=str(result["rationale"]),
            method_card=method_card,
            evidence=evidence,
        )
