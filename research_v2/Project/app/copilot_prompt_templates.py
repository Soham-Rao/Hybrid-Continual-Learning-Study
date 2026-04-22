"""Prompt-template helpers for the dashboard copilot."""

from __future__ import annotations

from dataclasses import dataclass

from app.dashboard_data import DATASET_LABELS
from src.recommendation.engine import RecommendationRequest


@dataclass(frozen=True)
class PromptTemplate:
    key: str
    label: str
    prompt: str
    help_text: str


def build_prompt_templates(request: RecommendationRequest, recommended_method: str | None = None) -> list[PromptTemplate]:
    dataset_label = DATASET_LABELS.get(str(request.dataset), str(request.dataset))
    winner = recommended_method or "the recommended method"
    return [
        PromptTemplate(
            key="why_here",
            label="Why this fits",
            prompt=f"Explain why {winner} fits my current settings on {dataset_label}.",
            help_text="Best for a grounded explanation of the current recommendation.",
        ),
        PromptTemplate(
            key="compare_top3",
            label="Compare top choices",
            prompt=f"Compare the top recommendation with the next two alternatives for {dataset_label}.",
            help_text="Useful when you want trade-offs, not just the winner.",
        ),
        PromptTemplate(
            key="hardware_to_settings",
            label="Infer my settings",
            prompt="I have an old laptop GPU and limited memory. Suggest settings and tell me your assumptions before applying anything.",
            help_text="Best for turning hardware and resource descriptions into dashboard settings.",
        ),
        PromptTemplate(
            key="chart_help",
            label="Explain this chart",
            prompt=f"Explain the chart I am currently looking at for {dataset_label} in simple terms.",
            help_text="Use this when you want a chart interpreted without reading the whole report.",
        ),
        PromptTemplate(
            key="tradeoff",
            label="Trade-off view",
            prompt=f"What trade-off is the dashboard showing right now for {dataset_label}, especially between forgetting, runtime, and memory?",
            help_text="Good for understanding why methods move around across the plots.",
        ),
        PromptTemplate(
            key="literature_context",
            label="Use notes/literature",
            prompt=f"Use the project notes or literature context to explain why {winner} might work under these settings.",
            help_text="Best for a more conceptual explanation grounded in project notes and literature material.",
        ),
    ]
