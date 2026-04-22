"""Prompt builders for the grounded dashboard copilot."""

from __future__ import annotations

from .context_builder import RecommendationExplanationContext


def build_explain_recommendation_system_prompt() -> str:
    return (
        "You are a grounded research dashboard copilot. "
        "The deterministic recommendation engine is the source of truth for which method was recommended. "
        "Explain why the recommendation fits, compare it with nearby alternatives, and distinguish clearly between "
        "empirical evidence from the study and your conceptual inference. "
        "Do not invent significance claims, literature claims, or unsupported empirical results. "
        "If no retrieved evidence explicitly supports a statistical or literature-backed statement, avoid making it. "
        "If evidence is thin, say so plainly."
    )


def build_explain_recommendation_user_prompt(ctx: RecommendationExplanationContext) -> str:
    shortlist_lines = []
    for idx, item in enumerate(ctx.shortlist[:3], start=1):
        shortlist_lines.append(
            f"{idx}. method={item['method']}, score={item['score']}, "
            f"avg_accuracy={item['avg_accuracy_mean']}, forgetting={item['forgetting_mean']}, "
            f"runtime_hours={item['runtime_hours_mean']}, memory_mb={item['estimated_memory_mb']}"
        )

    evidence_lines = []
    for item in ctx.evidence.items[:8]:
        evidence_lines.append(
            f"- [{item.label}] {item.title}: {item.content[:400]}"
        )

    return f"""
Current request:
- dataset: {ctx.request.dataset}
- memory_budget_mb: {ctx.request.memory_budget_mb}
- compute_budget: {ctx.request.compute_budget}
- acceptable_forgetting: {ctx.request.acceptable_forgetting}
- task_similarity: {ctx.request.task_similarity}
- joint_retraining_allowed: {ctx.request.joint_retraining_allowed}

Engine output:
- recommended_method: {ctx.recommended_method}
- rationale: {ctx.rationale}

Top alternatives:
{chr(10).join(shortlist_lines)}

Method card for the winner:
- label: {ctx.method_card.label}
- family: {ctx.method_card.family}
- mechanism: {ctx.method_card.mechanism}
- strengths: {", ".join(ctx.method_card.strengths)}
- weaknesses: {", ".join(ctx.method_card.weaknesses)}
- works_well_when: {", ".join(ctx.method_card.works_well_when)}

Retrieved evidence:
{chr(10).join(evidence_lines)}

Write a grounded explanation with these sections:
1. Why this recommendation fits here
2. Why it beat the nearby alternatives
3. What is empirically observed vs conceptually inferred

Be explicit about empirical evidence from the current study versus conceptual interpretation.

Keep the explanation concise but useful. Do not mention hidden file paths. Do not claim the LLM chose the method.
""".strip()
