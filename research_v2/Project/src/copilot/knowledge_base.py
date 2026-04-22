"""Structured copilot knowledge base for recommendation and chart explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.dashboard_data import DATASET_LABELS

from .context_builder import RecommendationExplanationContext


DATASET_NOTES: dict[str, dict[str, str]] = {
    "permuted_mnist": {
        "label": "Permuted MNIST",
        "difficulty": "a lighter but still interference-sensitive benchmark",
        "implication": "memory and retention trade-offs can be studied cheaply here, but the benchmark can still punish methods that forget quickly.",
    },
    "split_cifar10": {
        "label": "Split CIFAR-10",
        "difficulty": "a moderate natural-image benchmark",
        "implication": "methods need a stronger balance between retention and image-class discrimination than on PMNIST.",
    },
    "split_cifar100": {
        "label": "Split CIFAR-100",
        "difficulty": "a harder many-class benchmark",
        "implication": "class granularity makes forgetting and replay design more important than on simpler datasets.",
    },
    "split_mini_imagenet": {
        "label": "Split Mini-ImageNet",
        "difficulty": "the hardest image benchmark in the current study matrix",
        "implication": "runtime, memory, and retention pressure all matter more here, so compromise methods often beat single-metric winners.",
    },
}


CHART_GUIDES: dict[str, dict[str, str]] = {
    "score_breakdown": {
        "title": "score-breakdown chart",
        "reading": "This view explains why the current winner scores highest for the active request, not which method is globally best.",
        "focus": "Positive bars help the winner under the current constraints, while negative bars show penalties from runtime, memory, or forgetting pressure.",
    },
    "accuracy_forgetting": {
        "title": "accuracy-vs-forgetting chart",
        "reading": "This view compares final performance against retention, so the sweet spot is high accuracy with low forgetting.",
        "focus": "A method dominates only if it improves both metrics at once; otherwise the chart is showing a genuine trade-off frontier.",
    },
    "accuracy_runtime": {
        "title": "accuracy-vs-runtime chart",
        "reading": "This view compares performance against training cost.",
        "focus": "Methods farther right may buy more accuracy, but only if the extra compute is acceptable for the current budget.",
    },
    "accuracy_memory": {
        "title": "accuracy-vs-estimated-memory chart",
        "reading": "This view compares performance against the proxy storage footprint of each method.",
        "focus": "Methods that stay high while using less memory are the most attractive when storage is tight.",
    },
    "significance": {
        "title": "significance matrix",
        "reading": "This matrix shows which method pairs still look different after correction within the selected dataset.",
        "focus": "Use it with effect size, because a low adjusted p-value alone does not tell you whether the difference is practically large.",
    },
    "effect": {
        "title": "effect-size matrix",
        "reading": "This matrix shows the direction and magnitude of pairwise differences.",
        "focus": "It complements the significance matrix by showing whether a difference is small, moderate, or large in practical terms.",
    },
    "heatmap_accuracy": {
        "title": "cross-dataset accuracy heatmap",
        "reading": "This heatmap shows where average accuracy is strongest across the full dataset-method matrix.",
        "focus": "Use it to spot consistent winners and datasets where all methods struggle.",
    },
    "heatmap_forgetting": {
        "title": "cross-dataset forgetting heatmap",
        "reading": "This heatmap shows where forgetting remains low or high across the study.",
        "focus": "Lower values mean better retention, so read it as a retention map rather than a leaderboard.",
    },
    "rank_slope": {
        "title": "rank slope chart",
        "reading": "This chart shows how method ordering shifts from one dataset to another.",
        "focus": "Steeper line movement means a method is less stable across datasets.",
    },
    "top_cluster": {
        "title": "top-cluster chart",
        "reading": "This chart shows how often a method remains statistically competitive with the leader.",
        "focus": "It is a consistency view, not an absolute-performance view.",
    },
    "ablation_runtime": {
        "title": "ablation runtime chart",
        "reading": "This view compares how ablation variants changed total runtime.",
        "focus": "Treat it as mechanism context rather than a primary-study claim.",
    },
    "ablation_memory": {
        "title": "ablation memory chart",
        "reading": "This view highlights memory-relevant settings such as replay buffer and batch size.",
        "focus": "It is proxy-oriented and should not be read as measured peak VRAM.",
    },
    "robustness": {
        "title": "robustness chart",
        "reading": "This view shows how restart/resume runs deviated from their primary-study counterparts.",
        "focus": "It is about pipeline reliability rather than headline method quality.",
    },
    "generic": {
        "title": "trade-off chart",
        "reading": "This view should be read as a trade-off surface rather than as a single-metric ranking.",
        "focus": "The recommended method is the best fit for the active request, not necessarily the global leader on any one metric.",
    },
}


@dataclass(frozen=True)
class ChartExplanationFacts:
    chart_focus: str
    dataset: str
    winner: dict[str, Any]
    best_accuracy: dict[str, Any]
    lowest_forgetting: dict[str, Any]
    fastest: dict[str, Any]
    smallest_memory: dict[str, Any]
    shortlist_summary: str


def recommendation_explanation_draft(ctx: RecommendationExplanationContext) -> tuple[str, str]:
    best = ctx.shortlist[0]
    alternatives = [item["method"] for item in ctx.shortlist[1:3]]
    alt_text = ", ".join(alternatives) if alternatives else "nearby alternatives"
    dataset_note = DATASET_NOTES.get(
        ctx.request.dataset,
        {
            "label": DATASET_LABELS.get(ctx.request.dataset, ctx.request.dataset),
            "difficulty": "the current benchmark",
            "implication": "trade-offs should be read in the context of the active request rather than in isolation.",
        },
    )
    cluster_note = (
        "It is also the current dataset leader."
        if bool(best.get("leader_flag"))
        else "It remains in the dataset's top non-inferior cluster."
        if bool(best.get("top_cluster_flag"))
        else "It is not the raw leader, but it currently fits the active constraints better than the nearby alternatives."
    )
    summary = (
        f"{ctx.method_card.label} fits this request because it gives the strongest constraint-adjusted trade-off "
        f"for {dataset_note['label']} under the active memory, runtime, and retention settings."
    )
    explanation = "\n\n".join(
        [
            (
                f"Why this fits here: {dataset_note['label']} is {dataset_note['difficulty']}, so {dataset_note['implication']} "
                f"For the current request, `{ctx.recommended_method}` reaches mean accuracy {best['avg_accuracy_mean']:.2f}, "
                f"forgetting {best['forgetting_mean']:.2f}, runtime {best['runtime_hours_mean']:.2f} hours, "
                f"and estimated memory {best['estimated_memory_mb']:.2f} MB. {cluster_note}"
            ),
            (
                f"Why it beats the nearby alternatives: the shortlist behind it is {alt_text}. "
                "The deterministic recommendation engine ranked the winner highest because its empirical score balance was better under the exact request, "
                "not because the copilot chose it."
            ),
            (
                f"Conceptual interpretation: {ctx.method_card.mechanism} "
                f"It tends to work best when {ctx.method_card.works_well_when[0].lower()}, "
                f"while its main weakness is that {ctx.method_card.weaknesses[0].lower()}."
            ),
        ]
    )
    return summary, explanation


def chart_explanation_draft(facts: ChartExplanationFacts) -> str:
    guide = CHART_GUIDES.get(facts.chart_focus, CHART_GUIDES["generic"])
    dataset_label = DATASET_LABELS.get(facts.dataset, facts.dataset)
    winner = facts.winner
    if facts.chart_focus == "score_breakdown":
        components = winner.get("score_components", {})
        sorted_components = sorted(
            ((name, value) for name, value in components.items()),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        dominant = ", ".join(f"{name.replace('_', ' ')}={float(value):.2f}" for name, value in sorted_components[:4])
        return "\n\n".join(
            [
                guide["reading"],
                (
                    f"For {dataset_label}, the current winner is `{winner['method']}` with total score {winner['score']:.2f}. "
                    f"The biggest score drivers right now are {dominant}."
                ),
                guide["focus"],
                f"Nearby alternatives in the same request are {facts.shortlist_summary}.",
            ]
        )
    if facts.chart_focus == "accuracy_forgetting":
        return "\n\n".join(
            [
                guide["reading"],
                (
                    f"For {dataset_label}, the highest-accuracy method is `{facts.best_accuracy['method']}` at "
                    f"{float(facts.best_accuracy['avg_accuracy_mean']):.2f}, while the lowest-forgetting method is "
                    f"`{facts.lowest_forgetting['method']}` at {float(facts.lowest_forgetting['forgetting_mean']):.2f} forgetting."
                ),
                (
                    f"The currently recommended method is `{winner['method']}` at accuracy {winner['avg_accuracy_mean']:.2f} "
                    f"and forgetting {winner['forgetting_mean']:.2f}, so the dashboard is favoring its balance under the active constraints."
                ),
                f"{guide['focus']} Nearby alternatives are {facts.shortlist_summary}.",
            ]
        )
    if facts.chart_focus == "accuracy_runtime":
        return "\n\n".join(
            [
                guide["reading"],
                (
                    f"For {dataset_label}, the accuracy leader is `{facts.best_accuracy['method']}` at "
                    f"{float(facts.best_accuracy['avg_accuracy_mean']):.2f}, while the fastest method is "
                    f"`{facts.fastest['method']}` at {float(facts.fastest['runtime_hours_mean']):.2f} hours."
                ),
                (
                    f"The current recommendation `{winner['method']}` sits at accuracy {winner['avg_accuracy_mean']:.2f} "
                    f"and runtime {winner['runtime_hours_mean']:.2f} hours."
                ),
                f"{guide['focus']} Nearby alternatives are {facts.shortlist_summary}.",
            ]
        )
    if facts.chart_focus == "accuracy_memory":
        return "\n\n".join(
            [
                guide["reading"],
                (
                    f"For {dataset_label}, the accuracy leader is `{facts.best_accuracy['method']}` at "
                    f"{float(facts.best_accuracy['avg_accuracy_mean']):.2f}, while the lightest method is "
                    f"`{facts.smallest_memory['method']}` at {float(facts.smallest_memory['estimated_memory_mb']):.1f} MB."
                ),
                (
                    f"The current recommendation `{winner['method']}` uses an estimated {winner['estimated_memory_mb']:.1f} MB "
                    f"for {winner['avg_accuracy_mean']:.2f} average accuracy."
                ),
                f"{guide['focus']} Nearby alternatives are {facts.shortlist_summary}.",
            ]
        )
    return "\n\n".join(
        [
            guide["reading"],
            (
                f"For {dataset_label}, the current recommendation is `{winner['method']}` with average accuracy "
                f"{winner['avg_accuracy_mean']:.2f}, forgetting {winner['forgetting_mean']:.2f}, runtime "
                f"{winner['runtime_hours_mean']:.2f} hours, and estimated memory {winner['estimated_memory_mb']:.1f} MB."
            ),
            (
                f"The raw-accuracy leader is `{facts.best_accuracy['method']}` at {float(facts.best_accuracy['avg_accuracy_mean']):.2f}, "
                f"the lowest-forgetting method is `{facts.lowest_forgetting['method']}` at {float(facts.lowest_forgetting['forgetting_mean']):.2f}, "
                f"the fastest is `{facts.fastest['method']}` at {float(facts.fastest['runtime_hours_mean']):.2f} hours, and the lightest is "
                f"`{facts.smallest_memory['method']}` at {float(facts.smallest_memory['estimated_memory_mb']):.1f} MB."
            ),
            f"{guide['focus']} Nearby alternatives are {facts.shortlist_summary}.",
        ]
    )
