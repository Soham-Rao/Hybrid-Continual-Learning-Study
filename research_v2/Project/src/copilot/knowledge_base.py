"""Structured copilot knowledge base for recommendation, chart explanations, and hardware cues."""

from __future__ import annotations

import re
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


HARDWARE_CARDS: dict[str, dict[str, str | float | None]] = {
    "gt210": {
        "kind": "gpu",
        "memory_budget_mb": 1024.0,
        "compute_budget": "low",
        "assumption": "Assumed GT 210 implies roughly 1 GB usable graphics memory and a low compute budget.",
    },
    "gt 210": {
        "kind": "gpu",
        "memory_budget_mb": 1024.0,
        "compute_budget": "low",
        "assumption": "Assumed GT 210 implies roughly 1 GB usable graphics memory and a low compute budget.",
    },
    "gt 730": {
        "kind": "gpu",
        "memory_budget_mb": 1024.0,
        "compute_budget": "low",
        "assumption": "Assumed GT 730 implies a low compute budget and about 1 GB of practical memory headroom.",
    },
    "mx130": {
        "kind": "gpu",
        "memory_budget_mb": 1024.0,
        "compute_budget": "low",
        "assumption": "Assumed MX130 implies a low compute laptop GPU with about 1 GB of practical headroom.",
    },
    "mx150": {
        "kind": "gpu",
        "memory_budget_mb": 2048.0,
        "compute_budget": "low",
        "assumption": "Assumed MX150 implies a low-to-mid laptop GPU with roughly 2 GB class memory.",
    },
    "gtx 1050": {
        "kind": "gpu",
        "memory_budget_mb": 2048.0,
        "compute_budget": "low",
        "assumption": "Assumed GTX 1050 implies a modest discrete GPU and a low compute budget.",
    },
    "gtx 1050 ti": {
        "kind": "gpu",
        "memory_budget_mb": 4096.0,
        "compute_budget": "low",
        "assumption": "Assumed GTX 1050 Ti implies about 4 GB of memory with a still modest compute budget.",
    },
    "gtx 1650": {
        "kind": "gpu",
        "memory_budget_mb": 4096.0,
        "compute_budget": "low",
        "assumption": "Assumed GTX 1650 implies about 4 GB memory and a low-to-mid local compute budget.",
    },
    "rtx 2050": {
        "kind": "gpu",
        "memory_budget_mb": 4096.0,
        "compute_budget": "medium",
        "assumption": "Assumed RTX 2050 implies an entry modern RTX laptop setup with about 4 GB memory.",
    },
    "rtx 3050": {
        "kind": "gpu",
        "memory_budget_mb": 4096.0,
        "compute_budget": "medium",
        "assumption": "Assumed RTX 3050 implies around 4 GB graphics memory and a medium compute budget.",
    },
    "rtx 4050": {
        "kind": "gpu",
        "memory_budget_mb": 6144.0,
        "compute_budget": "medium",
        "assumption": "Assumed RTX 4050 implies a mid-range local setup with roughly 6 GB graphics memory.",
    },
    "rtx 4060": {
        "kind": "gpu",
        "memory_budget_mb": 8192.0,
        "compute_budget": "high",
        "assumption": "Assumed RTX 4060 implies a stronger local setup with about 8 GB graphics memory.",
    },
    "rtx 4070": {
        "kind": "gpu",
        "memory_budget_mb": 8192.0,
        "compute_budget": "high",
        "assumption": "Assumed RTX 4070 implies a high compute local setup with about 8 GB graphics memory.",
    },
    "rtx 4090": {
        "kind": "gpu",
        "memory_budget_mb": 16384.0,
        "compute_budget": "high",
        "assumption": "Assumed RTX 4090 implies a very high local compute budget and abundant graphics memory.",
    },
    "intel hd 620": {
        "kind": "igpu",
        "memory_budget_mb": 512.0,
        "compute_budget": "low",
        "assumption": "Assumed Intel HD 620 implies integrated graphics and a very constrained compute budget.",
    },
    "uhd 620": {
        "kind": "igpu",
        "memory_budget_mb": 512.0,
        "compute_budget": "low",
        "assumption": "Assumed Intel UHD 620 implies integrated graphics and a very constrained compute budget.",
    },
    "iris xe": {
        "kind": "igpu",
        "memory_budget_mb": 1024.0,
        "compute_budget": "low",
        "assumption": "Assumed Intel Iris Xe implies integrated graphics with limited practical memory headroom.",
    },
    "cpu only": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "low",
        "assumption": "Interpreted 'CPU only' as a low compute setting for continual-learning training.",
    },
    "without gpu": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "low",
        "assumption": "Interpreted the no-GPU description as a low compute setting.",
    },
    "i3": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "low",
        "assumption": "Used a conservative estimate for an Intel i3-class CPU.",
    },
    "ryzen 3": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "low",
        "assumption": "Used a conservative estimate for a Ryzen 3-class CPU.",
    },
    "i5": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "medium",
        "assumption": "Used a moderate estimate for an Intel i5-class CPU.",
    },
    "ryzen 5": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "medium",
        "assumption": "Used a moderate estimate for a Ryzen 5-class CPU.",
    },
    "i7": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "high",
        "assumption": "Used a stronger estimate for an Intel i7-class CPU.",
    },
    "ryzen 7": {
        "kind": "cpu",
        "memory_budget_mb": None,
        "compute_budget": "high",
        "assumption": "Used a stronger estimate for a Ryzen 7-class CPU.",
    },
}


SETTINGS_QUERY_TERMS: tuple[str, ...] = (
    "gpu",
    "vram",
    "ram",
    "memory",
    "compute",
    "retrain",
    "retention",
    "forgetting",
    "laptop",
    "cpu",
    "hardware",
    "budget",
    "what should i do",
    "what should i use",
    "suggest settings",
    "infer my settings",
)


CHART_ALIASES: dict[str, tuple[str, ...]] = {
    "score_breakdown": ("score breakdown", "score contribution", "breakdown"),
    "shortlist": ("shortlist", "top-3 shortlist", "top 3 shortlist", "case study", "case-study"),
    "decision_flow": ("decision tree", "decision flow", "alluvial", "green flows", "grey flows", "gray flows"),
    "grouped_metrics": ("grouped metric", "metric bars", "grouped bar", "comparison bars", "bar chart"),
    "accuracy_forgetting": ("accuracy vs forgetting", "accuracy-versus-forgetting", "retention frontier"),
    "accuracy_runtime": ("accuracy vs runtime", "accuracy vs time", "accuracy-runtime", "compute time"),
    "accuracy_memory": ("accuracy vs memory", "accuracy vs estimated memory", "accuracy vs memory proxy", "storage footprint"),
    "significance": ("significance matrix", "pairwise p", "p-values", "holm-adjusted", "holm corrected", "significance heatmap"),
    "effect": ("effect-size matrix", "effect size", "rank-biserial", "effect heatmap"),
    "pareto": ("pareto", "non-joint pareto", "frontier"),
    "heatmap_accuracy": ("accuracy heatmap", "average accuracy heatmap", "cross-dataset accuracy heatmap"),
    "heatmap_forgetting": ("forgetting heatmap", "retention heatmap", "cross-dataset forgetting heatmap"),
    "rank_slope": ("rank slope", "slope chart", "ranking shifts", "rank plot"),
    "top_cluster": ("top cluster", "top-cluster", "cluster membership", "non-inferior cluster"),
    "friedman_rank": ("friedman", "average rank", "cross-dataset ranking", "friedman ranking"),
    "ablation_runtime": ("ablation runtime", "runtime sensitivity", "ablation compute"),
    "ablation_memory": ("ablation memory", "memory sensitivity", "proxy memory ablation"),
    "robustness": ("robustness", "resume", "restart", "resume robustness", "restart robustness"),
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
    "shortlist": {
        "title": "shortlist chart",
        "reading": "This view compares the saved top candidate set for one request rather than the whole dataset leaderboard.",
        "focus": "Read it as a ranked local comparison: the winner is best for that exact request, and the other bars show the nearest empirical alternatives.",
    },
    "decision_flow": {
        "title": "decision-flow chart",
        "reading": "This alluvial decision view is a visual layer over the same deterministic recommendation engine used elsewhere in the dashboard.",
        "focus": "Green flow marks the current path through bucketed constraints, while grey flows keep nearby alternatives visible for context.",
    },
    "grouped_metrics": {
        "title": "grouped metric comparison chart",
        "reading": "This view puts multiple metrics side by side so you can compare how methods trade accuracy, forgetting, runtime, and memory at once.",
        "focus": "No single bar decides the winner here; it is mainly a compact way to spot methods that are consistently strong or obviously lopsided.",
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
    "friedman_rank": {
        "title": "Friedman rank chart",
        "reading": "This chart summarizes cross-dataset average ranks and is descriptive rather than decisive because the study only has four datasets.",
        "focus": "Lower average rank is better, but treat it as a broad pattern summary alongside the per-dataset evidence.",
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


def chart_focus_from_text(text: str) -> str:
    lowered = text.lower()
    for chart_focus, aliases in CHART_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return chart_focus
    if "accuracy" in lowered and "forgetting" in lowered:
        return "accuracy_forgetting"
    if "accuracy" in lowered and ("runtime" in lowered or "time" in lowered or "speed" in lowered):
        return "accuracy_runtime"
    if "accuracy" in lowered and ("memory" in lowered or "proxy memory" in lowered or "estimated memory" in lowered):
        return "accuracy_memory"
    if "heatmap" in lowered and "forget" in lowered:
        return "heatmap_forgetting"
    if "heatmap" in lowered and ("accuracy" in lowered or "average accuracy" in lowered):
        return "heatmap_accuracy"
    return "generic"


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
                f"This {guide['title']} explains why the current winner scores highest for the active request.",
                (
                    f"For {dataset_label}, the current winner is `{winner['method']}` with total score {winner['score']:.2f}. "
                    f"The biggest score drivers right now are {dominant}."
                ),
                guide["focus"],
                f"Nearby alternatives in the same request are {facts.shortlist_summary}.",
            ]
        )
    if facts.chart_focus == "shortlist":
        return "\n\n".join(
            [
                f"This {guide['title']} compares the current winner against the nearest saved alternatives for the same request.",
                (
                    f"For {dataset_label}, the top-ranked candidate here is `{winner['method']}` with score {winner.get('score', 0.0):.2f}, "
                    f"average accuracy {winner['avg_accuracy_mean']:.2f}, forgetting {winner['forgetting_mean']:.2f}, "
                    f"runtime {winner['runtime_hours_mean']:.2f} hours, and estimated memory {winner['estimated_memory_mb']:.2f} MB."
                ),
                guide["focus"],
                f"The nearby alternatives in this saved case are {facts.shortlist_summary}.",
            ]
        )
    if facts.chart_focus == "decision_flow":
        return "\n\n".join(
            [
                f"This {guide['title']} is a bucketed visual explanation of the same recommendation logic used elsewhere in the dashboard.",
                (
                    f"For {dataset_label}, the current path ends at `{winner['method']}`, which is the method that best matches the active constraint bucket under the deterministic scorer."
                ),
                guide["focus"],
                "Use this view to understand which constraint buckets steer the recommendation, not as a second independent recommender.",
            ]
        )
    if facts.chart_focus == "grouped_metrics":
        return "\n\n".join(
            [
                f"This {guide['title']} compares multiple outcome dimensions side by side for the selected dataset.",
                (
                    f"For {dataset_label}, `{facts.best_accuracy['method']}` is the raw accuracy leader, "
                    f"`{facts.lowest_forgetting['method']}` retains best, `{facts.fastest['method']}` is the fastest, and "
                    f"`{facts.smallest_memory['method']}` is the lightest."
                ),
                (
                    f"The current recommendation `{winner['method']}` is the dashboard's best compromise under the active request, "
                    f"which is why it may not lead every single bar."
                ),
                guide["focus"],
            ]
        )
    if facts.chart_focus == "accuracy_forgetting":
        return "\n\n".join(
            [
                f"This {guide['title']} compares final performance against retention.",
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
                f"This {guide['title']} is about how much performance you buy with extra training time.",
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
                f"This {guide['title']} shows how much performance each method delivers for its proxy memory footprint.",
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
    if facts.chart_focus == "significance":
        return "\n\n".join(
            [
                f"This {guide['title']} should be read as evidence-of-difference, not evidence-of-size.",
                (
                    f"For {dataset_label}, use this matrix to see which method pairs still look meaningfully separated after Holm correction in the current dataset."
                ),
                guide["focus"],
                "Cells with lower adjusted p-values indicate stronger evidence that the row and column methods differ on the selected metric.",
            ]
        )
    if facts.chart_focus == "effect":
        return "\n\n".join(
            [
                f"This {guide['title']} is the practical-strength companion to the significance matrix.",
                (
                    f"For {dataset_label}, positive values mean the row method tends to outperform the column method, while negative values mean the reverse."
                ),
                guide["focus"],
                "Large-magnitude values indicate stronger practical separation even when significance alone is ambiguous.",
            ]
        )
    if facts.chart_focus == "pareto":
        return "\n\n".join(
            [
                f"This {guide['title']} shows only non-joint methods that are not dominated on the study's trade-off dimensions.",
                (
                    f"For {dataset_label}, the displayed points are the non-joint candidates that survive a Pareto-style screen over accuracy, runtime, and proxy memory."
                ),
                guide["focus"],
                "A point can disappear from this view even if it is good overall, as long as another non-joint method is at least as good on all included dimensions.",
            ]
        )
    if facts.chart_focus == "heatmap_accuracy":
        return "\n\n".join(
            [
                f"This {guide['title']} is a broad leaderboard-style overview of the full study matrix.",
                (
                    f"For {dataset_label}, it helps you see whether the selected dataset is one where strong methods clearly separate or where the whole field compresses."
                ),
                guide["focus"],
                "Use it to spot cross-dataset consistency patterns before drilling back into one dataset.",
            ]
        )
    if facts.chart_focus == "heatmap_forgetting":
        return "\n\n".join(
            [
                f"This {guide['title']} is a cross-dataset retention map rather than a direct performance leaderboard.",
                (
                    f"For {dataset_label}, darker or worse cells indicate methods that struggle more to preserve earlier tasks on that dataset."
                ),
                guide["focus"],
                "Compare it with the accuracy heatmap to see which methods trade retention for raw final performance.",
            ]
        )
    if facts.chart_focus == "rank_slope":
        return "\n\n".join(
            [
                f"This {guide['title']} shows how sensitive method ordering is across datasets.",
                (
                    f"For {dataset_label}, a flatter line indicates a method that behaves more consistently as the benchmark changes."
                ),
                guide["focus"],
                "It is especially useful for spotting methods that win in one dataset but drop sharply elsewhere.",
            ]
        )
    if facts.chart_focus == "top_cluster":
        return "\n\n".join(
            [
                f"This {guide['title']} is about consistency of statistical competitiveness, not about raw leaderboard position alone.",
                (
                    f"For {dataset_label}, it shows which methods most often stay in the non-inferior set around the dataset leader."
                ),
                guide["focus"],
                "A method with a high count here is often a safe recommendation even if it is not the absolute best on every dataset.",
            ]
        )
    if facts.chart_focus == "friedman_rank":
        return "\n\n".join(
            [
                f"This {guide['title']} summarizes cross-dataset ranking behavior across the full benchmark set.",
                (
                    f"For {dataset_label}, it provides context about where the selected dataset fits relative to the broader ranking pattern."
                ),
                guide["focus"],
                "Because only four datasets were used, treat this chart as a supporting pattern summary rather than the final decision rule.",
            ]
        )
    if facts.chart_focus == "ablation_runtime":
        return "\n\n".join(
            [
                f"This {guide['title']} is secondary evidence about engineering cost rather than about the main leaderboard.",
                (
                    f"For {dataset_label}, it shows how representative ablation variants changed total runtime relative to their baseline family."
                ),
                guide["focus"],
                "Use it to understand mechanism cost, not to replace the primary-study comparison.",
            ]
        )
    if facts.chart_focus == "ablation_memory":
        return "\n\n".join(
            [
                f"This {guide['title']} is a proxy-oriented view of memory-sensitive configuration changes.",
                (
                    f"For {dataset_label}, it highlights how settings like replay buffer and batch size shift the memory-related profile of ablation variants."
                ),
                guide["focus"],
                "This is configuration-level context, not measured peak-device telemetry.",
            ]
        )
    if facts.chart_focus == "robustness":
        return "\n\n".join(
            [
                f"This {guide['title']} is about pipeline trustworthiness rather than about method quality alone.",
                (
                    f"For {dataset_label}, points closer to zero mean restart/resume behavior stayed close to the original primary-study result."
                ),
                guide["focus"],
                "Large shifts suggest the run is more sensitive to interruption or checkpointing behavior than expected.",
            ]
        )
    return "\n\n".join(
        [
            f"This {guide['title']} should be read as a trade-off surface rather than as a single-metric ranking.",
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
