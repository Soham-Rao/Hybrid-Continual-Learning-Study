"""Recommendation engine built on the finalized study artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


DATASET_SAMPLE_BYTES_MB = {
    "permuted_mnist": (1 * 32 * 32 * 4) / (1024 * 1024),
    "split_cifar10": (3 * 32 * 32 * 4) / (1024 * 1024),
    "split_cifar100": (3 * 32 * 32 * 4) / (1024 * 1024),
    "split_mini_imagenet": (3 * 84 * 84 * 4) / (1024 * 1024),
}

METHOD_TRAITS: Dict[str, Dict[str, object]] = {
    "fine_tune": {"family": "baseline", "replay": False, "distill": False, "regularize": False, "joint": False},
    "joint_training": {"family": "baseline", "replay": True, "distill": False, "regularize": False, "joint": True},
    "ewc": {"family": "baseline", "replay": False, "distill": False, "regularize": True, "joint": False},
    "agem": {"family": "baseline", "replay": True, "distill": False, "regularize": False, "joint": False},
    "lwf": {"family": "baseline", "replay": False, "distill": True, "regularize": False, "joint": False},
    "der": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "joint": False},
    "xder": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "joint": False},
    "icarl": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "joint": False},
    "er_ewc": {"family": "hybrid", "replay": True, "distill": False, "regularize": True, "joint": False},
    "progress_compress": {"family": "hybrid", "replay": False, "distill": True, "regularize": True, "joint": False},
    "agem_distill": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "joint": False},
    "si_der": {"family": "hybrid", "replay": True, "distill": True, "regularize": True, "joint": False},
}

COMPUTE_BUDGET_MULTIPLIER = {"low": 0.75, "medium": 1.15, "high": 1.6}
SIMILARITY_WEIGHTS = {
    "low": {"replay": 2.0, "distill": 0.5, "regularize": 0.25},
    "medium": {"replay": 1.0, "distill": 1.0, "regularize": 1.0},
    "high": {"replay": 0.5, "distill": 1.5, "regularize": 1.5},
}


@dataclass
class RecommendationRequest:
    dataset: str
    memory_budget_mb: float
    compute_budget: str = "medium"
    acceptable_forgetting: float | None = None
    task_similarity: str = "medium"
    joint_retraining_allowed: bool = False


def method_traits(method: str) -> Dict[str, object]:
    return METHOD_TRAITS.get(str(method), {"family": "other", "replay": False, "distill": False, "regularize": False, "joint": False})


def estimate_memory_mb(dataset: str, method: str, buffer_size: float) -> float:
    sample_mb = DATASET_SAMPLE_BYTES_MB.get(str(dataset), DATASET_SAMPLE_BYTES_MB["split_cifar10"])
    base = max(float(buffer_size or 0.0) * sample_mb, 0.0)
    traits = method_traits(method)
    if traits.get("replay") and traits.get("distill"):
        base *= 1.2
    if method == "joint_training":
        return round(max(base, 2048.0), 4)
    if method == "progress_compress":
        return round(max(base, 256.0), 4)
    if traits.get("regularize") and base < 16.0:
        return 16.0
    return round(base, 4)


def build_recommendation_profiles(summary_df: pd.DataFrame, leaders_df: pd.DataFrame) -> pd.DataFrame:
    leader_lookup = {row.dataset: row.best_method for row in leaders_df.itertuples(index=False)}
    cluster_lookup = {
        row.dataset: set(str(row.top_cluster_methods).split("|")) if str(row.top_cluster_methods) else set()
        for row in leaders_df.itertuples(index=False)
    }
    profiles = summary_df.copy()
    profiles["estimated_memory_mb"] = profiles.apply(
        lambda row: estimate_memory_mb(str(row["dataset"]), str(row["method"]), float(row["buffer_size"])),
        axis=1,
    )
    profiles["method_family"] = profiles["method"].map(lambda item: str(method_traits(item).get("family", "other")))
    profiles["replay"] = profiles["method"].map(lambda item: bool(method_traits(item).get("replay", False)))
    profiles["distill"] = profiles["method"].map(lambda item: bool(method_traits(item).get("distill", False)))
    profiles["regularize"] = profiles["method"].map(lambda item: bool(method_traits(item).get("regularize", False)))
    profiles["joint"] = profiles["method"].map(lambda item: bool(method_traits(item).get("joint", False)))
    profiles["leader_flag"] = profiles.apply(
        lambda row: str(row["method"]) == str(leader_lookup.get(str(row["dataset"]), "")),
        axis=1,
    )
    profiles["top_cluster_flag"] = profiles.apply(
        lambda row: str(row["method"]) in cluster_lookup.get(str(row["dataset"]), set()),
        axis=1,
    )
    return profiles.sort_values(["dataset", "method"]).reset_index(drop=True)


class RecommendationEngine:
    """Recommend methods from the primary study dataset-method profiles."""

    REQUIRED_COLUMNS = {
        "dataset",
        "method",
        "avg_accuracy_mean",
        "forgetting_mean",
        "runtime_hours_mean",
        "estimated_memory_mb",
        "leader_flag",
        "top_cluster_flag",
        "replay",
        "distill",
        "regularize",
        "joint",
        "caveat_note",
    }

    def __init__(self, profiles_df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(profiles_df.columns)
        if missing:
            raise ValueError(f"Profile dataframe is missing required columns: {sorted(missing)}")
        self.profiles_df = profiles_df.copy()

    @classmethod
    def from_profiles_csv(cls, path: str | Path) -> "RecommendationEngine":
        return cls(pd.read_csv(path))

    def _dataset_pool(self, dataset: str) -> pd.DataFrame:
        pool = self.profiles_df[self.profiles_df["dataset"].astype(str) == str(dataset)].copy()
        if pool.empty:
            raise ValueError(f"No recommendation profiles are available for dataset '{dataset}'.")
        return pool

    def recommend(self, request: RecommendationRequest, top_k: int = 3) -> Dict[str, object]:
        full_pool = self._dataset_pool(request.dataset)
        reference_pool = full_pool[full_pool["joint"] != True].copy()  # noqa: E712
        if reference_pool.empty:
            reference_pool = full_pool.copy()

        pool = full_pool.copy()
        if not request.joint_retraining_allowed:
            pool = pool[pool["joint"] != True].copy()  # noqa: E712
        if pool.empty:
            raise ValueError("No eligible methods remain after applying the current constraints.")

        compute_budget = request.compute_budget if request.compute_budget in COMPUTE_BUDGET_MULTIPLIER else "medium"
        similarity = request.task_similarity if request.task_similarity in SIMILARITY_WEIGHTS else "medium"

        # Keep non-joint rankings stable even when the joint-training toggle changes.
        acc_min = float(reference_pool["avg_accuracy_mean"].min())
        acc_max = float(reference_pool["avg_accuracy_mean"].max())
        forget_min = float(reference_pool["forgetting_mean"].min())
        forget_max = float(reference_pool["forgetting_mean"].max())
        runtime_median = float(reference_pool["runtime_hours_mean"].median())
        runtime_limit = max(runtime_median * COMPUTE_BUDGET_MULTIPLIER[compute_budget], 0.05)

        candidates: List[Dict[str, object]] = []
        for row in pool.itertuples(index=False):
            acc_norm = 1.0 if acc_max == acc_min else (float(row.avg_accuracy_mean) - acc_min) / (acc_max - acc_min)
            forget_norm = 1.0 if forget_max == forget_min else 1.0 - (
                (float(row.forgetting_mean) - forget_min) / (forget_max - forget_min)
            )

            accuracy_component = acc_norm * 55.0
            forgetting_component = forget_norm * 25.0
            memory_component = 0.0
            runtime_component = 0.0
            forgetting_target_component = 0.0
            similarity_component = 0.0
            leader_component = 0.0
            score = accuracy_component + forgetting_component
            reasons: List[str] = []

            if float(row.estimated_memory_mb) <= float(request.memory_budget_mb):
                memory_component += 10.0
                score += 10.0
                reasons.append("fits the stated memory budget")
            else:
                overflow = (float(row.estimated_memory_mb) - float(request.memory_budget_mb)) / max(float(request.memory_budget_mb), 1.0)
                penalty = 30.0 * overflow
                memory_component -= penalty
                score -= penalty
                reasons.append("exceeds the stated memory budget")

            if float(row.runtime_hours_mean) <= runtime_limit:
                runtime_component += 10.0
                score += 10.0
                reasons.append("fits the stated compute budget")
            else:
                overflow = (float(row.runtime_hours_mean) - runtime_limit) / max(runtime_limit, 0.1)
                penalty = 24.0 * overflow
                runtime_component -= penalty
                score -= penalty
                reasons.append("needs more compute than requested")

            if request.acceptable_forgetting is not None:
                if float(row.forgetting_mean) <= float(request.acceptable_forgetting):
                    forgetting_target_component += 8.0
                    score += 8.0
                    reasons.append("meets the acceptable forgetting target")
                else:
                    overflow = (float(row.forgetting_mean) - float(request.acceptable_forgetting)) / max(float(request.acceptable_forgetting), 1.0)
                    penalty = 20.0 * overflow
                    forgetting_target_component -= penalty
                    score -= penalty
                    reasons.append("forgets more than requested")

            sim_weights = SIMILARITY_WEIGHTS[similarity]
            if bool(row.replay):
                similarity_component += sim_weights["replay"]
                score += sim_weights["replay"]
            if bool(row.distill):
                similarity_component += sim_weights["distill"]
                score += sim_weights["distill"]
            if bool(row.regularize):
                similarity_component += sim_weights["regularize"]
                score += sim_weights["regularize"]

            if bool(row.leader_flag):
                leader_component += 4.0
                score += 4.0
                reasons.append("is the current dataset leader in the primary study matrix")
            elif bool(row.top_cluster_flag):
                leader_component += 2.0
                score += 2.0
                reasons.append("belongs to the dataset's top non-inferior cluster")

            if bool(row.joint):
                reasons.append("acts as an upper-bound option when joint retraining is allowed")
            elif bool(row.replay):
                reasons.append("uses replay, which is typically helpful when tasks are less aligned")
            elif bool(row.regularize):
                reasons.append("leans on regularization instead of storing much past data")
            elif bool(row.distill):
                reasons.append("uses distillation to retain prior task behavior")

            if isinstance(row.caveat_note, str) and row.caveat_note.strip():
                reasons.append(f"caveat: {row.caveat_note.strip()}")

            candidates.append(
                {
                    "method": row.method,
                    "score": round(float(score), 4),
                    "avg_accuracy_mean": round(float(row.avg_accuracy_mean), 4),
                    "forgetting_mean": round(float(row.forgetting_mean), 4),
                    "runtime_hours_mean": round(float(row.runtime_hours_mean), 4),
                    "estimated_memory_mb": round(float(row.estimated_memory_mb), 4),
                    "leader_flag": bool(row.leader_flag),
                    "top_cluster_flag": bool(row.top_cluster_flag),
                    "score_components": {
                        "accuracy": round(float(accuracy_component), 4),
                        "forgetting": round(float(forgetting_component), 4),
                        "memory_fit": round(float(memory_component), 4),
                        "runtime_fit": round(float(runtime_component), 4),
                        "forgetting_target": round(float(forgetting_target_component), 4),
                        "similarity_bonus": round(float(similarity_component), 4),
                        "leader_bonus": round(float(leader_component), 4),
                    },
                    "reasons": reasons,
                }
            )

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        rationale = (
            f"Recommended `{best['method']}` for `{request.dataset}` because it offers the strongest "
            "constraint-adjusted empirical trade-off in the primary study matrix."
        )
        return {
            "recommended_method": best["method"],
            "rationale": rationale,
            "shortlist": ranked[:top_k],
        }


def default_recommendation_requests() -> List[RecommendationRequest]:
    return [
        RecommendationRequest("permuted_mnist", 32.0, "low", 40.0, "high", False),
        RecommendationRequest("split_cifar10", 128.0, "low", 35.0, "high", False),
        RecommendationRequest("split_cifar100", 256.0, "medium", 25.0, "medium", False),
        RecommendationRequest("split_mini_imagenet", 512.0, "medium", 20.0, "low", False),
        RecommendationRequest("split_mini_imagenet", 4096.0, "high", 15.0, "low", True),
    ]


def generate_phase8_artifacts(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    summary_path = analysis_dir / "paper_ready_summary.csv"
    leaders_path = analysis_dir / "dataset_leaders.csv"
    if not summary_path.exists() or not leaders_path.exists():
        raise FileNotFoundError(
            "Recommendation artifact generation requires `paper_ready_summary.csv` and `dataset_leaders.csv`."
        )

    summary_df = pd.read_csv(summary_path)
    leaders_df = pd.read_csv(leaders_path)
    profiles_df = build_recommendation_profiles(summary_df, leaders_df)
    profiles_path = analysis_dir / "recommendation_profiles.csv"
    profiles_df.to_csv(profiles_path, index=False)

    engine = RecommendationEngine(profiles_df)
    case_rows: List[Dict[str, object]] = []
    for case_id, request in enumerate(default_recommendation_requests(), start=1):
        result = engine.recommend(request, top_k=3)
        for rank, candidate in enumerate(result["shortlist"], start=1):
            case_rows.append(
                {
                    "case_id": case_id,
                    "dataset": request.dataset,
                    "memory_budget_mb": request.memory_budget_mb,
                    "compute_budget": request.compute_budget,
                    "acceptable_forgetting": request.acceptable_forgetting,
                    "task_similarity": request.task_similarity,
                    "joint_retraining_allowed": request.joint_retraining_allowed,
                    "rank": rank,
                    "recommended_method": result["recommended_method"],
                    "method": candidate["method"],
                    "score": candidate["score"],
                    "avg_accuracy_mean": candidate["avg_accuracy_mean"],
                    "forgetting_mean": candidate["forgetting_mean"],
                    "runtime_hours_mean": candidate["runtime_hours_mean"],
                    "estimated_memory_mb": candidate["estimated_memory_mb"],
                    "leader_flag": candidate["leader_flag"],
                    "top_cluster_flag": candidate["top_cluster_flag"],
                    "rationale": " | ".join(candidate["reasons"]),
                    "summary_rationale": result["rationale"],
                }
            )
    cases_df = pd.DataFrame(case_rows)
    cases_df.to_csv(analysis_dir / "recommendation_cases.csv", index=False)

    notes_lines = [
        "# Recommendation Notes",
        "",
        "The recommendation engine consumes only the finalized primary-study summaries.",
        "Memory is reported as a proxy estimate derived from dataset shape and configured replay storage.",
        "Scoring is empirical-first: average accuracy, forgetting, runtime fit, and memory fit dominate the score.",
        "Task similarity contributes only a small tie-breaker through method traits.",
        "Joint training is excluded only when the request disallows joint retraining.",
        "",
        "## Fixed Case Studies",
        "",
    ]
    for request in default_recommendation_requests():
        notes_lines.append(
            f"- `{request.dataset}`: memory={request.memory_budget_mb} MB, compute={request.compute_budget}, "
            f"forgetting<={request.acceptable_forgetting}, similarity={request.task_similarity}, "
            f"joint_allowed={request.joint_retraining_allowed}"
        )
    notes_lines.append("")
    notes_path = analysis_dir / "phase8_recommendation_notes.md"
    notes_path.write_text("\n".join(notes_lines), encoding="utf-8")

    return {
        "profiles": profiles_df,
        "cases": cases_df,
    }
