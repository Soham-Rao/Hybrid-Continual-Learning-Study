"""Constraint-aware recommendation engine built on empirical Phase 5 summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


DATASET_SAMPLE_BYTES_MB = {
    "permuted_mnist": (1 * 28 * 28 * 4) / (1024 * 1024),
    "split_cifar10": (3 * 32 * 32 * 4) / (1024 * 1024),
    "split_cifar100": (3 * 32 * 32 * 4) / (1024 * 1024),
    "split_mini_imagenet": (3 * 84 * 84 * 4) / (1024 * 1024),
    "seq_tiny_imagenet": (3 * 64 * 64 * 4) / (1024 * 1024),
}

METHOD_TRAITS: Dict[str, Dict[str, object]] = {
    "fine_tune": {"family": "baseline", "replay": False, "distill": False, "regularize": False, "architecture": False, "joint": False},
    "joint_training": {"family": "baseline", "replay": True, "distill": False, "regularize": False, "architecture": False, "joint": True},
    "ewc": {"family": "baseline", "replay": False, "distill": False, "regularize": True, "architecture": False, "joint": False},
    "agem": {"family": "baseline", "replay": True, "distill": False, "regularize": False, "architecture": False, "joint": False},
    "lwf": {"family": "baseline", "replay": False, "distill": True, "regularize": False, "architecture": False, "joint": False},
    "der": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "architecture": False, "joint": False},
    "xder": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "architecture": False, "joint": False},
    "icarl": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "architecture": False, "joint": False},
    "er_ewc": {"family": "hybrid", "replay": True, "distill": False, "regularize": True, "architecture": False, "joint": False},
    "progress_compress": {"family": "hybrid", "replay": False, "distill": True, "regularize": True, "architecture": True, "joint": False},
    "agem_distill": {"family": "hybrid", "replay": True, "distill": True, "regularize": False, "architecture": False, "joint": False},
    "si_der": {"family": "hybrid", "replay": True, "distill": True, "regularize": True, "architecture": False, "joint": False},
}

COMPUTE_BUDGET_MULTIPLIER = {"low": 0.75, "medium": 1.15, "high": 1.75}
SIMILARITY_WEIGHTS = {
    "low": {"replay": 4.0, "distill": 1.0, "regularize": 0.5},
    "medium": {"replay": 2.0, "distill": 2.0, "regularize": 2.0},
    "high": {"replay": 1.0, "distill": 3.0, "regularize": 3.0},
}


@dataclass
class RecommendationRequest:
    dataset: str
    memory_budget_mb: float
    compute_budget: str = "medium"
    acceptable_forgetting: float | None = None
    task_similarity: str = "medium"
    joint_retraining_allowed: bool = False


class RecommendationEngine:
    """Recommend methods from aggregated empirical summaries."""

    def __init__(self, summary_df: pd.DataFrame) -> None:
        required = {
            "dataset",
            "method",
            "source_group",
            "avg_accuracy_mean",
            "forgetting_mean",
            "runtime_hours_mean",
            "buffer_size_mean",
        }
        missing = required - set(summary_df.columns)
        if missing:
            raise ValueError(f"Summary dataframe is missing required columns: {sorted(missing)}")
        self.summary_df = summary_df.copy()

    @classmethod
    def from_summary_csv(cls, path: str) -> "RecommendationEngine":
        return cls(pd.read_csv(path))

    def _main_rows(self, dataset: str) -> pd.DataFrame:
        pool = self.summary_df[
            (self.summary_df["dataset"].astype(str) == dataset)
            & (self.summary_df["source_group"].isin([
                "baseline_fwt",
                "hybrid_fwt",
                "hybrid_fwt_sider_fix",
                "phase4_local_mini",
            ]))
        ].copy()
        if "is_primary_run" in pool.columns:
            pool = pool[pool["is_primary_run"] == True].copy()
        sort_cols = [col for col in ["seeds", "avg_accuracy_mean"] if col in pool.columns]
        if sort_cols:
            ascending = [False for _ in sort_cols]
            pool = pool.sort_values(sort_cols, ascending=ascending)
        return pool.drop_duplicates(subset=["method"], keep="first")

    def _estimated_memory_mb(self, dataset: str, method: str, buffer_size_mean: float) -> float:
        sample_mb = DATASET_SAMPLE_BYTES_MB.get(dataset, DATASET_SAMPLE_BYTES_MB["split_cifar10"])
        base = float(buffer_size_mean or 0.0) * sample_mb
        if METHOD_TRAITS.get(method, {}).get("distill", False) and METHOD_TRAITS.get(method, {}).get("replay", False):
            base *= 1.2
        if method == "joint_training":
            return max(base, 2048.0)
        if method == "progress_compress":
            return max(base, 256.0)
        return max(base, 16.0 if METHOD_TRAITS.get(method, {}).get("regularize", False) else base)

    def recommend(self, request: RecommendationRequest, top_k: int = 3) -> Dict[str, object]:
        pool = self._main_rows(request.dataset)
        if pool.empty:
            raise ValueError(f"No summary rows available for dataset '{request.dataset}'.")

        if not request.joint_retraining_allowed:
            pool = pool[pool["method"] != "joint_training"].copy()
        if pool.empty:
            raise ValueError("No eligible methods remain after applying the joint-training constraint.")

        compute_budget = request.compute_budget if request.compute_budget in COMPUTE_BUDGET_MULTIPLIER else "medium"
        task_similarity = request.task_similarity if request.task_similarity in SIMILARITY_WEIGHTS else "medium"

        acc_min = float(pool["avg_accuracy_mean"].min())
        acc_max = float(pool["avg_accuracy_mean"].max())
        f_min = float(pool["forgetting_mean"].min())
        f_max = float(pool["forgetting_mean"].max())
        runtime_median = float(pool["runtime_hours_mean"].median()) if not pool["runtime_hours_mean"].empty else 0.0
        runtime_limit = max(runtime_median * COMPUTE_BUDGET_MULTIPLIER[compute_budget], 0.05)

        candidates: List[Dict[str, object]] = []
        for row in pool.itertuples(index=False):
            traits = METHOD_TRAITS.get(row.method, {})
            est_memory = self._estimated_memory_mb(str(row.dataset), row.method, float(row.buffer_size_mean))

            acc_norm = 1.0 if acc_max == acc_min else (float(row.avg_accuracy_mean) - acc_min) / (acc_max - acc_min)
            forget_norm = 1.0 if f_max == f_min else 1.0 - ((float(row.forgetting_mean) - f_min) / (f_max - f_min))
            score = acc_norm * 55.0 + forget_norm * 25.0
            reasons: List[str] = []

            if est_memory <= request.memory_budget_mb:
                score += 10.0
                reasons.append("fits the current memory budget")
            else:
                overflow = (est_memory - request.memory_budget_mb) / max(request.memory_budget_mb, 1.0)
                score -= 30.0 * overflow
                reasons.append("exceeds the stated memory budget")

            if float(row.runtime_hours_mean) <= runtime_limit:
                score += 6.0
                reasons.append("matches the stated compute budget")
            else:
                overflow = (float(row.runtime_hours_mean) - runtime_limit) / max(runtime_limit, 0.1)
                score -= 18.0 * overflow
                reasons.append("needs more compute time than the requested budget")

            if request.acceptable_forgetting is not None:
                if float(row.forgetting_mean) <= request.acceptable_forgetting:
                    score += 8.0
                    reasons.append("stays within the acceptable forgetting target")
                else:
                    overflow = (float(row.forgetting_mean) - request.acceptable_forgetting) / max(request.acceptable_forgetting, 1.0)
                    score -= 25.0 * overflow
                    reasons.append("forgets more than the requested threshold")

            sim_weights = SIMILARITY_WEIGHTS[task_similarity]
            if traits.get("replay"):
                score += sim_weights["replay"]
            if traits.get("distill"):
                score += sim_weights["distill"]
            if traits.get("regularize"):
                score += sim_weights["regularize"]

            if traits.get("joint"):
                reasons.append("acts as an upper-bound method when full retraining is allowed")
            elif traits.get("replay"):
                reasons.append("uses replay, which helps when tasks are weakly aligned")
            elif traits.get("regularize"):
                reasons.append("leans on parameter regularization rather than storing much past data")
            elif traits.get("distill"):
                reasons.append("uses distillation to preserve prior representations")

            candidates.append(
                {
                    "method": row.method,
                    "score": round(float(score), 4),
                    "avg_accuracy_mean": round(float(row.avg_accuracy_mean), 4),
                    "forgetting_mean": round(float(row.forgetting_mean), 4),
                    "runtime_hours_mean": round(float(row.runtime_hours_mean), 4),
                    "estimated_memory_mb": round(float(est_memory), 4),
                    "reasons": reasons,
                }
            )

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        rationale = (
            f"Recommended `{best['method']}` for `{request.dataset}` because it offers the strongest "
            f"constraint-adjusted trade-off among the eligible methods."
        )
        return {
            "recommended_method": best["method"],
            "rationale": rationale,
            "shortlist": ranked[:top_k],
        }
