"""Structured settings inference and confirmation-ready actions for the copilot."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.recommendation.engine import RecommendationRequest


DATASET_ALIASES = {
    "permuted mnist": "permuted_mnist",
    "pmnist": "permuted_mnist",
    "mnist": "permuted_mnist",
    "split cifar-10": "split_cifar10",
    "split cifar10": "split_cifar10",
    "cifar-10": "split_cifar10",
    "cifar10": "split_cifar10",
    "split cifar-100": "split_cifar100",
    "split cifar100": "split_cifar100",
    "cifar-100": "split_cifar100",
    "cifar100": "split_cifar100",
    "mini imagenet": "split_mini_imagenet",
    "mini-imagenet": "split_mini_imagenet",
    "split mini imagenet": "split_mini_imagenet",
    "split mini-imagenet": "split_mini_imagenet",
}

GPU_HINTS = {
    "gt210": {"memory_budget_mb": 1024.0, "compute_budget": "low", "assumption": "Assumed GT 210 implies roughly 1 GB usable graphics memory and a low compute budget."},
    "gt 210": {"memory_budget_mb": 1024.0, "compute_budget": "low", "assumption": "Assumed GT 210 implies roughly 1 GB usable graphics memory and a low compute budget."},
    "rtx 4050": {"memory_budget_mb": 6144.0, "compute_budget": "medium", "assumption": "Assumed RTX 4050 implies a mid-range local setup with roughly 6 GB graphics memory."},
}


@dataclass(frozen=True)
class InferredSettingsResult:
    request: RecommendationRequest
    assumptions: tuple[str, ...]
    parsed_from_text: str
    requires_confirmation: bool
    mode: str


def _first_dataset_match(text: str) -> str | None:
    lowered = text.lower()
    for alias, dataset in sorted(DATASET_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered:
            return dataset
    return None


def _parse_memory_budget_mb(text: str) -> tuple[float | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    for alias, hint in GPU_HINTS.items():
        if alias in lowered:
            assumptions.append(str(hint["assumption"]))
            return float(hint["memory_budget_mb"]), assumptions

    match = re.search(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)", lowered)
    if not match:
        return None, assumptions
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"gb", "gib"}:
        return value * 1024.0, assumptions
    return value, assumptions


def _parse_compute_budget(text: str) -> tuple[str | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    for alias, hint in GPU_HINTS.items():
        if alias in lowered:
            return str(hint["compute_budget"]), assumptions
    if any(token in lowered for token in ["old laptop", "very weak", "weak gpu", "cheap gpu", "low-end", "slow machine"]):
        assumptions.append("Interpreted the hardware description as a low compute budget.")
        return "low", assumptions
    if any(token in lowered for token in ["strong gpu", "high-end", "powerful", "server", "desktop gpu"]):
        assumptions.append("Interpreted the hardware description as a high compute budget.")
        return "high", assumptions
    if any(token in lowered for token in ["medium compute", "mid-range", "moderate compute"]):
        return "medium", assumptions
    return None, assumptions


def _parse_forgetting_target(text: str) -> tuple[float | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    explicit = re.search(r"forgetting[^0-9]*(\d+(?:\.\d+)?)", lowered)
    if explicit:
        return float(explicit.group(1)), assumptions
    if any(token in lowered for token in ["very stable", "minimal forgetting", "strict retention", "retention matters", "care a lot about forgetting"]):
        assumptions.append("Mapped strong retention language to a strict acceptable-forgetting target.")
        return 10.0, assumptions
    if any(token in lowered for token in ["balanced", "some forgetting", "moderate forgetting", "reasonable retention"]):
        assumptions.append("Mapped balanced retention language to a moderate acceptable-forgetting target.")
        return 25.0, assumptions
    if any(token in lowered for token in ["forgetting is okay", "can tolerate forgetting", "lenient retention", "i don't mind forgetting"]):
        assumptions.append("Mapped lenient retention language to a flexible acceptable-forgetting target.")
        return 45.0, assumptions
    return None, assumptions


def _parse_similarity(text: str) -> tuple[str | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    if any(token in lowered for token in ["very similar", "closely related", "same kind of tasks", "high similarity"]):
        return "high", assumptions
    if any(token in lowered for token in ["different tasks", "very different", "unrelated", "low similarity"]):
        return "low", assumptions
    if any(token in lowered for token in ["similarity medium", "moderately similar", "somewhat similar"]):
        return "medium", assumptions
    return None, assumptions


def _parse_joint_allowed(text: str) -> tuple[bool | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    if any(token in lowered for token in ["can retrain", "allow retraining", "joint retraining is okay", "overnight retraining is fine", "can retrain overnight"]):
        assumptions.append("Interpreted the description as allowing joint retraining.")
        return True, assumptions
    if any(token in lowered for token in ["cannot retrain", "can't retrain", "do not retrain", "don't want retraining", "no joint retraining"]):
        assumptions.append("Interpreted the description as disallowing joint retraining.")
        return False, assumptions
    return None, assumptions


def infer_settings_from_text(
    text: str,
    *,
    current_request: RecommendationRequest | None = None,
) -> InferredSettingsResult:
    base_request = current_request or RecommendationRequest(
        dataset="split_mini_imagenet",
        memory_budget_mb=256.0,
        compute_budget="medium",
        acceptable_forgetting=20.0,
        task_similarity="medium",
        joint_retraining_allowed=False,
    )

    assumptions: list[str] = []

    dataset = _first_dataset_match(text) or base_request.dataset
    memory_budget_mb, mem_assumptions = _parse_memory_budget_mb(text)
    compute_budget, compute_assumptions = _parse_compute_budget(text)
    acceptable_forgetting, forgetting_assumptions = _parse_forgetting_target(text)
    task_similarity, similarity_assumptions = _parse_similarity(text)
    joint_allowed, joint_assumptions = _parse_joint_allowed(text)

    assumptions.extend(mem_assumptions)
    assumptions.extend(compute_assumptions)
    assumptions.extend(forgetting_assumptions)
    assumptions.extend(similarity_assumptions)
    assumptions.extend(joint_assumptions)

    request = RecommendationRequest(
        dataset=dataset,
        memory_budget_mb=float(memory_budget_mb if memory_budget_mb is not None else base_request.memory_budget_mb),
        compute_budget=str(compute_budget or base_request.compute_budget),
        acceptable_forgetting=float(
            acceptable_forgetting if acceptable_forgetting is not None else (base_request.acceptable_forgetting or 20.0)
        ),
        task_similarity=str(task_similarity or base_request.task_similarity),
        joint_retraining_allowed=bool(
            base_request.joint_retraining_allowed if joint_allowed is None else joint_allowed
        ),
    )

    if dataset == base_request.dataset and current_request is not None and _first_dataset_match(text) is None:
        assumptions.append("Kept the current dataset because the description did not name a different benchmark.")
    if memory_budget_mb is None:
        assumptions.append("Kept the current memory budget because no explicit or inferable memory signal was found.")
    if compute_budget is None:
        assumptions.append("Kept the current compute budget because the description did not clearly imply low, medium, or high compute.")
    if acceptable_forgetting is None:
        assumptions.append("Kept the current acceptable-forgetting target because the retention preference was not explicit enough to remap safely.")
    if task_similarity is None:
        assumptions.append("Kept the current task-similarity setting because the description did not clearly indicate task relatedness.")
    if joint_allowed is None:
        assumptions.append("Kept the current joint-retraining setting because retraining tolerance was not clearly stated.")

    return InferredSettingsResult(
        request=request,
        assumptions=tuple(assumptions),
        parsed_from_text=text,
        requires_confirmation=True,
        mode="heuristic",
    )
