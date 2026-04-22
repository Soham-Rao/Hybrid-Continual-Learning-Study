"""Structured settings inference and clarification-ready actions for the copilot."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.recommendation.engine import RecommendationRequest
from .knowledge_base import HARDWARE_CARDS, SETTINGS_QUERY_TERMS


DATASET_ALIASES = {
    "split mini-imagenet": "split_mini_imagenet",
    "split mini imagenet": "split_mini_imagenet",
    "mini-imagenet": "split_mini_imagenet",
    "mini imagenet": "split_mini_imagenet",
    "split cifar-100": "split_cifar100",
    "split cifar100": "split_cifar100",
    "cifar-100": "split_cifar100",
    "cifar100": "split_cifar100",
    "split cifar-10": "split_cifar10",
    "split cifar10": "split_cifar10",
    "cifar-10": "split_cifar10",
    "cifar10": "split_cifar10",
    "permuted mnist": "permuted_mnist",
    "pmnist": "permuted_mnist",
    "mnist": "permuted_mnist",
}

OUT_OF_SCOPE_DATASETS = {
    "fashion mnist": "Dataset is outside the exact evaluated scope; any suggestion is only an approximate mapping to the nearest supported benchmark.",
    "tiny imagenet": "Dataset is outside the exact evaluated scope; any suggestion is only an approximate mapping to the nearest supported benchmark.",
    "imagenet": "Full ImageNet is outside the exact evaluated scope; any suggestion is only an approximate mapping to the nearest supported benchmark.",
    "coco": "COCO is outside the exact evaluated scope; any suggestion is only an approximate mapping to the nearest supported benchmark.",
}


def _contains_alias(lowered: str, alias: str) -> bool:
    return re.search(rf"(?<!\\w){re.escape(alias)}(?!\\w)", lowered) is not None

OUT_OF_SCOPE_MODELS = {
    "vit": "Transformer backbone mentions are only partially inside scope because the main compulsory study used slim_resnet18.",
    "convnext": "ConvNeXt is only in the optional backbone scope, not the compulsory primary matrix.",
    "bert": "Language-model backbones are outside the exact evaluated scope of this continual-learning vision study.",
    "yolo": "Object-detection backbones are outside the exact evaluated scope of this study.",
}


@dataclass(frozen=True)
class InferredSettingsResult:
    request: RecommendationRequest
    assumptions: tuple[str, ...]
    clarification_questions: tuple[str, ...]
    scope_notes: tuple[str, ...]
    parsed_from_text: str
    requires_confirmation: bool
    needs_clarification: bool
    mode: str


def looks_like_settings_query(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in SETTINGS_QUERY_TERMS):
        return True
    if any(_contains_alias(lowered, alias) for alias in HARDWARE_CARDS):
        return True
    if any(alias in lowered for alias in DATASET_ALIASES):
        return True
    if re.search(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)", lowered):
        return True
    return False


def _first_dataset_match(text: str) -> str | None:
    lowered = text.lower()
    for alias, dataset in sorted(DATASET_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered:
            return dataset
    return None


def _parse_memory_budget_mb(text: str) -> tuple[float | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    for alias, hint in HARDWARE_CARDS.items():
        if _contains_alias(lowered, alias) and hint.get("memory_budget_mb") is not None:
            assumptions.append(str(hint["assumption"]))
            return float(hint["memory_budget_mb"]), assumptions

    match = re.search(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)", lowered)
    if not match:
        return None, assumptions
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"gb", "gib"}:
        mb_value = value * 1024.0
        if "ram" in lowered and "vram" not in lowered and "memory budget" not in lowered:
            assumptions.append("Mapped system RAM to an approximate method-memory budget; this is a heuristic rather than a directly tested setting.")
            if value <= 8:
                return 1024.0, assumptions
            if value <= 16:
                return 2048.0, assumptions
            if value <= 32:
                return 4096.0, assumptions
            return min(mb_value / 2.0, 8192.0), assumptions
        return mb_value, assumptions
    return value, assumptions


def _parse_compute_budget(text: str) -> tuple[str | None, list[str]]:
    lowered = text.lower()
    assumptions: list[str] = []
    for alias, hint in HARDWARE_CARDS.items():
        if _contains_alias(lowered, alias):
            assumptions.append(str(hint["assumption"]))
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
    if any(token in lowered for token in ["cannot retrain", "can't retrain", "do not retrain", "do not want retraining", "don't want retraining", "no joint retraining"]):
        assumptions.append("Interpreted the description as disallowing joint retraining.")
        return False, assumptions
    return None, assumptions


def _scope_notes(text: str) -> list[str]:
    lowered = text.lower()
    notes: list[str] = []
    if any(_contains_alias(lowered, alias) for alias in HARDWARE_CARDS) or "ram" in lowered:
        notes.append(
            "Hardware-based settings are heuristic estimates derived from the description and are not exact benchmark conditions that were directly tested in the study."
        )
    for alias, note in OUT_OF_SCOPE_DATASETS.items():
        if alias in lowered:
            notes.append(note)
    for alias, note in OUT_OF_SCOPE_MODELS.items():
        if alias in lowered:
            notes.append(note)
    deduped: list[str] = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return deduped


def _clarification_questions(
    *,
    current_request: RecommendationRequest | None,
    dataset_explicit: bool,
    memory_found: bool,
    compute_found: bool,
    forgetting_found: bool,
    similarity_found: bool,
    joint_found: bool,
    limit: int = 3,
) -> tuple[str, ...]:
    questions: list[str] = []
    if current_request is None and not dataset_explicit:
        questions.append("Which benchmark should I anchor to: Permuted MNIST, Split CIFAR-10, Split CIFAR-100, or Split Mini-ImageNet?")
    if not memory_found and not compute_found:
        questions.append("Should I assume a low, medium, or high hardware budget for memory and compute?")
    if not forgetting_found:
        questions.append("Do you want strict retention, a balanced trade-off, or are you okay with more forgetting for speed or simplicity?")
    if not joint_found:
        questions.append("Is joint retraining acceptable, or should I exclude methods that rely on it?")
    if not similarity_found:
        questions.append("Are your tasks closely related, moderately related, or quite different from one another?")
    return tuple(questions[:limit])


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
    dataset_match = _first_dataset_match(text)
    dataset = dataset_match or base_request.dataset
    memory_budget_mb, mem_assumptions = _parse_memory_budget_mb(text)
    compute_budget, compute_assumptions = _parse_compute_budget(text)
    acceptable_forgetting, forgetting_assumptions = _parse_forgetting_target(text)
    task_similarity, similarity_assumptions = _parse_similarity(text)
    joint_allowed, joint_assumptions = _parse_joint_allowed(text)
    scope_notes = _scope_notes(text)

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
        joint_retraining_allowed=bool(base_request.joint_retraining_allowed if joint_allowed is None else joint_allowed),
    )

    if dataset == base_request.dataset and current_request is not None and dataset_match is None:
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

    clarification_questions = _clarification_questions(
        current_request=current_request,
        dataset_explicit=dataset_match is not None,
        memory_found=memory_budget_mb is not None,
        compute_found=compute_budget is not None,
        forgetting_found=acceptable_forgetting is not None,
        similarity_found=task_similarity is not None,
        joint_found=joint_allowed is not None,
    )

    return InferredSettingsResult(
        request=request,
        assumptions=tuple(assumptions),
        clarification_questions=clarification_questions,
        scope_notes=tuple(scope_notes),
        parsed_from_text=text,
        requires_confirmation=True,
        needs_clarification=bool(clarification_questions),
        mode="heuristic",
    )
