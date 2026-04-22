"""Compact method knowledge cards for grounded copilot explanations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodCard:
    method: str
    label: str
    family: str
    mechanism: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    works_well_when: tuple[str, ...]


METHOD_CARDS: dict[str, MethodCard] = {
    "fine_tune": MethodCard(
        method="fine_tune",
        label="Fine-Tune",
        family="baseline",
        mechanism="Sequentially updates one model without replay, regularization, or distillation support.",
        strengths=("simple", "cheap on memory", "fast to run"),
        weaknesses=("usually forgets heavily", "weak retention under task drift"),
        works_well_when=("you only need a weak lower baseline", "speed matters more than retention"),
    ),
    "joint_training": MethodCard(
        method="joint_training",
        label="Joint Training",
        family="baseline",
        mechanism="Replays all observed data and acts as a strong upper-bound style baseline inside the sequential protocol.",
        strengths=("very strong accuracy", "very low forgetting", "useful as an upper bound"),
        weaknesses=("high memory proxy", "less realistic when joint retraining is disallowed"),
        works_well_when=("joint retraining is acceptable", "the goal is best empirical performance over efficiency"),
    ),
    "ewc": MethodCard(
        method="ewc",
        label="EWC",
        family="baseline",
        mechanism="Constrains important weights to preserve older tasks without storing replay memory.",
        strengths=("memory-light", "regularization-based retention"),
        weaknesses=("can underperform replay on harder shifts", "sensitive to penalty strength"),
        works_well_when=("memory is tight", "you want regularization instead of replay storage"),
    ),
    "agem": MethodCard(
        method="agem",
        label="A-GEM",
        family="baseline",
        mechanism="Uses replay examples to project updates away from directions that would hurt past tasks.",
        strengths=("explicit interference control", "conceptually clean replay baseline"),
        weaknesses=("often weaker than stronger replay hybrids", "can be compute-heavier than plain regularization"),
        works_well_when=("memory is available but you still want a simpler replay method", "task interference is the main concern"),
    ),
    "lwf": MethodCard(
        method="lwf",
        label="LwF",
        family="baseline",
        mechanism="Uses distillation from the previous model to preserve output behavior on old tasks.",
        strengths=("no replay storage", "conceptually simple distillation baseline"),
        weaknesses=("can drift when old-task signals are too weak", "less robust than stronger hybrids"),
        works_well_when=("replay is undesirable", "you want a lightweight distillation baseline"),
    ),
    "der": MethodCard(
        method="der",
        label="DER",
        family="hybrid",
        mechanism="Combines replay with distillation-style targets to preserve richer behavior from prior states.",
        strengths=("strong retention", "good replay-plus-distillation tradeoff"),
        weaknesses=("uses memory", "not the lightest option"),
        works_well_when=("you want a strong replay method without joint retraining", "retention matters a lot"),
    ),
    "xder": MethodCard(
        method="xder",
        label="X-DER",
        family="hybrid",
        mechanism="Extends replay-plus-distillation with stronger correction and revision behavior for hard continual settings.",
        strengths=("often very strong accuracy-retention balance", "good on tougher benchmarks"),
        weaknesses=("more complex", "heavier than simpler baselines"),
        works_well_when=("you can afford a stronger hybrid", "you want a high-performing non-joint option"),
    ),
    "icarl": MethodCard(
        method="icarl",
        label="iCaRL",
        family="hybrid",
        mechanism="Uses exemplar replay with distillation-style retention and class-incremental bias handling.",
        strengths=("strong class-incremental pedigree", "good exemplar-based retention"),
        weaknesses=("memory dependent", "method assumptions can be less universal"),
        works_well_when=("class-incremental behavior matters", "you want a proven exemplar method"),
    ),
    "er_ewc": MethodCard(
        method="er_ewc",
        label="ER+EWC",
        family="hybrid",
        mechanism="Combines replay with regularization so memory and weight-protection work together.",
        strengths=("balanced hybrid", "combines replay strength with regularization stability"),
        weaknesses=("more knobs to tune", "heavier than either component alone"),
        works_well_when=("you want hybrid retention pressure", "simple replay alone feels too weak"),
    ),
    "progress_compress": MethodCard(
        method="progress_compress",
        label="Progress & Compress",
        family="hybrid",
        mechanism="Separates plastic and stable pathways so new learning and old knowledge are partially decoupled.",
        strengths=("strong conceptual isolation of new versus old knowledge", "can reduce interference"),
        weaknesses=("architecturally heavier", "not always the most efficient practical choice"),
        works_well_when=("interference control matters more than simplicity", "you can tolerate architectural complexity"),
    ),
    "agem_distill": MethodCard(
        method="agem_distill",
        label="A-GEM+Distill",
        family="hybrid",
        mechanism="Combines A-GEM style replay constraints with distillation to preserve both gradients and past behavior.",
        strengths=("more retention pressure than plain A-GEM", "useful replay-distill compromise"),
        weaknesses=("still not always the strongest hybrid", "heavier than plain A-GEM"),
        works_well_when=("you want a replay method stronger than plain A-GEM", "retention needs more than gradient projection alone"),
    ),
    "si_der": MethodCard(
        method="si_der",
        label="SI-DER",
        family="hybrid",
        mechanism="Combines replay, distillation, and regularization-style stability signals in one hybrid method.",
        strengths=("strong balanced hybrid", "often performs well across multiple constraints"),
        weaknesses=("heavier than simpler baselines", "can be less interpretable than single-mechanism methods"),
        works_well_when=("you want a strong all-round non-joint choice", "multiple forms of retention pressure help"),
    ),
}


def get_method_card(method: str) -> MethodCard:
    method = str(method)
    return METHOD_CARDS.get(
        method,
        MethodCard(
            method=method,
            label=method,
            family="other",
            mechanism="Uses a continual-learning strategy represented in the current study artifacts.",
            strengths=("grounded by study evidence",),
            weaknesses=("no dedicated method card is available yet",),
            works_well_when=("the empirical results support it under the current constraints",),
        ),
    )
