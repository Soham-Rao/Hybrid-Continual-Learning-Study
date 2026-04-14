"""Path helpers for the v2 research workspace.

These helpers centralize how run outputs are placed inside the new
workspace-level ``results/`` tree. The goal is to keep:

- run artifacts separate from figures
- primary runs separate from ablations
- seed-scoped outputs nested predictably
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


BASELINE_METHODS = {
    "fine_tune",
    "joint_training",
    "ewc",
    "agem",
    "lwf",
}

HYBRID_METHODS = {
    "der",
    "xder",
    "icarl",
    "er_ewc",
    "progress_compress",
    "agem_distill",
    "si_der",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = WORKSPACE_ROOT / "results"


def epoch_label(n_epochs: int) -> str:
    """Convert an integer epoch count to the canonical folder label."""
    return f"epoch_{int(n_epochs)}"


def method_family(method_name: str) -> str:
    """Map a method name to its high-level family directory."""
    if method_name in BASELINE_METHODS:
        return "baselines"
    if method_name in HYBRID_METHODS:
        return "hybrids"
    return "other_methods"


def build_run_name(cfg: Dict[str, Any], seed: int) -> str:
    """Create a stable run identifier for one dataset/method/seed tuple."""
    run_name = f"{cfg['dataset']}_{cfg['method']}"
    run_tag = cfg.get("run_tag")
    if run_tag:
        run_name = f"{run_name}_{run_tag}"
    return f"{run_name}_seed{seed}"


def resolve_data_root(raw_path: str | None) -> Path:
    """Resolve the configured dataset root against the v2 workspace."""
    if not raw_path:
        return PROJECT_ROOT / "data_local"
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "project":
        return WORKSPACE_ROOT / path
    return PROJECT_ROOT / path


def resolve_results_root(raw_path: str | None) -> Path:
    """Resolve the configured results root against the workspace root."""
    if not raw_path:
        return RESULTS_ROOT
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def build_output_layout(cfg: Dict[str, Any], seed: int) -> Dict[str, Path]:
    """Return the canonical output paths for a single run."""
    results_root = resolve_results_root(cfg.get("results_root"))
    epoch_dir = epoch_label(cfg.get("n_epochs", 1))
    dataset = cfg["dataset"]
    method = cfg["method"]
    family = method_family(method)
    seed_dir = f"seed_{seed}"
    run_tag = cfg.get("run_tag")
    method_dir = method if not run_tag else f"{method}__{run_tag}"

    is_ablation = bool(cfg.get("ablation_family")) or cfg.get("result_group") == "ablations"
    if is_ablation:
        ablation_family = cfg.get("ablation_family", "misc")
        run_root = (
            results_root
            / "ablations"
            / epoch_dir
            / dataset
            / ablation_family
            / method_dir
            / seed_dir
        )
        figure_root = (
            results_root
            / "figures"
            / "ablations"
            / epoch_dir
            / dataset
            / ablation_family
            / method_dir
            / seed_dir
        )
    else:
        run_root = (
            results_root
            / "runs"
            / epoch_dir
            / dataset
            / family
            / method_dir
            / seed_dir
        )
        figure_root = (
            results_root
            / "figures"
            / epoch_dir
            / dataset
            / family
            / method_dir
            / seed_dir
        )

    return {
        "run_root": run_root,
        "log_dir": run_root / "logs",
        "metrics_dir": run_root / "metrics",
        "checkpoint_dir": run_root / "checkpoints",
        "artifact_dir": run_root / "artifacts",
        "figure_dir": figure_root,
    }


def prepare_run_config(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Return a copied config enriched with resolved v2 paths."""
    prepared = deepcopy(cfg)
    prepared["seed"] = seed
    prepared["run_name"] = build_run_name(prepared, seed)
    prepared["project_root"] = str(PROJECT_ROOT)
    prepared["workspace_root"] = str(WORKSPACE_ROOT)
    prepared["data_root"] = str(resolve_data_root(prepared.get("data_root")))
    prepared["results_root"] = str(resolve_results_root(prepared.get("results_root")))

    layout = build_output_layout(prepared, seed)
    for key, value in layout.items():
        prepared[key] = str(value)

    return prepared


def metrics_csv_path(cfg: Dict[str, Any], seed: int) -> Path:
    """Return the exact metrics CSV path for a run without creating it."""
    prepared = prepare_run_config(cfg, seed)
    return Path(prepared["metrics_dir"]) / f"{prepared['run_name']}_metrics.csv"
