"""Local Phase 4/5 runner for Sequential Tiny-ImageNet.

This script keeps the old "top methods only" intent, but uses a local-first
workflow with resume and checkpoint cleanup on success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import run


METHOD_OVERRIDES: Dict[str, Dict] = {
    "fine_tune": {},
    "der": {"buffer_size": 500, "der_alpha": 0.5},
    "xder": {"buffer_size": 500, "der_alpha": 0.5, "xder_beta": 0.5},
    "si_der": {"buffer_size": 500, "der_alpha": 0.5, "si_lambda": 1.0},
    "icarl": {"buffer_size": 2000, "icarl_temp": 2.0, "use_nmc": True},
}


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_methods(raw: str) -> List[str]:
    if raw == "all":
        return list(METHOD_OVERRIDES.keys())
    methods = _parse_csv(raw)
    unknown = sorted(set(methods) - set(METHOD_OVERRIDES))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def _base_cfg(args: argparse.Namespace) -> Dict:
    if args.model == "vit_small":
        batch_size = args.batch_size or 4
        lr = args.lr or 0.001
        pretrained = True
    else:
        batch_size = args.batch_size or 32
        lr = args.lr or 0.03
        pretrained = False

    return {
        "dataset": "seq_tiny_imagenet",
        "data_root": str(Path(args.data_root)),
        "model": args.model,
        "pretrained": pretrained,
        "n_epochs": 1,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "fp16": args.device != "cpu",
        "disable_tqdm": False,
        "disable_checkpoints": False,
        "disable_plots": False,
        "resume": args.resume,
        "cleanup_checkpoints_on_success": args.cleanup_checkpoints,
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "figure_dir": str(Path(args.figure_dir)),
        "log_dir": str(Path(args.log_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sequential Tiny-ImageNet locally.")
    parser.add_argument("--methods", default="all",
                        help="Comma-separated methods or 'all'.")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="vit_small",
                        choices=["slim_resnet18", "vit_small"])
    parser.add_argument("--data-root",
                        default=str(ROOT / "Project" / "data_local" / "tiny-imagenet-200"))
    parser.add_argument("--log-dir",
                        default=str(ROOT / "results" / "phase4" / "local_tiny" / "raw"))
    parser.add_argument("--figure-dir",
                        default=str(ROOT / "results" / "phase4" / "local_tiny" / "figures"))
    parser.add_argument("--checkpoint-dir",
                        default=str(ROOT / "results" / "phase4" / "local_tiny" / "checkpoints"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-checkpoints", dest="cleanup_checkpoints", action="store_false")
    parser.set_defaults(resume=True, cleanup_checkpoints=True)
    args = parser.parse_args()

    methods = _resolve_methods(args.methods)
    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    base_cfg = _base_cfg(args)
    failures: List[str] = []

    for method_name in methods:
        method_cfg = {**base_cfg, **METHOD_OVERRIDES[method_name], "method": method_name}
        for seed in seeds:
            try:
                print(f"\n=== seq_tiny_imagenet | {method_name} | seed={seed} ===")
                run(dict(method_cfg), seed=seed, device=args.device)
            except Exception as exc:  # pragma: no cover - execution script
                failures.append(f"{method_name}:seed{seed}: {exc}")
                print(f"FAILED -> {method_name} seed={seed}: {exc}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nAll requested Tiny-ImageNet runs finished successfully.")


if __name__ == "__main__":
    main()
