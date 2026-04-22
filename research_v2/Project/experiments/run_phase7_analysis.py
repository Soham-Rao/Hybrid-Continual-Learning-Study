"""Run the Phase 7 statistical analysis pipeline for v2 epoch-level results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import run_phase7_pipeline
from src.utils.paths import PROJECT_ROOT, epoch_label, resolve_results_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 v2 statistical analysis.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch count to analyze (default 1).")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional results root override. Defaults to the workspace results/ directory.",
    )
    args = parser.parse_args()

    results_root = resolve_results_root(args.results_root)
    outputs = run_phase7_pipeline(PROJECT_ROOT, results_root, epoch=args.epoch)
    analysis_dir = results_root / "analysis" / epoch_label(args.epoch)
    print(f"Saved Phase 7 analysis under -> {analysis_dir}")
    print(f"  master rows: {len(outputs['master'])}")
    print(f"  summary rows: {len(outputs['summary'])}")
    print(f"  pairwise rows: {len(outputs['pairwise'])}")
    print(f"  effect-size rows: {len(outputs['effect_sizes'])}")


if __name__ == "__main__":
    main()
