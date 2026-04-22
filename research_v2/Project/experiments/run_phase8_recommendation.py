"""Run the Phase 8 recommendation artifact pipeline on top of Phase 7 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation import generate_phase8_artifacts
from src.utils.paths import epoch_label, resolve_results_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 8 v2 recommendation artifact generation.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch count to consume (default 1).")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional results root override. Defaults to the workspace results/ directory.",
    )
    args = parser.parse_args()

    results_root = resolve_results_root(args.results_root)
    analysis_dir = results_root / "analysis" / epoch_label(args.epoch)
    outputs = generate_phase8_artifacts(analysis_dir)
    print(f"Saved Phase 8 recommendation artifacts under -> {analysis_dir}")
    print(f"  profile rows: {len(outputs['profiles'])}")
    print(f"  recommendation case rows: {len(outputs['cases'])}")


if __name__ == "__main__":
    main()
