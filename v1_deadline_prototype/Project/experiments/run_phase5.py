"""Run the Phase 5 analysis, recommendation engine, and report artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.phase5 import run_phase5_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 aggregation and recommendation analysis.")
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parents[2] / "results"),
        help="Repository results root containing epoch_1 and phase4 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "analysis" / "phase5"),
        help="Output directory for Phase 5 analysis artifacts.",
    )
    args = parser.parse_args()

    results = run_phase5_pipeline(Path(args.results_root), Path(args.output_dir))

    master = results["master"]
    summary = results["summary"]
    cases = results["cases"]
    print("Phase 5 complete.")
    print(f"  master rows: {len(master)}")
    print(f"  summary rows: {len(summary)}")
    print(f"  recommendation case rows: {len(cases)}")
    print(f"  outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
