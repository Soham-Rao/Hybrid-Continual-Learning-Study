"""Logging utility: writes metrics to a CSV file and echoes to stdout."""

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Logs per-row metric dictionaries to a CSV file and the console.

    Usage::

        logger = RunLogger("results/raw", "der_cifar100_seed42")
        logger.log({"task": 0, "epoch": 5, "train_loss": 0.23, "val_acc": 88.1})
    """

    def __init__(self, log_dir: str, run_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self.log_path = self.log_dir / f"{run_name}.log"
        self._header_written = False
        self._fieldnames: Optional[list] = None

    # ------------------------------------------------------------------
    def log(self, data: Dict[str, Any]) -> None:
        """Append one row of metrics."""
        if not self._header_written:
            self._fieldnames = list(data.keys())
            with open(self.csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(data)
            self._header_written = True
        else:
            with open(self.csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writerow(data)

        # Console + plain-text log.
        ts = time.strftime("%H:%M:%S")
        line = "  ".join(f"{k}={v}" for k, v in data.items())
        print(f"[{ts}] {line}")
        with open(self.log_path, "a") as fh:
            fh.write(f"[{ts}] {line}\n")

    # ------------------------------------------------------------------
    def print(self, message: str) -> None:
        """Write a free-form message (no CSV row)."""
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {message}")
        with open(self.log_path, "a") as fh:
            fh.write(f"[{ts}] {message}\n")
