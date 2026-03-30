"""Core Continual Learning training harness.

Runs the task-sequential training loop and fills the accuracy matrix
A[t][i] = accuracy on task i's test set after training on task t.

All 4 CL metrics (AA, F, BWT, FWT) are derived from this matrix.

The trainer is method-agnostic — it calls the three method hooks:
    method.before_task(task_id, train_loader, n_new_classes)
    method.observe(x, y, task_id)
    method.after_task(task_id, train_loader)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.base_dataset import BaseCLDataset
from ..methods.base_method import BaseCLMethod
from ..methods.hybrid.icarl import iCaRL
from ..metrics.continual_metrics import compute_all_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint
from ..utils.logger import RunLogger


class CLTrainer:
    """Task-sequential continual learning trainer.

    Args:
        method:  An instantiated :class:`BaseCLMethod`.
        dataset: A :class:`BaseCLDataset` providing task loaders.
        cfg:     Full experiment configuration dict.
        logger:  :class:`RunLogger` instance.
        device:  Torch device.
    """

    def __init__(
        self,
        method:  BaseCLMethod,
        dataset: BaseCLDataset,
        cfg:     Dict[str, Any],
        logger:  RunLogger,
        device:  torch.device,
    ) -> None:
        self.method  = method
        self.dataset = dataset
        self.cfg     = cfg
        self.logger  = logger
        self.device  = device

        self.n_tasks    = dataset.n_tasks
        self.n_epochs   = cfg.get("n_epochs", 1)
        self.batch_size = cfg.get("batch_size", 32)

        # Accuracy matrix: acc_matrix[t][i] = acc on task i after training task t.
        # NaN means "not yet seen".
        self.acc_matrix = np.full((self.n_tasks, self.n_tasks), np.nan)

        # Baseline acc on each task BEFORE learning it (for FWT).
        self.baseline_acc: List[float] = []

        # Per-task wall-clock times.
        self.task_times: List[float] = []

        self._checkpoint_dir = cfg.get("checkpoint_dir", "results/checkpoints")
        self._run_name       = cfg.get("run_name", "run")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def train(self, start_task: int = 0) -> Dict[str, Any]:
        """Run the full continual learning sequence.

        Args:
            start_task: Resume from this task ID (used for Colab checkpoint
                        recovery — 0 means start fresh).

        Returns:
            Dictionary with ``acc_matrix``, ``metrics``, and ``task_times``.
        """
        all_test_loaders = self.dataset.get_all_test_loaders()

        for task_id in range(start_task, self.n_tasks):
            train_loader, test_loader = self.dataset.get_task_loaders(task_id)
            n_new = self.dataset.n_classes_per_task

            self.logger.print(
                f"\n{'='*60}\n"
                f" Task {task_id + 1}/{self.n_tasks}  "
                f"| Classes {self.dataset.task_class_range(task_id)}"
                f"\n{'='*60}"
            )

            # ── Prepare method for new task ────────────────────────────
            self.method.before_task(
                task_id,
                train_loader,
                n_new,
                n_classes_total=self.dataset.n_classes_total,
            )

            # ── Baseline (chance) accuracy for FWT reference ───────────
            baseline = self._chance_baseline(task_id)
            self.baseline_acc.append(baseline)
            self.logger.print(
                f"Task {task_id} baseline (chance): {baseline:.2f}%"
            )

            # ── Zero-shot accuracy BEFORE learning new task ────────────
            # This populates A[task_id-1, task_id] for FWT.
            if task_id > 0:
                zero_shot = self._evaluate_task(test_loader, task_id)
                if np.isnan(self.acc_matrix[task_id - 1][task_id]):
                    self.acc_matrix[task_id - 1][task_id] = zero_shot
                self.logger.print(
                    f"Task {task_id} zero-shot (pre-train): {zero_shot:.2f}%"
                )

            # ── Training epochs ────────────────────────────────────────
            t0 = time.time()
            for epoch in range(self.n_epochs):
                epoch_loss = self._train_epoch(train_loader, task_id)
                self.logger.log({
                    "task": task_id,
                    "epoch": epoch,
                    "train_loss": round(epoch_loss, 4),
                })

            task_time = time.time() - t0
            self.task_times.append(task_time)

            # ── Post-task hook ─────────────────────────────────────────
            self.method.after_task(task_id, train_loader)

            # ── Evaluate on ALL tasks seen so far ──────────────────────
            self._evaluate_all(all_test_loaders, task_id)

            # Log per-task summary row.
            self.logger.log({
                "task":       task_id,
                "epoch":      "END",
                "time_sec":   round(task_time, 1),
                **{f"acc_task{i}": round(float(self.acc_matrix[task_id][i]), 2)
                   for i in range(task_id + 1)},
            })

            # ── Checkpoint ────────────────────────────────────────────
            self._save_ckpt(task_id)

        # ── Final metrics ──────────────────────────────────────────────
        metrics = compute_all_metrics(
            self.acc_matrix, self.baseline_acc, self.n_tasks
        )
        self.logger.print("\n" + self._metrics_str(metrics))
        return {
            "acc_matrix":  self.acc_matrix,
            "metrics":     metrics,
            "task_times":  self.task_times,
        }

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _train_epoch(
        self, train_loader: DataLoader, task_id: int
    ) -> float:
        """One epoch of training — returns mean loss."""
        self.method.model.train()
        total_loss = 0.0
        n_batches  = 0
        pbar = tqdm(
            train_loader,
            desc=f"  Task {task_id} training",
            leave=False,
            disable=self.cfg.get("disable_tqdm", False),
        )
        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = self.method.observe(x, y, task_id)
            total_loss += loss
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _chance_baseline(self, task_id: int) -> float:
        """Chance-level accuracy for the current task.

        For Class-IL, the head grows as tasks arrive, so chance is
        100 / (#classes_seen_so_far). For Domain-IL, classes are shared.
        """
        if self.dataset.scenario == "class-il":
            n_seen = (task_id + 1) * self.dataset.n_classes_per_task
            n_seen = min(n_seen, self.dataset.n_classes_total)
        else:
            n_seen = self.dataset.n_classes_total
        return 100.0 / max(n_seen, 1)

    def _evaluate_task(self, loader: DataLoader, task_id: int) -> float:
        """Evaluate the model on one task's test set (Class-IL: all heads active).

        For iCaRL, uses the Nearest-Mean Classifier path.
        """
        self.method.model.eval()
        correct = 0
        total   = 0

        use_nmc = (
            isinstance(self.method, iCaRL)
            and self.method.use_nmc
            and bool(self.method.class_means)
        )

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if use_nmc:
                    pred = self.method.predict_nmc(x)
                else:
                    with torch.cuda.amp.autocast(enabled=self.cfg.get("fp16", False)):
                        logits = self.method.model(x)
                    pred = logits.argmax(1)

                correct += (pred == y).sum().item()
                total   += y.size(0)

        self.method.model.train()
        return 100.0 * correct / max(total, 1)

    def _evaluate_all(
        self, all_test_loaders: List[DataLoader], current_task: int
    ) -> None:
        """Fill row ``current_task`` of the accuracy matrix."""
        for i in range(current_task + 1):
            self.acc_matrix[current_task][i] = self._evaluate_task(
                all_test_loaders[i], i
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_ckpt(self, task_id: int) -> None:
        if self.cfg.get("disable_checkpoints", False):
            return
        path = (
            f"{self._checkpoint_dir}/"
            f"{self._run_name}_task{task_id}.pt"
        )
        buf = self.method.get_buffer()
        save_checkpoint(
            {
                "task_id":      task_id,
                "model_state":  self.method.model.state_dict(),
                "acc_matrix":   self.acc_matrix,
                "baseline_acc": self.baseline_acc,
                "task_times":   self.task_times,
                "buffer":       buf._storage if buf else None,
            },
            path,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _metrics_str(metrics: Dict[str, float]) -> str:
        lines = ["Final Metrics:"]
        for k, v in metrics.items():
            lines.append(f"  {k:<20} {v:+.4f}")
        return "\n".join(lines)
