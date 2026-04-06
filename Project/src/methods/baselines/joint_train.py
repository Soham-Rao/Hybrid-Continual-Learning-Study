"""Joint Training baseline (upper bound).

Accumulates ALL data from every task seen so far and trains from scratch
(well, from current weights) on the full combined dataset on each task.

This sets the performance ceiling: what would happen if we had access to
all past data at every step, with no memory constraint.

⚠️ Only feasible for small datasets (CIFAR-10/100) locally.
For Mini/Tiny-ImageNet limit ``max_samples_per_task`` to keep it tractable.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from ..base_method import BaseCLMethod


class JointTraining(BaseCLMethod):
    """Accumulate all past training data and retrain jointly.

    Training in :meth:`observe` only sees the current task's batch.
    The joint retraining over ALL data happens at the end of each task
    via :meth:`after_task` — which is what distinguishes this from fine-tune.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self._all_x: List[torch.Tensor] = []
        self._all_y: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        """Standard CE step on the current batch (used during after_task replay)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            logits = self.model(x)
            loss   = self._ce_loss(logits, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Collect this task's data and do a full-dataset replay pass."""
        # Gather new task data (CPU tensors).
        max_samples = self.cfg.get("max_samples_per_task")
        collected = 0
        for x, y in train_loader:
            if max_samples is None:
                self._all_x.append(x.cpu())
                self._all_y.append(y.cpu())
                continue

            remaining = max_samples - collected
            if remaining <= 0:
                break
            take = min(remaining, x.size(0))
            self._all_x.append(x[:take].cpu())
            self._all_y.append(y[:take].cpu())
            collected += take

        # Rebuild a TensorDataset over all accumulated data.
        all_x = torch.cat(self._all_x, dim=0)
        all_y = torch.cat(self._all_y, dim=0)

        joint_loader = DataLoader(
            TensorDataset(all_x, all_y),
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=True,
            num_workers=0,
        )

        n_epochs = self.cfg.get("joint_replay_epochs", 1)
        for _ in range(n_epochs):
            for x_b, y_b in joint_loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                self.observe(x_b, y_b, task_id)

    def _method_state(self) -> Dict[str, Any]:
        return {
            "all_x": self._all_x,
            "all_y": self._all_y,
        }

    def _load_method_state(self, state: Dict[str, Any]) -> None:
        self._all_x = state.get("all_x", [])
        self._all_y = state.get("all_y", [])
