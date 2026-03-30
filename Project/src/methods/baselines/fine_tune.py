"""Naive fine-tuning baseline (lower bound).

Standard SGD/CE training with no mechanism to prevent forgetting.
This is the worst-case reference: it maximally forgets old tasks.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod


class FineTune(BaseCLMethod):
    """Naive sequential fine-tuning — no replay, no regularisation.

    Serves as the **lower bound** in all comparison tables and plots.
    """

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            logits = self.model(x)
            loss   = self._ce_loss(logits, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()
