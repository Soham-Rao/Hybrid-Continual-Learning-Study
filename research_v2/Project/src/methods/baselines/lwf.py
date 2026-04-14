"""Learning without Forgetting (LwF) — distillation baseline.

Preserves old task outputs by distilling from a snapshot of the model
taken BEFORE the new task begins, using only the new task's data.
No stored replay data needed.

Reference:
    Li & Hoiem (2017) — Learning without Forgetting.
    IEEE TPAMI. https://arxiv.org/abs/1606.09282

Loss:
    L = L_CE(new_task) + λ · L_KD(old_snapshot_logits, current_logits)
"""

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod


class LwF(BaseCLMethod):
    """Learning without Forgetting.

    Config keys:
        ``lwf_lambda``  (float, default 1.0):  distillation weight.
        ``lwf_temp``    (float, default 2.0):  softmax temperature.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.lwf_lambda = cfg.get("lwf_lambda",  1.0)
        self.lwf_temp   = cfg.get("lwf_temp",    2.0)
        self._teacher: Optional[nn.Module] = None   # frozen snapshot

    # ------------------------------------------------------------------
    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_new_classes: int,
        n_classes_total: Optional[int] = None,
    ) -> None:
        """Snapshot current model BEFORE head expansion."""
        if task_id > 0:
            # Snapshot BEFORE expansion so it has old dims.
            self._teacher = copy.deepcopy(self.model)
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            self._teacher.eval()

        # Expand head and rebuild optimizer.
        super().before_task(task_id, train_loader, n_new_classes, n_classes_total)

    # ------------------------------------------------------------------
    def _distillation_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        if self._teacher is None:
            return torch.tensor(0.0, device=self.device)

        T = self.lwf_temp
        old_n = self._teacher.n_classes         # classes the teacher knows

        with torch.no_grad():
            teacher_logits = self._teacher(x)   # (B, old_n)

        # Match only the old class outputs.
        student_old = student_logits[:, :old_n] / T
        teacher_old = teacher_logits[:, :old_n] / T

        loss_kd = F.kl_div(
            F.log_softmax(student_old, dim=1),
            F.softmax(teacher_old, dim=1),
            reduction="batchmean",
        ) * (T ** 2)
        return loss_kd

    # ------------------------------------------------------------------
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            logits  = self.model(x)
            ce_loss = self._ce_loss(logits, y)
            kd_loss = self._distillation_loss(logits, x)
            loss    = ce_loss + self.lwf_lambda * kd_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()
