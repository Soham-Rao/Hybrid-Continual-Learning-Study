"""A-GEM + Knowledge Distillation — Novel hybrid method.

Combines gradient projection (A-GEM) with knowledge distillation from a
frozen model snapshot.  A-GEM ensures gradient updates do not hurt old
tasks at the gradient level; distillation provides a soft-label signal
to preserve output distributions.

Motivation: A-GEM alone may allow slow drift in output distributions even
when gradients are not explicitly harmful.  Distillation adds a direct
distributional constraint on the output space.

This is an **original hybrid** implemented for this study.

Loss per step:
    L = L_CE(x, y)  + λ_d · L_KD(snapshot_logits, current_logits)

+ A-GEM gradient projection applied BEFORE the optimizer step.
"""

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class AGEM_Distill(BaseCLMethod):
    """A-GEM + Knowledge Distillation (novel hybrid).

    Config keys:
        ``buffer_size``    (int,   default 200): episodic memory capacity.
        ``agem_mem_batch`` (int,   default 64):  memory gradient batch size.
        ``distill_lambda`` (float, default 1.0): distillation loss weight.
        ``distill_temp``   (float, default 2.0): KD softmax temperature.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer       = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.mem_batch    = cfg.get("agem_mem_batch", 64)
        self.distill_lam  = cfg.get("distill_lambda", 1.0)
        self.distill_temp = cfg.get("distill_temp",   2.0)
        self._teacher: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_new_classes: int,
        n_classes_total: Optional[int] = None,
    ) -> None:
        if task_id > 0:
            self._teacher = copy.deepcopy(self.model)
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            self._teacher.eval()
        super().before_task(task_id, train_loader, n_new_classes, n_classes_total)

    # ------------------------------------------------------------------
    def _distillation_loss(
        self, logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        if self._teacher is None:
            return torch.tensor(0.0, device=self.device)
        T    = self.distill_temp
        old_n = self._teacher.n_classes
        with torch.no_grad():
            t_out = self._teacher(x)
        return F.kl_div(
            F.log_softmax(logits[:, :old_n] / T, dim=1),
            F.softmax(t_out / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

    # ------------------------------------------------------------------
    def _memory_grad_vector(self) -> torch.Tensor:
        """Compute the A-GEM memory gradient without reusing the AMP scaler."""
        if self.buffer.is_empty():
            return torch.empty(0, device=self.device)

        mem = self.buffer.sample(self.mem_batch, self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        with self.autocast(enabled=False):
            m_logits = self.model(mem["x"])
            m_loss = F.cross_entropy(m_logits, mem["y"])
        grads = torch.autograd.grad(m_loss, params, allow_unused=True)
        flat = [g.reshape(-1) for g in grads if g is not None]
        if not flat:
            return torch.empty(0, device=self.device)
        return torch.cat(flat).detach()

    def _project_gradient(self, g_cur: torch.Tensor) -> torch.Tensor:
        """A-GEM gradient projection onto memory constraint."""
        if self.buffer.is_empty():
            return g_cur
        g_mem = self._memory_grad_vector()
        if g_mem.numel() == 0:
            return g_cur

        dot = (g_cur * g_mem).sum()
        mem_norm = (g_mem * g_mem).sum()
        if dot < 0 and mem_norm > 0:
            return g_cur - (dot / mem_norm) * g_mem
        return g_cur

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
            logits      = self.model(x)
            ce_loss     = self._ce_loss(logits, y)
            distill_loss = self._distillation_loss(logits, x)
            loss        = ce_loss + self.distill_lam * distill_loss

        self.scaler.scale(loss).backward()
        if self.fp16:
            self.scaler.unscale_(self.optimizer)

        g_cur = self._grad_vector().detach().clone()
        if not self.buffer.is_empty():
            g_cur = self._project_gradient(g_cur)
        self._write_grad_vector(g_cur)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        n_per_task = max(1, self.buffer.capacity // (task_id + 1))
        collected  = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            take  = min(n_per_task - collected, x.size(0))
            self.buffer.add(x[:take], y[:take], task_id)
            collected += take
            if collected >= n_per_task:
                break
