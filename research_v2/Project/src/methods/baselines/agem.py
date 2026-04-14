"""Averaged GEM (A-GEM) — replay baseline.

Projects the current task gradient so that the average loss on all past
tasks does not increase.  Uses an episodic memory buffer to compute the
reference gradient.

Reference:
    Chaudhry et al. (2019) — Efficient Lifelong Learning with A-GEM.
    ICLR. https://arxiv.org/abs/1812.00420
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class AGEM(BaseCLMethod):
    """Averaged Gradient Episodic Memory.

    Config keys:
        ``buffer_size``    (int,   default 200): episodic memory capacity.
        ``agem_mem_batch`` (int,   default 64):  batch size for memory gradient.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer     = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.mem_batch  = cfg.get("agem_mem_batch", 64)
        self._task_id   = 0

    # ------------------------------------------------------------------
    def _memory_grad_vector(self) -> torch.Tensor:
        """Compute the reference memory gradient without touching AMP scaler state."""
        if self.buffer.is_empty():
            return torch.empty(0, device=self.device)

        mem = self.buffer.sample(self.mem_batch, self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        with self.autocast(enabled=False):
            mem_logits = self.model(mem["x"])
            mem_loss = F.cross_entropy(mem_logits, mem["y"])
        grads = torch.autograd.grad(mem_loss, params, allow_unused=True)
        flat = [g.reshape(-1) for g in grads if g is not None]
        if not flat:
            return torch.empty(0, device=self.device)
        return torch.cat(flat).detach()

    def _project_gradient(self, g_cur: torch.Tensor) -> torch.Tensor:
        """Project current grad onto the constraint g̃ · g_mem ≥ 0."""
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
        self._task_id = task_id
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Forward on current task.
        with self.autocast():
            logits = self.model(x)
            loss   = self._ce_loss(logits, y)
        self.scaler.scale(loss).backward()
        if self.fp16:
            self.scaler.unscale_(self.optimizer)

        g_cur = self._grad_vector().detach().clone()

        # Project gradient if we have past memory.
        if not self.buffer.is_empty():
            g_cur = self._project_gradient(g_cur)

        self._write_grad_vector(g_cur)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Collect a fixed set of examples for the episodic memory."""
        n_per_task = max(1, self.buffer.capacity // (task_id + 1))
        collected  = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            take = min(n_per_task - collected, x.size(0))
            self.buffer.add(x[:take], y[:take], task_id)
            collected += take
            if collected >= n_per_task:
                break
