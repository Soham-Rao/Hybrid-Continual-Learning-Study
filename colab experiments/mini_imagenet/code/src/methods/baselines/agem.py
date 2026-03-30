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
    def _project_gradient(self) -> None:
        """Project current grad onto the constraint g̃ · g_mem ≥ 0."""
        if self.buffer.is_empty():
            return

        mem = self.buffer.sample(self.mem_batch, self.device)
        mem_x, mem_y = mem["x"], mem["y"]

        # Compute memory gradient g_mem.
        self.model.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.fp16):
            mem_logits = self.model(mem_x)
            mem_loss   = F.cross_entropy(mem_logits, mem_y)
        mem_loss.backward()
        g_mem = torch.cat([
            p.grad.view(-1) for p in self.model.parameters()
            if p.grad is not None
        ])

        # Collect current task gradient g_cur (already computed).
        g_cur = torch.cat([
            p.grad.view(-1) for p in self.model.parameters()
            if p.grad is not None
        ])

        dot = (g_cur * g_mem).sum()
        if dot < 0:
            # Violation: project g_cur onto the halfspace.
            g_proj = g_cur - (dot / (g_mem * g_mem).sum()) * g_mem
            # Write projected gradients back.
            offset = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    n = p.grad.numel()
                    p.grad.copy_(g_proj[offset: offset + n].view_as(p.grad))
                    offset += n

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
        with torch.cuda.amp.autocast(enabled=self.fp16):
            logits = self.model(x)
            loss   = self._ce_loss(logits, y)
        loss.backward()

        # Project gradient if we have past memory.
        if not self.buffer.is_empty():
            self._project_gradient()

        self.optimizer.step()
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
