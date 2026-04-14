"""Progress & Compress — Architecture + EWC + Distillation hybrid.

A two-phase framework for scalable continual learning:
- **Progress phase**: an "online" network learns the new task using the
  current backbone, with distillation from a frozen "knowledge base" copy
  to preserve prior knowledge.
- **Compress phase**: after each task the online network's knowledge is
  distilled INTO the knowledge base, and EWC is applied to protect the
  knowledge base's important weights.

Reference:
    Schwarz et al. (2018) — Progress & Compress: A Scalable Framework
    for Continual Learning. ICML. https://arxiv.org/abs/1805.06370

This implementation uses a single shared backbone with two linear heads:
one "active" (learning) and one "kb" (knowledge base, partially protected).
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod


class ProgressCompress(BaseCLMethod):
    """Progress & Compress with shared backbone.

    Config keys:
        ``pc_distill_w``    (float, default 1.0): KD weight during progress phase.
        ``pc_ewc_lambda``   (float, default 100): EWC lambda on KB during compress.
        ``pc_compress_epochs`` (int, default 3): compress phase epochs per task.
        ``fisher_samples``  (int,   default 200).
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        # Knowledge base = copy of current model, frozen during progress phase.
        self.kb: nn.Module = copy.deepcopy(model)
        self.kb.to(self.device)
        for p in self.kb.parameters():
            p.requires_grad_(False)
        self.kb.eval()

        self.distill_w       = cfg.get("pc_distill_w",      1.0)
        self.ewc_lambda      = cfg.get("pc_ewc_lambda",     100.0)
        self.compress_epochs = cfg.get("pc_compress_epochs", 3)
        self.fisher_samp     = cfg.get("fisher_samples",    200)

        self._kb_old_params: Dict[str, torch.Tensor] = {}
        self._kb_fisher:     Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_new_classes: int,
        n_classes_total: Optional[int] = None,
    ) -> None:
        # Expand head on the active model; also expand KB to match.
        super().before_task(task_id, train_loader, n_new_classes, n_classes_total)
        kb_expand = self.model.n_classes - self.kb.n_classes
        if kb_expand > 0:
            self.kb.expand(kb_expand)
        # Ensure KB stays on the same device after any head expansion.
        self.kb.to(self.device)
        for p in self.kb.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _distill_from_kb(
        self, active_logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            kb_logits = self.kb(x)
        old_n = kb_logits.size(1)
        if old_n == 0:
            return torch.tensor(0.0, device=self.device)
        return F.kl_div(
            F.log_softmax(active_logits[:, :old_n], dim=1),
            F.softmax(kb_logits, dim=1),
            reduction="batchmean",
        )

    def _ewc_kb_penalty(self) -> torch.Tensor:
        if not self._kb_old_params:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.kb.named_parameters():
            if n in self._kb_old_params and n in self._kb_fisher:
                f = self._kb_fisher[n]
                o = self._kb_old_params[n]
                if p.shape == o.shape and p.shape == f.shape:
                    penalty += (
                        f
                        * (p - o).pow(2)
                    ).sum()
                else:
                    # Handle head expansion by penalizing only overlapping dims.
                    slices = tuple(
                        slice(0, min(p_dim, o_dim, f_dim))
                        for p_dim, o_dim, f_dim in zip(p.shape, o.shape, f.shape)
                    )
                    penalty += (
                        f[slices]
                        * (p[slices] - o[slices]).pow(2)
                    ).sum()
        return (self.ewc_lambda / 2) * penalty

    # ------------------------------------------------------------------
    # Progress phase
    # ------------------------------------------------------------------
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            logits      = self.model(x)
            ce_loss     = self._ce_loss(logits, y)
            distill_loss = self._distill_from_kb(logits, x)
            loss        = ce_loss + self.distill_w * distill_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    # ------------------------------------------------------------------
    # Compress phase
    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Distill active model into KB, then compute Fisher on KB."""
        # Snapshot KB before compress for EWC protection.
        self._kb_old_params = {
            n: p.detach().clone()
            for n, p in self.kb.named_parameters()
        }

        # Temporarily unfreeze KB for compress training.
        for p in self.kb.parameters():
            p.requires_grad_(True)
        kb_optim = torch.optim.SGD(
            self.kb.parameters(),
            lr=self.cfg.get("lr", 0.03) * 0.1,
            momentum=0.9,
        )

        self.kb.train()
        for _ in range(self.compress_epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                kb_optim.zero_grad()
                with torch.no_grad():
                    teacher_logits = self.model(x)      # active model
                student_logits = self.kb(x)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1),
                    reduction="batchmean",
                )
                ewc_pen = self._ewc_kb_penalty()
                (distill_loss + ewc_pen).backward()
                kb_optim.step()

        # Refreeze KB.
        for p in self.kb.parameters():
            p.requires_grad_(False)
        self.kb.eval()

        # Estimate Fisher on KB.
        # Enable grads temporarily to estimate Fisher properly.
        for p in self.kb.parameters():
            p.requires_grad_(True)
        self._kb_fisher = {}
        for n, p in self.kb.named_parameters():
            self._kb_fisher[n] = torch.zeros_like(p)
        n_seen = 0
        self.kb.train()
        for x, y in train_loader:
            x = x.to(self.device)
            self.kb.zero_grad()
            logits = self.kb(x)
            F.nll_loss(F.log_softmax(logits, dim=1), logits.argmax(1).detach()).backward()
            for n, p in self.kb.named_parameters():
                if p.grad is not None:
                    self._kb_fisher[n] += p.grad.detach().pow(2)
            n_seen += x.size(0)
            if n_seen >= self.fisher_samp:
                break
        for n in self._kb_fisher:
            self._kb_fisher[n] /= max(n_seen, 1)

        self.kb.eval()
        for p in self.kb.parameters():
            p.requires_grad_(False)
        self.model.train()

    def _method_state(self) -> Dict[str, Any]:
        return {
            "kb_state": self.kb.state_dict(),
            "kb_n_classes": self.kb.n_classes,
            "kb_old_params": self._kb_old_params,
            "kb_fisher": self._kb_fisher,
        }

    def _load_method_state(self, state: Dict[str, Any]) -> None:
        kb_n_classes = state.get("kb_n_classes", 0)
        kb_expand = kb_n_classes - self.kb.n_classes
        if kb_expand > 0:
            self.kb.expand(kb_expand)
        kb_state = state.get("kb_state")
        if kb_state is not None:
            self.kb.load_state_dict(kb_state)
        self.kb.to(self.device)
        self.kb.eval()
        for p in self.kb.parameters():
            p.requires_grad_(False)
        self._kb_old_params = state.get("kb_old_params", {})
        self._kb_fisher = state.get("kb_fisher", {})
