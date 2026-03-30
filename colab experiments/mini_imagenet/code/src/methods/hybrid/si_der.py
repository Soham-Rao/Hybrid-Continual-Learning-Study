"""SI + DER — Synaptic Intelligence + Dark Experience Replay (Novel hybrid).

Combines:
- **Synaptic Intelligence (SI)**: online per-weight importance accumulation
  that protects weights that contributed most to past task learning.
- **Dark Experience Replay (DER)**: replays past samples with stored logits
  (soft targets) rather than hard labels.

Motivation: SI's regularisation operates in weight space continuously
during training; DER's replay constrains the output space using rich
distributional information.  Together they provide dual protection:
SI guards the learning trajectory, DER anchors the output distribution.

This is an **original hybrid** for this study.

Reference SI:  Zenke et al. (2017) https://arxiv.org/abs/1703.04200
Reference DER: Buzzega et al. (2020) https://arxiv.org/abs/2004.07211
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class SI_DER(BaseCLMethod):
    """Synaptic Intelligence + Dark Experience Replay (novel hybrid).

    Config keys:
        ``buffer_size``   (int,   default 200).
        ``der_alpha``     (float, default 0.5): DER dark replay weight.
        ``si_lambda``     (float, default 1.0): SI regularisation weight.
        ``si_xi``         (float, default 0.1): SI damping constant.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer    = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.der_alpha = cfg.get("der_alpha", 0.5)
        self.si_lambda = cfg.get("si_lambda", 1.0)
        self.si_xi     = cfg.get("si_xi",     0.1)

        # SI state: per-parameter running contributions and weight snapshots.
        self._si_omega:    Dict[str, torch.Tensor] = {}  # cumulative importance
        self._si_w_start:  Dict[str, torch.Tensor] = {}  # params at task start
        self._si_w_prev:   Dict[str, torch.Tensor] = {}  # params at prev step
        self._si_running:  Dict[str, torch.Tensor] = {}  # path integral accumulator

    # ------------------------------------------------------------------
    def _init_si_task(self) -> None:
        """Snapshot weights at the start of each task for SI."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._si_w_start[n] = p.detach().clone()
                self._si_w_prev[n]  = p.detach().clone()
                self._si_running[n] = torch.zeros_like(p)

    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_new_classes: int,
        n_classes_total: Optional[int] = None,
    ) -> None:
        super().before_task(task_id, train_loader, n_new_classes, n_classes_total)
        self._init_si_task()

    # ------------------------------------------------------------------
    def _accumulate_si(self) -> None:
        """Update SI path integrals after each gradient step."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._si_w_prev:
                delta = p.detach() - self._si_w_prev[n]
                if p.grad is not None:
                    grad = p.grad.detach()
                    if not torch.isfinite(grad).all():
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                    # Contribution: –grad * delta (change in loss / change in weight).
                    self._si_running[n] -= grad * delta
                self._si_w_prev[n] = p.detach().clone()

    def _si_penalty(self) -> torch.Tensor:
        if not self._si_omega:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._si_omega and p.requires_grad:
                omega = self._si_omega[n]
                w0 = self._si_w_start[n]
                if omega.shape == p.shape:
                    penalty += (omega * (p - w0).pow(2)).sum()
                else:
                    # Handle expanded heads by penalizing only overlapping dims.
                    slices = tuple(
                        slice(0, min(o_dim, p_dim))
                        for o_dim, p_dim in zip(omega.shape, p.shape)
                    )
                    penalty += (
                        omega[slices] * (p[slices] - w0[slices]).pow(2)
                    ).sum()
        return (self.si_lambda / 2) * penalty

    def _update_si_omega(self) -> None:
        """At task end, consolidate per-weight importance."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._si_running:
                delta_sq = (p.detach() - self._si_w_start[n]).pow(2) + self.si_xi
                importance = self._si_running[n] / delta_sq
                if not torch.isfinite(importance).all():
                    importance = torch.nan_to_num(
                        importance, nan=0.0, posinf=0.0, neginf=0.0
                    )
                if n in self._si_omega:
                    old = self._si_omega[n]
                    if old.shape == importance.shape:
                        self._si_omega[n] = (
                            old + importance.clamp(min=0)
                        ).detach()
                    else:
                        # Expand old omega to the new shape (new params start at 0).
                        new_omega = torch.zeros_like(importance)
                        slices = tuple(
                            slice(0, min(o_dim, i_dim))
                            for o_dim, i_dim in zip(old.shape, importance.shape)
                        )
                        new_omega[slices] = old[slices]
                        new_omega += importance.clamp(min=0)
                        self._si_omega[n] = new_omega.detach()
                else:
                    self._si_omega[n] = importance.clamp(min=0).detach()

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
            logits  = self.model(x)
            ce_loss = self._ce_loss(logits, y)

            # DER dark replay.
            dark_loss = torch.tensor(0.0, device=self.device)
            if not self.buffer.is_empty():
                mem = self.buffer.sample(x.size(0), self.device)
                buf_out    = self.model(mem["x"])
                stored_dim = mem["logits"].size(1)
                if mem.get("logit_sizes") is not None:
                    mask = (
                        torch.arange(stored_dim, device=self.device)[None, :]
                        < mem["logit_sizes"][:, None]
                    )
                    dark_loss = F.mse_loss(
                        buf_out[:, :stored_dim][mask],
                        mem["logits"][mask],
                    )
                else:
                    dark_loss  = F.mse_loss(buf_out[:, :stored_dim], mem["logits"])

            # SI penalty.
            si_pen = self._si_penalty()
            loss   = ce_loss + self.der_alpha * dark_loss + si_pen

        self.scaler.scale(loss).backward()
        if self.fp16:
            # Unscale grads before using them for SI accumulation.
            self.scaler.unscale_(self.optimizer)
        self._accumulate_si()          # track path integral BEFORE step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Store with logits for DER.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                stored_logits = self.model(x)
        self.buffer.add(x, y, task_id, logits=stored_logits.detach())

        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        self._update_si_omega()
