"""Elastic Weight Consolidation (EWC) — regularisation baseline.

Protects important weights from previous tasks by adding a Fisher-weighted
L2 penalty to the loss function.

Reference:
    Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in
    neural networks. PNAS 114(13). https://arxiv.org/abs/1612.00796

Loss:
    L(θ) = L_task(θ) + (λ/2) Σ_i F_i (θ_i − θ*_i)²

where F_i is the diagonal Fisher Information for parameter i and θ* are
the optimal weights after the previous task.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod


class EWC(BaseCLMethod):
    """Elastic Weight Consolidation.

    Config keys:
        ``ewc_lambda`` (float, default 100): regularisation strength.
        ``fisher_samples`` (int, default 200): samples used to estimate Fisher.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.ewc_lambda     = cfg.get("ewc_lambda", 100.0)
        self.fisher_samples = cfg.get("fisher_samples", 200)

        # Lists accumulate across tasks.
        self._old_params: List[Dict[str, torch.Tensor]] = []
        self._fishers:    List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    def _compute_fisher(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher Information via sample gradients."""
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.eval()
        n_samples = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                logits = self.model(x)
                log_probs = F.log_softmax(logits, dim=1)
                # Use the model's own predictions as pseudo-labels.
                pseudo_y = logits.argmax(1).detach()
                loss = F.nll_loss(log_probs, pseudo_y)

            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            n_samples += x.size(0)
            if n_samples >= self.fisher_samples:
                break

        normaliser = max(n_samples, 1)
        return {n: f / normaliser for n, f in fisher.items()}

    # ------------------------------------------------------------------
    def _ewc_penalty(self) -> torch.Tensor:
        """Compute the EWC regularisation term across all previous tasks."""
        if not self._old_params:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for old_params, fisher in zip(self._old_params, self._fishers):
            for n, p in self.model.named_parameters():
                if n in old_params and p.requires_grad:
                    old = old_params[n]
                    fish = fisher[n]
                    if p.shape != old.shape:
                        # Handle expanded classifier head by penalizing only
                        # the overlapping (old) slice.
                        if p.dim() == 2 and old.dim() == 2 and p.size(1) == old.size(1):
                            p_slice = p[: old.size(0)]
                            fish = fish[: old.size(0)]
                        elif p.dim() == 1 and old.dim() == 1:
                            p_slice = p[: old.size(0)]
                            fish = fish[: old.size(0)]
                        else:
                            # Skip parameters that don't align.
                            continue
                    else:
                        p_slice = p
                    penalty += (fish * (p_slice - old).pow(2)).sum()
        return (self.ewc_lambda / 2) * penalty

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
            reg     = self._ewc_penalty()
            loss    = ce_loss + reg

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Snapshot weights and compute Fisher after training ends."""
        # Store current optimal params.
        self._old_params.append(
            {n: p.detach().clone()
             for n, p in self.model.named_parameters()
             if p.requires_grad}
        )
        # Estimate Fisher.
        self._fishers.append(self._compute_fisher(train_loader))
        self.model.train()

    def _method_state(self) -> Dict[str, Any]:
        return {
            "old_params": self._old_params,
            "fishers": self._fishers,
        }

    def _load_method_state(self, state: Dict[str, Any]) -> None:
        self._old_params = state.get("old_params", [])
        self._fishers = state.get("fishers", [])
