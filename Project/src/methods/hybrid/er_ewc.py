"""ER-Reservoir + EWC — Replay + Regularisation hybrid.

Combines:
- **Experience Replay** (reservoir sampling) to revisit past data.
- **EWC** Fisher-weighted regularisation to additionally protect
  important weights from being overwritten.

The combination addresses a weakness of each alone: replay alone can
miss subtle weight shifts, and EWC alone has no explicit data mechanism.

Reference:
    ER: Chaudhry et al. (2019) — On Tiny Episodic Memories in CL.
        https://arxiv.org/abs/1902.10486
    EWC: Kirkpatrick et al. (2017). https://arxiv.org/abs/1612.00796
"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class ER_EWC(BaseCLMethod):
    """Experience Replay + Elastic Weight Consolidation (hybrid).

    Config keys:
        ``buffer_size``     (int,   default 200).
        ``ewc_lambda``      (float, default 50.0): EWC regularisation weight.
        ``fisher_samples``  (int,   default 200): samples for Fisher estimate.
        ``er_replay_ratio`` (float, default 0.5): fraction of batch from buffer.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer       = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.ewc_lambda   = cfg.get("ewc_lambda",      50.0)
        self.fisher_samp  = cfg.get("fisher_samples",  200)
        self.replay_ratio = cfg.get("er_replay_ratio", 0.5)

        self._old_params: List[Dict[str, torch.Tensor]] = []
        self._fishers:    List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    def _ewc_penalty(self) -> torch.Tensor:
        if not self._old_params:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for old_p, fisher in zip(self._old_params, self._fishers):
            for n, p in self.model.named_parameters():
                if n in old_p and p.requires_grad:
                    if p.shape == old_p[n].shape:
                        penalty += (fisher[n] * (p - old_p[n]).pow(2)).sum()
                    else:
                        # Handle expanded heads by penalizing only the overlap.
                        slices = tuple(
                            slice(0, min(p_dim, o_dim))
                            for p_dim, o_dim in zip(p.shape, old_p[n].shape)
                        )
                        penalty += (
                            fisher[n][slices]
                            * (p[slices] - old_p[n][slices]).pow(2)
                        ).sum()
        return (self.ewc_lambda / 2) * penalty

    def _compute_fisher(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        n_seen = 0
        for x, y in loader:
            x = x.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, logits.argmax(1).detach())
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            n_seen += x.size(0)
            if n_seen >= self.fisher_samp:
                break
        return {n: f / max(n_seen, 1) for n, f in fisher.items()}

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
            # Replay mix: append buffer samples to the current batch.
            if not self.buffer.is_empty():
                n_replay = max(1, int(x.size(0) * self.replay_ratio))
                mem      = self.buffer.sample(n_replay, self.device)
                x_all    = torch.cat([x, mem["x"]])
                y_all    = torch.cat([y, mem["y"]])
            else:
                x_all, y_all = x, y

            logits  = self.model(x_all)
            ce_loss = self._ce_loss(logits, y_all)
            ewc_pen = self._ewc_penalty()
            loss    = ce_loss + ewc_pen

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.buffer.add(x, y, task_id)
        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        self._old_params.append(
            {n: p.detach().clone()
             for n, p in self.model.named_parameters() if p.requires_grad}
        )
        self._fishers.append(self._compute_fisher(train_loader))
        self.model.train()
