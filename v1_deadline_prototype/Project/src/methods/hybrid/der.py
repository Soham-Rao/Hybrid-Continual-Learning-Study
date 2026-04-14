"""Dark Experience Replay (DER) — Replay + Distillation hybrid.

Stores past inputs AND their model logits at the time of storage.
Replays with an MSE loss on logits instead of CE on hard labels — this
preserves richer distributional information (class relationships).

Reference:
    Buzzega et al. (2020) — Dark Experience for General Continual Learning:
    A Strong, Simple Baseline. NeurIPS 33.
    https://arxiv.org/abs/2004.07211

Loss:
    L = L_CE(x_new, y_new)
      + α · MSE(f(x_mem), z_mem)    ← "dark" logit matching
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class DER(BaseCLMethod):
    """Dark Experience Replay.

    Config keys:
        ``buffer_size``  (int,   default 200): replay buffer capacity.
        ``der_alpha``    (float, default 0.5): weight of the dark replay loss.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer    = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.der_alpha = cfg.get("der_alpha", 0.5)

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
            # ── Current task loss ──────────────────────────────────────
            logits  = self.model(x)
            ce_loss = self._ce_loss(logits, y)

            # ── Dark replay loss ───────────────────────────────────────
            dark_loss = torch.tensor(0.0, device=self.device)
            if not self.buffer.is_empty():
                mem = self.buffer.sample(x.size(0), self.device)
                buf_logits_now = self.model(mem["x"])

                # Match only the stored logit dimensions (old class space).
                if mem.get("logit_sizes") is not None:
                    stored_dim = mem["logits"].size(1)
                    mask = (
                        torch.arange(stored_dim, device=self.device)[None, :]
                        < mem["logit_sizes"][:, None]
                    )
                    dark_loss = F.mse_loss(
                        buf_logits_now[:, :stored_dim].float()[mask],
                        mem["logits"].float()[mask],
                    )
                else:
                    stored_dim = mem["logits"].size(1)
                    dark_loss  = F.mse_loss(
                        buf_logits_now[:, :stored_dim].float(),
                        mem["logits"].float(),
                    )

            loss = ce_loss + self.der_alpha * dark_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Store current-task samples WITH their detached logits.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                stored_logits = self.model(x)
        self.buffer.add(x, y, task_id, logits=stored_logits.detach().float())

        return loss.item()
