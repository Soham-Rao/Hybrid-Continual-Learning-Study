"""Extended DER (X-DER) — Replay + Distillation + Buffer Revision hybrid.

Extends DER with two additions:
1. **Buffer revision**: after each task, stored logits are updated via a
   forward pass so they always reflect the CURRENT model (not stale ones).
2. **Future-class preparation**: a secondary CE loss on buffer samples
   against their hard labels to ensure old-class outputs stay calibrated.

Reference:
    Boschini et al. (2022) — Class-Incremental Continual Learning into
    the eXtended DER-verse. IEEE TPAMI.
    https://arxiv.org/abs/2201.00766

Loss:
    L = L_CE(x_new, y_new)
      + α · MSE(f(x_mem), z_mem)         ← dark logit matching
      + β · L_CE(f(x_mem), y_mem)        ← hard-label calibration
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class XDER(BaseCLMethod):
    """eXtended Dark Experience Replay.

    Config keys:
        ``buffer_size``   (int,   default 200).
        ``der_alpha``     (float, default 0.5): dark logit loss weight.
        ``xder_beta``     (float, default 0.5): hard-label calibration weight.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer    = ReplayBuffer(cfg.get("buffer_size", 200), "reservoir")
        self.der_alpha = cfg.get("der_alpha",   0.5)
        self.xder_beta = cfg.get("xder_beta",   0.5)

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

            dark_loss = torch.tensor(0.0, device=self.device)
            cal_loss  = torch.tensor(0.0, device=self.device)

            if not self.buffer.is_empty():
                mem = self.buffer.sample(x.size(0), self.device)

                buf_logits_now = self.model(mem["x"])
                stored_dim     = mem["logits"].size(1)

                # Dark logit matching (mask to true stored dims per sample).
                if mem.get("logit_sizes") is not None:
                    mask = (
                        torch.arange(stored_dim, device=self.device)[None, :]
                        < mem["logit_sizes"][:, None]
                    )
                    dark_loss = F.mse_loss(
                        buf_logits_now[:, :stored_dim].float()[mask],
                        mem["logits"].float()[mask],
                    )
                else:
                    dark_loss = F.mse_loss(
                        buf_logits_now[:, :stored_dim].float(),
                        mem["logits"].float(),
                    )
                # Hard-label calibration.
                cal_loss = F.cross_entropy(buf_logits_now, mem["y"])

            loss = ce_loss + self.der_alpha * dark_loss + self.xder_beta * cal_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Store with logits.
        with torch.no_grad():
            with self.autocast():
                stored_logits = self.model(x)
        self.buffer.add(x, y, task_id, logits=stored_logits.detach().float())

        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Revise all stored logits to reflect the current model state."""
        self.buffer.update_logits(self.model, self.device)
