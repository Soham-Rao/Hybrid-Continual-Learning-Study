"""Abstract base class for all continual learning methods + ReplayBuffer.

Every CL method implements the same three hooks so the trainer is
method-agnostic::

    method.before_task(task_id, train_loader, dataset)
    loss = method.observe(x, y, task_id)
    method.after_task(task_id, train_loader)

The :class:`ReplayBuffer` supports both **reservoir sampling** (uniform
random, O(1) per step) and **herding** (select exemplars closest to the
class mean in feature space — used by iCaRL).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ===========================================================================
# Replay buffer
# ===========================================================================
class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Supports two storage policies:
    - ``'reservoir'``: uniform random replacement (default, O(1)).
    - ``'herding'``:  exemplar selection by mean-feature proximity
                      (call :meth:`add_task_exemplars` instead of :meth:`add`).

    Stored tensors (CPU to avoid GPU VRAM usage):
    - ``x``      (torch.Tensor) — input image
    - ``y``      (torch.long)   — global class label
    - ``logits`` (torch.Tensor | None) — model output at storage time (DER/X-DER)
    - ``task_id``(int)          — which task this sample belongs to

    Args:
        capacity:   Maximum total number of samples.
        strategy:   ``'reservoir'`` or ``'herding'``.
    """

    def __init__(self, capacity: int, strategy: str = "reservoir") -> None:
        if strategy not in ("reservoir", "herding"):
            raise ValueError(f"Unknown buffer strategy '{strategy}'")
        self.capacity  = capacity
        self.strategy  = strategy
        self._storage: List[Dict[str, Any]] = []
        self._n_seen   = 0               # total samples ever offered to reservoir

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._storage)

    def is_empty(self) -> bool:
        return len(self._storage) == 0

    # ------------------------------------------------------------------
    # Reservoir sampling
    # ------------------------------------------------------------------
    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a mini-batch to the buffer using reservoir sampling.

        Each sample in the batch is processed independently.
        """
        if self.strategy != "reservoir":
            raise RuntimeError("Use add_task_exemplars() for herding strategy.")
        x, y = x.detach().cpu(), y.detach().cpu()
        logits_cpu = logits.detach().cpu() if logits is not None else None

        for i in range(x.size(0)):
            sample = {
                "x":       x[i],
                "y":       y[i],
                "task_id": task_id,
                "logits":  logits_cpu[i] if logits_cpu is not None else None,
                "logit_size": (
                    int(logits_cpu[i].numel()) if logits_cpu is not None else None
                ),
            }
            self._n_seen += 1
            if len(self._storage) < self.capacity:
                self._storage.append(sample)
            else:
                # Reservoir: replace a random existing entry.
                j = random.randrange(self._n_seen)
                if j < self.capacity:
                    self._storage[j] = sample

    # ------------------------------------------------------------------
    # Herding — iCaRL exemplar selection
    # ------------------------------------------------------------------
    def add_task_exemplars(
        self,
        features: torch.Tensor,        # (N, D) — pre-computed, CPU
        x_raw:   torch.Tensor,         # (N, C, H, W) — original images, CPU
        y_raw:   torch.Tensor,         # (N,) — global labels, CPU
        task_id: int,
        n_exemplars: int,
    ) -> None:
        """Select *n_exemplars* samples per class via herding for iCaRL.

        Herding greedily picks exemplars whose running mean is closest to
        the class mean in feature space.
        """
        classes = y_raw.unique().tolist()
        for cls in classes:
            mask  = (y_raw == cls)
            feats = features[mask]           # (Nc, D)
            xs    = x_raw[mask]
            ys    = y_raw[mask]

            class_mean = feats.mean(0)       # (D,)
            selected: List[int] = []
            running_sum = torch.zeros_like(class_mean)

            for _ in range(min(n_exemplars, feats.size(0))):
                # Greedily pick the sample that moves running mean closest.
                candidates = (running_sum.unsqueeze(0) + feats) / (len(selected) + 1)
                dists = (candidates - class_mean).pow(2).sum(1)
                chosen = int(dists.argmin().item())
                selected.append(chosen)
                running_sum += feats[chosen]

            for idx in selected:
                sample = {
                    "x":       xs[idx],
                    "y":       ys[idx],
                    "task_id": task_id,
                    "logits":  None,
                }
                if len(self._storage) < self.capacity:
                    self._storage.append(sample)
                else:
                    # Replace oldest entry of the same class (circular).
                    for s_idx, s in enumerate(self._storage):
                        if s["y"].item() == cls:
                            self._storage[s_idx] = sample
                            break

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Sample *batch_size* items uniformly; returns tensors on *device*.

        Returns dict with keys ``x``, ``y``, ``logits`` (may be None if
        logits were never stored), ``task_ids``.
        """
        size    = min(batch_size, len(self._storage))
        indices = random.sample(range(len(self._storage)), size)
        batch   = [self._storage[i] for i in indices]

        xs       = torch.stack([b["x"] for b in batch]).to(device)
        ys       = torch.stack([b["y"] for b in batch]).to(device)
        task_ids = torch.tensor([b["task_id"] for b in batch], device=device)

        logits_list = [b["logits"] for b in batch]
        if logits_list[0] is None or any(l is None for l in logits_list):
            logits = None
            logit_sizes = None
        else:
            logit_sizes = torch.tensor(
                [b["logit_size"] for b in batch], device=device
            )
            max_dim = int(logit_sizes.max().item())
            padded = torch.zeros(size, max_dim, dtype=logits_list[0].dtype)
            for i, l in enumerate(logits_list):
                padded[i, : l.numel()] = l
            logits = padded.to(device)

        return {
            "x": xs,
            "y": ys,
            "logits": logits,
            "logit_sizes": logit_sizes,
            "task_ids": task_ids,
        }

    # ------------------------------------------------------------------
    def update_logits(self, model: nn.Module, device: torch.device) -> None:
        """Re-compute and overwrite stored logits (used by X-DER).

        Runs a forward pass through ``model`` on every stored sample.
        """
        model.eval()
        with torch.no_grad():
            for entry in self._storage:
                x = entry["x"].unsqueeze(0).to(device)
                entry["logits"] = model(x).squeeze(0).cpu()
                entry["logit_size"] = int(entry["logits"].numel())
        model.train()

    def get_all_data(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ALL stored (x, y) for joint training / AGEM."""
        xs = torch.stack([e["x"] for e in self._storage]).to(device)
        ys = torch.stack([e["y"] for e in self._storage]).to(device)
        return xs, ys

    def clear(self) -> None:
        self._storage.clear()
        self._n_seen = 0


# ===========================================================================
# Abstract base method
# ===========================================================================
class BaseCLMethod(ABC):
    """Abstract interface: every CL method implements these three hooks.

    Args:
        model:      :class:`~src.models.classifier_head.CLModel` instance.
        cfg:        Configuration dict from YAML (hyperparam access).
        device:     Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model  = model
        self.cfg    = cfg
        self.device = device

        lr       = cfg.get("lr", 0.03)
        momentum = cfg.get("momentum", 0.9)
        wd       = cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("fp16", False))
        self.fp16   = cfg.get("fp16", False)

    # ------------------------------------------------------------------
    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_new_classes: int,
        n_classes_total: Optional[int] = None,
    ) -> None:
        """Called BEFORE training on *task_id*.

        Default implementation expands the classifier head up to
        ``n_classes_total`` (when provided).

        This handles both:
        - **Class-IL**: expand by ``n_new_classes`` each task until full.
        - **Domain-IL**: expand only once (task 0), then stop.

        Always call ``super().before_task(...)`` in subclasses.
        """
        # Expand head up to total classes (if known); otherwise fall back to
        # expanding by n_new_classes each task.
        if n_classes_total is not None:
            remaining = n_classes_total - self.model.n_classes
            if remaining > 0:
                self.model.expand(min(n_new_classes, remaining))
        else:
            if self.model.n_classes == 0:
                self.model.expand(n_new_classes)
            else:
                self.model.expand(n_new_classes)

        # Ensure any newly created head parameters land on the correct device.
        self.model.to(self.device)

        # Rebuild optimizer to include any newly created head parameters.
        lr       = self.cfg.get("lr", 0.03)
        momentum = self.cfg.get("momentum", 0.9)
        wd       = self.cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )


    @abstractmethod
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        """Process ONE mini-batch: forward pass, loss, backward, step.

        Returns:
            Scalar loss value (Python float).
        """

    def after_task(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> None:
        """Called AFTER training on *task_id* (before evaluation).

        Override to store buffer exemplars, consolidate EWC, etc.
        Default: no-op.
        """

    # ------------------------------------------------------------------
    def get_buffer(self) -> Optional[ReplayBuffer]:
        """Return the replay buffer if this method uses one, else None."""
        return getattr(self, "buffer", None)

    def _ce_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels)
