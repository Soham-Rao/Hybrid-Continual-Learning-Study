"""iCaRL — Replay + Distillation + Nearest-Mean Classifier hybrid.

Three combined strategies:
1. **Herding exemplar replay**: selects the samples closest to the class
   mean in feature space as remembered exemplars.
2. **Knowledge distillation**: LwF-style distillation from the model
   before head expansion preserves old class representations.
3. **Nearest-Mean Classifier (NMC)**: at test time, classifies by finding
   the stored class prototype (mean feature vector) nearest to the query.

Reference:
    Rebuffi et al. (2017) — iCaRL: Incremental Classifier and
    Representation Learning. CVPR. https://arxiv.org/abs/1611.07725
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_method import BaseCLMethod, ReplayBuffer


class iCaRL(BaseCLMethod):
    """Incremental Classifier and Representation Learning.

    Config keys:
        ``buffer_size``       (int,   default 2000): total exemplar budget.
        ``icarl_distill_w``   (float, default 1.0): distillation weight.
        ``icarl_temp``        (float, default 2.0): KD temperature.
        ``use_nmc``           (bool,  default True): use NMC at test time.
    """

    def __init__(self, model, cfg, device) -> None:
        super().__init__(model, cfg, device)
        self.buffer       = ReplayBuffer(cfg.get("buffer_size", 2000), "herding")
        self.distill_w    = cfg.get("icarl_distill_w", 1.0)
        self.temp         = cfg.get("icarl_temp",      2.0)
        self.use_nmc      = cfg.get("use_nmc",         True)

        self._teacher: Optional[nn.Module] = None
        # Class means for NMC: {class_id → mean_feature_vector}
        self.class_means: Dict[int, torch.Tensor] = {}

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
        T = self.temp
        old_n = self._teacher.n_classes
        with torch.no_grad():
            teacher_out = self._teacher(x)
        student_old  = logits[:, :old_n] / T
        teacher_soft = F.softmax(teacher_out / T, dim=1)
        return F.kl_div(
            F.log_softmax(student_old, dim=1),
            teacher_soft,
            reduction="batchmean",
        ) * (T ** 2)

    # ------------------------------------------------------------------
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Combine new data with herding exemplars.
        if not self.buffer.is_empty():
            mem  = self.buffer.sample(x.size(0), self.device)
            x_all = torch.cat([x, mem["x"]], dim=0)
            y_all = torch.cat([y, mem["y"]], dim=0)
        else:
            x_all, y_all = x, y

        with self.autocast():
            logits      = self.model(x_all)
            ce_loss     = self._ce_loss(logits[:x.size(0)], y)
            distill_loss = self._distillation_loss(logits, x_all)
            loss        = ce_loss + self.distill_w * distill_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Update herding exemplars and class prototype means."""
        # Collect all features and raw images for this task.
        self.model.eval()
        all_features: List[torch.Tensor] = []
        all_x:        List[torch.Tensor] = []
        all_y:        List[torch.Tensor] = []

        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(self.device)
                feats = self.model.get_features(x)
                all_features.append(feats.cpu())
                all_x.append(x.cpu())
                all_y.append(y.cpu())

        features = torch.cat(all_features)
        xs       = torch.cat(all_x)
        ys       = torch.cat(all_y)

        # Herding exemplar selection.
        n_per_class = max(1, self.buffer.capacity // self.model.n_classes)
        self.buffer.prune_per_class(n_per_class)
        self.buffer.add_task_exemplars(features, xs, ys, task_id, n_per_class)

        # Recompute class means from the actual stored exemplar set.
        self.class_means = {}
        if self.buffer._storage:
            buf_x = torch.stack([entry["x"] for entry in self.buffer._storage]).to(self.device)
            buf_y = torch.stack([entry["y"] for entry in self.buffer._storage]).to(self.device)
            with torch.no_grad():
                buf_features = self.model.get_features(buf_x)
            for cls in buf_y.unique().tolist():
                mask = (buf_y == cls)
                self.class_means[int(cls)] = buf_features[mask].mean(0).detach()

        self.model.train()

    # ------------------------------------------------------------------
    def predict_nmc(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest-Mean Classifier prediction (replaces softmax argmax).

        Compares query features against all stored class prototypes and
        returns the index of the nearest prototype.
        """
        self.model.eval()
        with torch.no_grad():
            feats = self.model.get_features(x)           # (B, D)

        if not self.class_means:
            return self.model(x).argmax(1)

        classes  = sorted(self.class_means.keys())
        means    = torch.stack([self.class_means[c] for c in classes])  # (C, D)
        # Euclidean squared distances: (B, C)
        dists = (feats.unsqueeze(1) - means.unsqueeze(0)).pow(2).sum(-1)
        pred_local = dists.argmin(1)                      # index in `classes`
        return torch.tensor([classes[i] for i in pred_local.tolist()], device=x.device)

    def _method_state(self) -> Dict[str, Any]:
        return {"class_means": self.class_means}

    def _load_method_state(self, state: Dict[str, Any]) -> None:
        self.class_means = state.get("class_means", {})
