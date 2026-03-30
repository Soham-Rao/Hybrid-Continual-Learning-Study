"""Expandable multi-class classifier head for Class-Incremental Learning.

In Class-IL the model must distinguish between ALL classes it has ever seen
without knowing the current task ID.  New classes are added by ``expand()``,
which appends fresh output neurons while preserving previously learned weights.

The full model is composed as::

    features = backbone(x)            # (B, feature_dim)
    logits   = classifier_head(features)  # (B, n_classes_seen)

For Domain-IL (Permuted MNIST) the head is created once and never expanded.
"""

import torch
import torch.nn as nn


class ExpandableLinearHead(nn.Module):
    """A linear classifier that grows its output dimension on demand.

    Args:
        feature_dim:      Input feature dimensionality (from backbone).
        initial_classes:  Number of output classes to start with.
                          Use ``0`` to defer until the first ``expand()`` call.
    """

    def __init__(self, feature_dim: int, initial_classes: int = 0) -> None:
        super().__init__()
        self.feature_dim  = feature_dim
        self.n_classes    = initial_classes

        if initial_classes > 0:
            self.weight = nn.Parameter(torch.empty(initial_classes, feature_dim))
            self.bias   = nn.Parameter(torch.zeros(initial_classes))
            nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        else:
            # Deferred allocation — register as buffers so state_dict works.
            self.register_buffer("weight", torch.empty(0, feature_dim))
            self.register_buffer("bias",   torch.empty(0))

    # ------------------------------------------------------------------
    def expand(self, n_new_classes: int) -> None:
        """Append *n_new_classes* fresh output neurons.

        Existing weights are preserved exactly; new neurons are Kaiming-
        initialised.
        """
        if n_new_classes <= 0:
            return

        old_n    = self.n_classes
        new_n    = old_n + n_new_classes
        feat_dim = self.feature_dim

        new_weight = torch.empty(n_new_classes, feat_dim, device=self._device())
        new_bias   = torch.zeros(n_new_classes,          device=self._device())
        nn.init.kaiming_uniform_(new_weight, a=(5 ** 0.5))

        if old_n == 0:
            combined_w = new_weight
            combined_b = new_bias
        else:
            old_w = (
                self.weight.data
                if isinstance(self.weight, nn.Parameter)
                else self.weight
            )
            old_b = (
                self.bias.data
                if isinstance(self.bias, nn.Parameter)
                else self.bias
            )
            combined_w = torch.cat([old_w, new_weight], dim=0)
            combined_b = torch.cat([old_b, new_bias],   dim=0)

        # Replace buffers/params with proper nn.Parameter tensors.
        if hasattr(self, "weight") and not isinstance(self.weight, nn.Parameter):
            del self._buffers["weight"]
            del self._buffers["bias"]

        self.weight = nn.Parameter(combined_w)
        self.bias   = nn.Parameter(combined_b)
        self.n_classes = new_n

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, feature_dim) → (B, n_classes)``."""
        return nn.functional.linear(features, self.weight, self.bias)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __repr__(self) -> str:
        return (
            f"ExpandableLinearHead("
            f"feature_dim={self.feature_dim}, "
            f"n_classes={self.n_classes})"
        )


# ---------------------------------------------------------------------------
# Convenience wrapper: backbone + expandable head in one nn.Module
# ---------------------------------------------------------------------------
class CLModel(nn.Module):
    """Backbone + ExpandableLinearHead packaged as a single model.

    Usage::

        model = CLModel(backbone, feature_dim=256)
        model.expand(n_new_classes=5)          # called before each new task
        logits = model(x)                      # (B, n_classes_seen_so_far)
        features = model.get_features(x)       # (B, feature_dim)
    """

    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = ExpandableLinearHead(feature_dim, initial_classes=0)
        self.feature_dim   = feature_dim

    # ------------------------------------------------------------------
    def expand(self, n_new_classes: int) -> None:
        """Expand the classifier for *n_new_classes* new classes."""
        self.head.expand(n_new_classes)

    @property
    def n_classes(self) -> int:
        return self.head.n_classes

    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (used by Progress & Compress)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)
