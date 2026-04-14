"""ConvNeXt-Tiny backbone wrapper for optional v2 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class ConvNeXtTinyBackbone(nn.Module):
    """``convnext_tiny`` backbone via timm.

    The timm model is created without its classifier head so it returns a
    pooled feature vector directly. ConvNeXt can naturally operate on
    variable image sizes, which makes it a practical optional v2 backbone.
    """

    feature_dim: int = 768

    def __init__(
        self,
        pretrained: bool = False,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ConvNeXt-Tiny. Install with: pip install timm"
            )

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if grad_checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_convnext_tiny(
    pretrained: bool = False,
    grad_checkpointing: bool = True,
) -> ConvNeXtTinyBackbone:
    """Return a ConvNeXt-Tiny backbone ready for :class:`CLModel`."""
    return ConvNeXtTinyBackbone(
        pretrained=pretrained,
        grad_checkpointing=grad_checkpointing,
    )
