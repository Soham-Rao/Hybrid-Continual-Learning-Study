"""ViT-Small wrapper for Colab experiments.

Uses the ``timm`` library.  Gradient checkpointing and FP16 (autocast) are
enabled by default to fit within ~15 GB VRAM on a Colab T4.

Intended usage: import this file in the Colab notebooks; do NOT run locally
unless you have ≥ 8 GB VRAM free.

Requirements (Colab cell)::

    !pip install timm -q
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class ViTSmallBackbone(nn.Module):
    """``vit_small_patch16_224`` backbone via timm with gradient checkpointing.

    Outputs a flat feature vector of dimension 384 (ViT-Small hidden dim).
    The pretrained ImageNet weights are loaded by default to give CL
    methods a better starting point.

    Args:
        pretrained:          Load ImageNet-21k pretrained weights (default True).
        grad_checkpointing:  Enable gradient checkpointing to cut activation
                             memory by ~50 % (small throughput cost).
        image_size:          Input spatial resolution. 224 for Mini/Tiny-ImageNet.
    """

    feature_dim: int = 384

    def __init__(
        self,
        pretrained: bool = True,
        grad_checkpointing: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ViT-Small. Install with: pip install timm"
            )

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # remove timm's own head; we use CLModel's head
            img_size=image_size,
        )
        if grad_checkpointing:
            self.vit.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token features: ``(B, 384)``."""
        return self.vit(x)                          # timm returns (B, feature_dim) when num_classes=0


# ---------------------------------------------------------------------------
# Convenience builder (mirrors get_backbone in resnet.py)
# ---------------------------------------------------------------------------
def get_vit_small(
    pretrained: bool = True,
    grad_checkpointing: bool = True,
    image_size: int = 224,
) -> ViTSmallBackbone:
    """Return a ViT-Small backbone ready to plug into :class:`CLModel`."""
    return ViTSmallBackbone(
        pretrained=pretrained,
        grad_checkpointing=grad_checkpointing,
        image_size=image_size,
    )
