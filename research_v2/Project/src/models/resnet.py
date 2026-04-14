"""Custom ResNet implementations optimised for continual learning.

Models:
- ``ResNet8``       — tiny model for fast debugging on MNIST/CIFAR
- ``SlimResNet18``  — half-width ResNet-18 (~2M params, main local model)

Both expose a ``feature_dim`` attribute and output raw feature vectors
(not class logits). The classification head lives in ``classifier_head.py``
so it can be expanded incrementally as new classes arrive.
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Standard ResNet basic block (no bottleneck) with optional projection."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# ResNet-8 — compact model for MNIST / quick ablations
# ---------------------------------------------------------------------------
class ResNet8(nn.Module):
    """Tiny ResNet for fast experiments on 28×28 or 32×32 inputs.

    Architecture: Conv + [16, 16, 32, 64] residual blocks + GAP.
    Feature dimension: 64.
    Approx parameters: ~80K.
    """

    feature_dim: int = 64

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(16, 16, stride=1)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return x.flatten(1)                         # (B, 64)


# ---------------------------------------------------------------------------
# Slim ResNet-18 — main local model (~2M params)
# ---------------------------------------------------------------------------
class SlimResNet18(nn.Module):
    """ResNet-18 with width multiplier 0.5 for memory-efficient continual learning.

    Channel widths: [32, 64, 128, 256] (vs. standard [64,128,256,512]).
    Feature dimension: 256.
    Approx parameters: ~2M.
    Compatible with any input ≥ 32×32.
    """

    feature_dim: int = 256

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Stem — no max-pool for 32×32 inputs (CIFAR-sized).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32,  32,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(32,  64,  n_blocks=2, stride=2)
        self.layer3 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x.flatten(1)                         # (B, 256)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKBONES = {
    "resnet8":        lambda in_ch: ResNet8(in_channels=in_ch),
    "slim_resnet18":  lambda in_ch: SlimResNet18(in_channels=in_ch),
}


def get_backbone(name: str, in_channels: int = 3) -> nn.Module:
    """Return an un-initialised backbone by name.

    Args:
        name:        ``'resnet8'`` or ``'slim_resnet18'``.
        in_channels: Input channels (1 for MNIST, 3 for RGB).
    """
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(_BACKBONES)}")
    return _BACKBONES[name](in_channels)
