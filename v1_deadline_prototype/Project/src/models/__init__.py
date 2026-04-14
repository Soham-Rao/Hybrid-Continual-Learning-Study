"""Model registry — build the full CLModel (backbone + expandable head) by name."""

from .resnet import ResNet8, SlimResNet18, get_backbone
from .classifier_head import ExpandableLinearHead, CLModel
from .vit_small import ViTSmallBackbone, get_vit_small


_FEATURE_DIMS = {
    "resnet8":       64,
    "slim_resnet18": 256,
    "vit_small":     384,
}


def get_model(
    name: str,
    in_channels: int = 3,
    pretrained: bool = False,
) -> CLModel:
    """Return a :class:`CLModel` (backbone + empty expandable head).

    Call ``model.expand(n_classes)`` before training each task.

    Args:
        name:        ``'resnet8'``, ``'slim_resnet18'``, or ``'vit_small'``.
        in_channels: Input channels (1 for MNIST, 3 otherwise).
        pretrained:  Load pretrained weights (only valid for ``vit_small``).
    """
    if name == "vit_small":
        backbone = get_vit_small(pretrained=pretrained)
    else:
        backbone = get_backbone(name, in_channels=in_channels)

    feature_dim = _FEATURE_DIMS[name]
    return CLModel(backbone, feature_dim)


__all__ = [
    "ResNet8",
    "SlimResNet18",
    "ViTSmallBackbone",
    "ExpandableLinearHead",
    "CLModel",
    "get_model",
    "get_backbone",
    "get_vit_small",
]
