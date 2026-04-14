"""Model registry — build the full CLModel (backbone + expandable head) by name."""

from .resnet import ResNet8, SlimResNet18, get_backbone
from .classifier_head import ExpandableLinearHead, CLModel
from .vit_small import ViTSmallBackbone, get_vit_small
from .convnext_tiny import ConvNeXtTinyBackbone, get_convnext_tiny


_FEATURE_DIMS = {
    "resnet8":       64,
    "slim_resnet18": 256,
    "vit_small":     384,
    "vit_small_patch16_224": 384,
    "convnext_tiny": 768,
}


def get_model(
    name: str,
    in_channels: int = 3,
    pretrained: bool = False,
    image_size: int | None = None,
) -> CLModel:
    """Return a :class:`CLModel` (backbone + empty expandable head).

    Call ``model.expand(n_classes)`` before training each task.

    Args:
        name:        ``'resnet8'``, ``'slim_resnet18'``, ``'vit_small'``,
                     ``'vit_small_patch16_224'``, or ``'convnext_tiny'``.
        in_channels: Input channels (1 for MNIST, 3 otherwise).
        pretrained:  Load pretrained weights for supported timm backbones.
        image_size:  Optional target image size for ViT-style models.
    """
    if name in {"vit_small", "vit_small_patch16_224"}:
        backbone = get_vit_small(
            pretrained=pretrained,
            image_size=image_size or 224,
        )
    elif name == "convnext_tiny":
        backbone = get_convnext_tiny(pretrained=pretrained)
    else:
        backbone = get_backbone(name, in_channels=in_channels)

    feature_dim = _FEATURE_DIMS[name]
    return CLModel(backbone, feature_dim)


__all__ = [
    "ResNet8",
    "SlimResNet18",
    "ViTSmallBackbone",
    "ConvNeXtTinyBackbone",
    "ExpandableLinearHead",
    "CLModel",
    "get_model",
    "get_backbone",
    "get_vit_small",
    "get_convnext_tiny",
]
