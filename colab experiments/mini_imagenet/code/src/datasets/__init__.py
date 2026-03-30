"""Dataset registry — import datasets by name from a single location."""

from .base_dataset import BaseCLDataset
from .permuted_mnist import PermutedMNIST
from .split_cifar10 import SplitCIFAR10
from .split_cifar100 import SplitCIFAR100
from .split_mini_imagenet import SplitMiniImageNet
from .seq_tiny_imagenet import SeqTinyImageNet

_REGISTRY = {
    "permuted_mnist":      PermutedMNIST,
    "split_cifar10":       SplitCIFAR10,
    "split_cifar100":      SplitCIFAR100,
    "split_mini_imagenet": SplitMiniImageNet,
    "seq_tiny_imagenet":   SeqTinyImageNet,
}


def get_dataset(name: str, **kwargs) -> BaseCLDataset:
    """Factory function: return a dataset instance by registry key.

    Args:
        name:    One of ``permuted_mnist``, ``split_cifar10``,
                 ``split_cifar100``, ``split_mini_imagenet``,
                 ``seq_tiny_imagenet``.
        **kwargs: Forwarded to the dataset constructor.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseCLDataset",
    "PermutedMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
    "SplitMiniImageNet",
    "SeqTinyImageNet",
    "get_dataset",
]
