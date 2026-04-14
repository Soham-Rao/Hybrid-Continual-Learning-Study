"""Permuted MNIST — Domain-Incremental continual learning benchmark.

Each task applies a **fixed random permutation** to the 784 input pixels,
creating a different input distribution while keeping the 10-class label
space shared across all tasks.  No head expansion is needed.

Reference: Goodfellow et al. (2013) — An empirical investigation of
catastrophic forgetting in gradient-based neural networks.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseCLDataset


# ---------------------------------------------------------------------------
class PermutedMNISTTask(Dataset):
    """Wraps a standard MNIST dataset with a fixed pixel permutation applied."""

    def __init__(
        self,
        mnist_data: Dataset,
        permutation: np.ndarray,
        image_size: int = 28,
        out_channels: int = 1,
    ) -> None:
        self.data = mnist_data
        self.perm = torch.from_numpy(permutation).long()
        self.image_size = image_size
        self.out_channels = out_channels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.data[idx]           # img: (1, 28, 28)
        flat = img.view(-1)                   # (784,)
        permuted = flat[self.perm].view(1, 28, 28)
        if self.image_size != 28:
            permuted = F.interpolate(
                permuted.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.out_channels == 3:
            permuted = permuted.repeat(3, 1, 1)
        return permuted, label


# ---------------------------------------------------------------------------
class PermutedMNIST(BaseCLDataset):
    """10-task Permuted MNIST benchmark (Domain-IL).

    Args:
        root:        Directory where MNIST will be downloaded.
        n_tasks:     Number of permutations / tasks (default 10).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        seed:        RNG seed used to generate permutations so they are
                     reproducible across runs.
    """

    def __init__(
        self,
        root: str = "data",
        n_tasks: int = 10,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 0,
        image_size: int = 28,
        out_channels: int = 1,
    ) -> None:
        super().__init__(root, batch_size, num_workers)
        self._n_tasks = n_tasks
        self._image_size = image_size
        self._out_channels = out_channels

        # Pre-generate all permutations deterministically.
        rng = np.random.default_rng(seed)
        self._permutations: list[np.ndarray] = [
            rng.permutation(784) for _ in range(n_tasks)
        ]

        # Base MNIST (normalised to [-1, 1]).
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self._train_base = datasets.MNIST(root, train=True,  download=True, transform=transform)
        self._test_base  = datasets.MNIST(root, train=False, download=True, transform=transform)

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @property
    def n_classes_per_task(self) -> int:
        return 10   # all 10 digits shared across tasks

    @property
    def n_classes_total(self) -> int:
        return 10

    @property
    def input_size(self) -> Tuple[int, int, int]:
        return (self._out_channels, self._image_size, self._image_size)

    @property
    def scenario(self) -> str:
        return "domain-il"

    # ------------------------------------------------------------------
    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        perm = self._permutations[task_id]
        train_ds = PermutedMNISTTask(
            self._train_base,
            perm,
            image_size=self._image_size,
            out_channels=self._out_channels,
        )
        test_ds  = PermutedMNISTTask(
            self._test_base,
            perm,
            image_size=self._image_size,
            out_channels=self._out_channels,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader
