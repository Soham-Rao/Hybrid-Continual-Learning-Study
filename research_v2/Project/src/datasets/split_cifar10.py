"""Split CIFAR-10 — Class-Incremental continual learning benchmark.

CIFAR-10's 10 classes are split into **5 sequential tasks of 2 classes each**.
Labels are global (0-9) so the expanding classifier head works correctly.
Task identity is NOT given at test time (Class-IL).

Reference: Used as a standard benchmark in DER, iCaRL, A-GEM papers.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .base_dataset import BaseCLDataset

# CIFAR-10 normalisation constants (per-channel).
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

_N_TASKS            = 5
_CLASSES_PER_TASK   = 2


def _build_transforms(train: bool, image_size: int = 32) -> transforms.Compose:
    if image_size != 32:
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ])
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _class_indices(dataset: Dataset, classes: List[int]) -> List[int]:
    """Return sample indices whose label is in *classes*."""
    targets = np.array(dataset.targets)
    mask = np.isin(targets, classes)
    return np.where(mask)[0].tolist()


class SplitCIFAR10(BaseCLDataset):
    """5-task Split CIFAR-10 benchmark (Class-IL, 2 classes/task).

    Args:
        root:        Directory where CIFAR-10 will be downloaded.
        n_tasks:     Optional override for number of tasks (must divide 10).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        root: str = "data",
        n_tasks: int | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
        image_size: int = 32,
    ) -> None:
        super().__init__(root, batch_size, num_workers)
        total_classes = 10
        if n_tasks is None:
            n_tasks = _N_TASKS
        if total_classes % n_tasks != 0:
            raise ValueError(
                f"n_tasks={n_tasks} must divide {total_classes} for SplitCIFAR10."
            )
        self._n_tasks = n_tasks
        self._classes_per_task = total_classes // n_tasks
        self._task_classes: List[List[int]] = [
            list(range(i * self._classes_per_task, (i + 1) * self._classes_per_task))
            for i in range(self._n_tasks)
        ]
        self._image_size = image_size
        self._train_base = datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=_build_transforms(True, image_size=image_size),
        )
        self._test_base  = datasets.CIFAR10(
            root,
            train=False,
            download=True,
            transform=_build_transforms(False, image_size=image_size),
        )

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @property
    def n_classes_per_task(self) -> int:
        return self._classes_per_task

    @property
    def n_classes_total(self) -> int:
        return 10

    @property
    def input_size(self) -> Tuple[int, int, int]:
        return (3, self._image_size, self._image_size)

    @property
    def scenario(self) -> str:
        return "class-il"

    def class_names(self) -> List[str]:
        return self._train_base.classes

    # ------------------------------------------------------------------
    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        classes = self._task_classes[task_id]

        train_idx = _class_indices(self._train_base, classes)
        test_idx  = _class_indices(self._test_base,  classes)

        train_loader = DataLoader(
            Subset(self._train_base, train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            Subset(self._test_base, test_idx),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader

    def task_classes(self, task_id: int) -> List[int]:
        return self._task_classes[task_id]
