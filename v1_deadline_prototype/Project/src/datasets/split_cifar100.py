"""Split CIFAR-100 — Class-Incremental continual learning benchmark.

CIFAR-100's 100 classes are split into **20 sequential tasks of 5 classes each**.
Labels are global (0-99).  Task identity is NOT available at test time (Class-IL).

Reference: Main benchmark in DER, X-DER, iCaRL papers.
"""

from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseCLDataset

_MEAN = (0.5071, 0.4867, 0.4408)
_STD  = (0.2675, 0.2565, 0.2761)

_N_TASKS          = 20
_CLASSES_PER_TASK = 5


def _build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _class_indices(dataset: Dataset, classes: List[int]) -> List[int]:
    targets = np.array(dataset.targets)
    return np.where(np.isin(targets, classes))[0].tolist()


class SplitCIFAR100(BaseCLDataset):
    """20-task Split CIFAR-100 benchmark (Class-IL, 5 classes/task).

    Args:
        root:        Directory where CIFAR-100 will be downloaded.
        n_tasks:     Optional override for number of tasks (must divide 100).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        root: str = "data",
        n_tasks: int | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> None:
        super().__init__(root, batch_size, num_workers)
        total_classes = 100
        if n_tasks is None:
            n_tasks = _N_TASKS
        if total_classes % n_tasks != 0:
            raise ValueError(
                f"n_tasks={n_tasks} must divide {total_classes} for SplitCIFAR100."
            )
        self._n_tasks = n_tasks
        self._classes_per_task = total_classes // n_tasks
        self._task_classes: List[List[int]] = [
            list(range(i * self._classes_per_task, (i + 1) * self._classes_per_task))
            for i in range(self._n_tasks)
        ]
        self._train_base = datasets.CIFAR100(root, train=True,  download=True, transform=_build_transforms(True))
        self._test_base  = datasets.CIFAR100(root, train=False, download=True, transform=_build_transforms(False))

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @property
    def n_classes_per_task(self) -> int:
        return self._classes_per_task

    @property
    def n_classes_total(self) -> int:
        return 100

    @property
    def input_size(self) -> Tuple[int, int, int]:
        return (3, 32, 32)

    @property
    def scenario(self) -> str:
        return "class-il"

    # ------------------------------------------------------------------
    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        classes = self._task_classes[task_id]

        train_loader = DataLoader(
            Subset(self._train_base, _class_indices(self._train_base, classes)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            Subset(self._test_base, _class_indices(self._test_base, classes)),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader

    def task_classes(self, task_id: int) -> List[int]:
        return self._task_classes[task_id]
