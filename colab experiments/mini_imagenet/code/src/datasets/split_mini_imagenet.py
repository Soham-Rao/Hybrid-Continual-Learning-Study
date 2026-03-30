"""Split Mini-ImageNet — Class-Incremental benchmark (Colab / high-VRAM only).

Mini-ImageNet: 100 classes, 64,000 images at 84×84 RGB pixels.
Split into **20 sequential tasks of 5 classes each** (Class-IL).

Dataset download:
  Option A — Kaggle: https://www.kaggle.com/datasets/arjunashok33/miniimagenet
  Option B — Drive:  Upload the 'mini-imagenet' folder to Google Drive,
                     then mount Drive and set ``root`` to the folder path.

Expected directory layout::

    <root>/
        train/
            n01532829/   (class folder — WordNet ID)
                *.jpg
            n01558993/
            ...
        test/
            n01532829/
            ...

If no separate test split exists, this loader performs an 80/20 stratified
split automatically from the train folder.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .base_dataset import BaseCLDataset

_MEAN = (0.473, 0.449, 0.403)
_STD  = (0.278, 0.269, 0.284)

_N_TASKS          = 20
_CLASSES_PER_TASK = 5
_N_CLASSES        = 100
_IMAGE_SIZE       = 84


def _build_transforms(train: bool, image_size: int = _IMAGE_SIZE) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class SplitMiniImageNet(BaseCLDataset):
    """20-task Split Mini-ImageNet benchmark (Class-IL, 5 classes/task).

    This dataset is **intended for Colab execution** where adequate VRAM
    (≥12 GB) is available.

    Args:
        root:        Path to the mini-imagenet root directory.
        batch_size:  Mini-batch size (16 recommended for ViT-Small on T4).
        num_workers: DataLoader workers (4 recommended on Colab).
        seed:        Used for reproducible train/test split when no
                     separate test folder exists.
    """

    _TASK_CLASSES: List[List[int]] = [
        list(range(i * _CLASSES_PER_TASK, (i + 1) * _CLASSES_PER_TASK))
        for i in range(_N_TASKS)
    ]

    def __init__(
        self,
        root: str = "data/mini-imagenet",
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        image_size: int = _IMAGE_SIZE,
    ) -> None:
        super().__init__(root, batch_size, num_workers)
        train_dir = os.path.join(root, "train")
        test_dir  = os.path.join(root, "test")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Mini-ImageNet train split not found at: {train_dir}\n"
                "Download from Kaggle or Google Drive and set `root` correctly."
            )

        self._train_base = datasets.ImageFolder(
            train_dir, transform=_build_transforms(True, image_size=image_size)
        )

        if os.path.isdir(test_dir):
            self._test_base = datasets.ImageFolder(
            test_dir, transform=_build_transforms(False, image_size=image_size)
        )
        else:
            # Fallback: 80/20 stratified split from training folder.
            base = datasets.ImageFolder(
                train_dir, transform=_build_transforms(False, image_size=image_size)
            )
            train_idx, test_idx = self._stratified_split(base, seed)
            self._train_base = Subset(
                datasets.ImageFolder(
                    train_dir, transform=_build_transforms(True, image_size=image_size)
                ),
                train_idx
            )
            self._test_base = Subset(base, test_idx)

    # ------------------------------------------------------------------
    @staticmethod
    def _stratified_split(
        dataset: datasets.ImageFolder,
        seed: int,
        test_ratio: float = 0.2,
    ) -> Tuple[List[int], List[int]]:
        targets = np.array(dataset.targets)
        rng = np.random.default_rng(seed)
        train_idx, test_idx = [], []
        for cls in np.unique(targets):
            idx = np.where(targets == cls)[0]
            rng.shuffle(idx)
            n_test = max(1, int(len(idx) * test_ratio))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
        return train_idx, test_idx

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return _N_TASKS

    @property
    def n_classes_per_task(self) -> int:
        return _CLASSES_PER_TASK

    @property
    def n_classes_total(self) -> int:
        return _N_CLASSES

    @property
    def input_size(self) -> Tuple[int, int, int]:
        return (3, _IMAGE_SIZE, _IMAGE_SIZE)

    @property
    def scenario(self) -> str:
        return "class-il"

    # ------------------------------------------------------------------
    def _get_indices(self, base, classes: List[int]) -> List[int]:
        if isinstance(base, Subset):
            targets = np.array(base.dataset.targets)[base.indices]
            local_idx = np.where(np.isin(targets, classes))[0]
            return local_idx.tolist()
        targets = np.array(base.targets)
        return np.where(np.isin(targets, classes))[0].tolist()

    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        classes = self._TASK_CLASSES[task_id]
        train_loader = DataLoader(
            Subset(self._train_base, self._get_indices(self._train_base, classes))
            if not isinstance(self._train_base, Subset)
            else Subset(self._train_base.dataset,
                        [self._train_base.indices[i]
                         for i in self._get_indices(self._train_base, classes)]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            Subset(self._test_base, self._get_indices(self._test_base, classes))
            if not isinstance(self._test_base, Subset)
            else Subset(self._test_base.dataset,
                        [self._test_base.indices[i]
                         for i in self._get_indices(self._test_base, classes)]),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader
