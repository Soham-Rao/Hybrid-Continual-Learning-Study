"""Sequential Tiny-ImageNet — Class-Incremental benchmark for local or cloud runs.

Tiny-ImageNet: 200 classes, 110,000 images at 64×64 RGB pixels.
Split into **10 sequential tasks of 20 classes each** (Class-IL).

Download::

    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip -d data/

Expected layout::

    data/tiny-imagenet-200/
        train/
            n01443537/
                images/
                    n01443537_0.JPEG
                    ...
        val/
            images/
                val_0.JPEG
                ...
            val_annotations.txt
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .base_dataset import BaseCLDataset

_MEAN = (0.480, 0.448, 0.397)
_STD  = (0.277, 0.269, 0.282)

_N_TASKS          = 10
_CLASSES_PER_TASK = 20
_N_CLASSES        = 200
_IMAGE_SIZE       = 64


def _build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ---------------------------------------------------------------------------
class TinyImageNetDataset(Dataset):
    """Flat file-based loader that handles Tiny-ImageNet's quirky structure."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        wnid_dirs = sorted(os.listdir(os.path.join(root, "train")))
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnid_dirs)}

        if split == "train":
            for wnid, idx in self.class_to_idx.items():
                img_dir = os.path.join(root, "train", wnid, "images")
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fname), idx))
        else:
            # Validation split — uses val_annotations.txt for label mapping.
            ann_path = os.path.join(root, "val", "val_annotations.txt")
            img_dir  = os.path.join(root, "val", "images")
            with open(ann_path) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    fname, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        self.samples.append(
                            (os.path.join(img_dir, fname), self.class_to_idx[wnid])
                        )

        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
class SeqTinyImageNet(BaseCLDataset):
    """10-task Sequential Tiny-ImageNet benchmark (Class-IL, 20 classes/task).

    Args:
        root:        Path to ``tiny-imagenet-200`` root directory.
        batch_size:  Small batches are recommended for local ViT runs.
        num_workers: DataLoader workers for the local machine.
    """

    _TASK_CLASSES: List[List[int]] = [
        list(range(i * _CLASSES_PER_TASK, (i + 1) * _CLASSES_PER_TASK))
        for i in range(_N_TASKS)
    ]

    def __init__(
        self,
        root: str = "data/tiny-imagenet-200",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__(root, batch_size, num_workers)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at: {root}\n"
                "Download: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
        self._train_base = TinyImageNetDataset(root, "train", _build_transforms(True))
        self._test_base  = TinyImageNetDataset(root, "val",   _build_transforms(False))

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
    def _get_indices(self, base: TinyImageNetDataset, classes: List[int]) -> List[int]:
        targets = np.array(base.targets)
        return np.where(np.isin(targets, classes))[0].tolist()

    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        classes = self._TASK_CLASSES[task_id]
        train_loader = DataLoader(
            Subset(self._train_base, self._get_indices(self._train_base, classes)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            Subset(self._test_base, self._get_indices(self._test_base, classes)),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader
