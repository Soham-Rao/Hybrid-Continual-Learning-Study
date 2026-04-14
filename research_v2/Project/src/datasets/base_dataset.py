"""Abstract base class for all continual-learning dataset wrappers.

Every dataset exposes the same interface so the trainer can call::

    for task_id in range(dataset.n_tasks):
        train_loader, test_loader = dataset.get_task_loaders(task_id)

Labels are **global** (not remapped to 0..n_classes_per_task-1), so the
shared expanding classifier head can be trained correctly.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch.utils.data import DataLoader


class BaseCLDataset(ABC):
    """Common interface for all continual-learning benchmark datasets."""

    def __init__(
        self,
        root: str = "data",
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> None:
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    # Properties that every subclass must define
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_tasks(self) -> int:
        """Total number of sequential tasks."""

    @property
    @abstractmethod
    def n_classes_per_task(self) -> int:
        """Number of new classes introduced by each task. (Domain-IL datasets
        may return the shared class count here instead.)"""

    @property
    @abstractmethod
    def n_classes_total(self) -> int:
        """Total number of unique output classes across all tasks."""

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int, int]:
        """Input tensor shape as ``(C, H, W)``."""

    @property
    @abstractmethod
    def scenario(self) -> str:
        """One of ``'class-il'`` or ``'domain-il'``."""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_task_loaders(
        self, task_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Return ``(train_loader, test_loader)`` for *task_id*.

        Train loader:  yields ``(x, y)`` where ``y`` uses global labels.
        Test  loader:  same, used for per-task accuracy evaluation.
        """

    def get_all_test_loaders(self) -> List[DataLoader]:
        """Return one test loader per task (used to fill the accuracy matrix)."""
        return [self.get_task_loaders(t)[1] for t in range(self.n_tasks)]

    def task_class_range(self, task_id: int) -> Tuple[int, int]:
        """Global label range ``[start, end)`` for *task_id*.

        For Domain-IL datasets (shared label space) both ``start`` and
        ``end`` span the full class set.
        """
        if self.scenario == "domain-il":
            return 0, self.n_classes_total
        start = task_id * self.n_classes_per_task
        end = start + self.n_classes_per_task
        return start, end

    def task_classes(self, task_id: int) -> List[int]:
        """Return the list of global class IDs for *task_id*.

        For Domain-IL this returns the full class list (0..n_classes_total-1).
        For Class-IL this returns exactly ``n_classes_per_task`` IDs.
        """
        start, end = self.task_class_range(task_id)
        return list(range(start, end))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tasks={self.n_tasks}, "
            f"classes_per_task={self.n_classes_per_task}, "
            f"scenario={self.scenario})"
        )
