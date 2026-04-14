# src/metrics/__init__.py
from .continual_metrics import (
    average_accuracy,
    forgetting,
    backward_transfer,
    forward_transfer,
    compute_all_metrics,
)
__all__ = [
    "average_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
    "compute_all_metrics",
]
