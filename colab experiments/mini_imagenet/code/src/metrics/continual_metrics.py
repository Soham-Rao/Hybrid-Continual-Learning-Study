"""Continual Learning evaluation metrics.

All four standard metrics are computed from the accuracy matrix A where
A[t][i] = accuracy on task i's test set after training on task t.

Metrics:
    AA  — Average Accuracy (final-state performance)
    F   — Forgetting Measure (worst-case backward drift)
    BWT — Backward Transfer (did new tasks help old ones?)
    FWT — Forward Transfer (did old tasks help new ones?)
"""

from typing import Dict, List, Optional

import numpy as np


def average_accuracy(acc_matrix: np.ndarray) -> float:
    """Mean accuracy across ALL tasks after learning the LAST task.

    Equation: AA = (1/T) Σ_i A[T-1, i]

    Higher is better.  Ranges [0, 100].
    """
    T = acc_matrix.shape[0]
    # Last row: performance after learning all tasks.
    return float(np.nanmean(acc_matrix[T - 1, :T]))


def forgetting(acc_matrix: np.ndarray) -> float:
    """Average drop from peak performance for each old task.

    Equation: F = (1/(T-1)) Σ_{i=0}^{T-2} (max_{j≤T-1} A[j,i] − A[T-1,i])

    Lower is better.  Positive → forgetting occurred.
    Negative BWT (beneficial backward transfer) is possible but rare.
    """
    T = acc_matrix.shape[0]
    if T == 1:
        return 0.0
    f_values = []
    for i in range(T - 1):
        col    = acc_matrix[:T, i]  # column i, all training steps
        valid  = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        peak   = float(np.max(valid))
        final  = float(acc_matrix[T - 1, i])
        f_values.append(peak - final)
    return float(np.mean(f_values)) if f_values else 0.0


def backward_transfer(acc_matrix: np.ndarray) -> float:
    """Effect of future task learning on past task performance.

    Equation: BWT = (1/(T-1)) Σ_{i=0}^{T-2} (A[T-1,i] − A[i,i])

    Positive BWT  → learning new tasks IMPROVED old task performance.
    Negative BWT  → catastrophic forgetting occurred.
    """
    T = acc_matrix.shape[0]
    if T == 1:
        return 0.0
    entries = []
    for i in range(T - 1):
        final = acc_matrix[T - 1, i]
        after = acc_matrix[i, i]          # accuracy right after learning task i
        if not np.isnan(final) and not np.isnan(after):
            entries.append(final - after)
    return float(np.mean(entries)) if entries else 0.0


def forward_transfer(
    acc_matrix: np.ndarray,
    baseline_acc: List[float],
) -> float:
    """Effect of prior tasks on zero-shot performance on future tasks.

    Equation: FWT = (1/(T-1)) Σ_{i=1}^{T-1} (A[i-1,i] − b_i)

    where b_i is the task i accuracy BEFORE learning starts (random init baseline).

    Positive FWT → prior knowledge gives a head-start on new tasks.
    """
    T = acc_matrix.shape[0]
    if T == 1:
        return 0.0
    entries = []
    for i in range(1, T):
        # A[i-1, i] = zero-shot accuracy on task i after learning tasks 0..i-1
        zero_shot = acc_matrix[i - 1, i]
        if np.isnan(zero_shot):
            continue
        b_i = baseline_acc[i] if i < len(baseline_acc) else 0.0
        entries.append(zero_shot - b_i)
    return float(np.mean(entries)) if entries else 0.0


def compute_all_metrics(
    acc_matrix: np.ndarray,
    baseline_acc: List[float],
    n_tasks: int,
) -> Dict[str, float]:
    """Compute and return all four CL metrics as a labelled dict.

    Args:
        acc_matrix:   Shape (T, T) — filled by :class:`CLTrainer`.
        baseline_acc: Zero-shot accuracy per task (length T list).
        n_tasks:      T (used as sanity check).

    Returns:
        Dict with keys ``avg_accuracy``, ``forgetting``,
        ``backward_transfer``, ``forward_transfer`` — all as percentages.
    """
    return {
        "avg_accuracy":       round(average_accuracy(acc_matrix),    4),
        "forgetting":         round(forgetting(acc_matrix),           4),
        "backward_transfer":  round(backward_transfer(acc_matrix),    4),
        "forward_transfer":   round(forward_transfer(acc_matrix, baseline_acc), 4),
    }
