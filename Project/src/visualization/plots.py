"""All visualization functions for the CL comparative study.

Generates publication-quality figures and saves them as PNG.

Plots produced after every full experiment:
  1. Accuracy heatmap (tasks × tasks)
  2. Forgetting bar chart (per method comparison)
  3. BWT & FWT bar charts
  4. Sequential training accuracy curves
  5. Pareto frontier (accuracy vs. memory)
  6. Compute cost chart (time per task)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless, safe for servers and notebook runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd

# ── Consistent style across all plots ─────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("tab10")

_FIG_DIR = "results/figures"


def _savefig(fig: plt.Figure, filename: str, fig_dir: str = _FIG_DIR) -> str:
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# 1. Accuracy heatmap (A[t][i])
# ===========================================================================
def plot_accuracy_heatmap(
    acc_matrix: np.ndarray,
    method_name: str,
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> str:
    """Heat-map of A[t][i]: columns = tasks, rows = training steps.

    Darker cells ≡ higher accuracy on that task at that training step.
    The diagonal is the accuracy right after learning each task.
    The bottom-left decay shows forgetting visually.
    """
    T = acc_matrix.shape[0]
    masked = np.where(np.isnan(acc_matrix), -1, acc_matrix)

    fig, ax = plt.subplots(figsize=(max(6, T * 0.7), max(5, T * 0.6)))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    im = ax.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{i}" for i in range(T)], rotation=45, ha="right")
    ax.set_yticklabels([f"T{i}" for i in range(T)])
    ax.set_xlabel("Task Evaluated On")
    ax.set_ylabel("Trained Up To Task")
    ax.set_title(f"Accuracy Matrix — {method_name} on {dataset_name}")

    # Annotate cells with accuracy values.
    for i in range(T):
        for j in range(T):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if val > 30 else "white")

    filename = f"heatmap_{dataset_name}_{method_name}.png"
    return _savefig(fig, filename, fig_dir)


# ===========================================================================
# 2. Metric bar charts (Forgetting, BWT, FWT, AA)
# ===========================================================================
def plot_metric_bars(
    results: Dict[str, Dict[str, float]],   # {method_name → metrics_dict}
    metric_key: str,
    dataset_name: str,
    ylabel: str = "",
    title: str = "",
    lower_is_better: bool = False,
    fig_dir: str = _FIG_DIR,
) -> str:
    """Grouped bar chart comparing a single metric across methods.

    Args:
        results:         {method → {metric_key → value}}.
        metric_key:      Key to extract from each method's results dict.
        dataset_name:    Used in filename.
        ylabel:          Y-axis label.
        title:           Chart title.
        lower_is_better: If True, colour the best bar differently.
    """
    methods = list(results.keys())
    values  = [results[m].get(metric_key, float("nan")) for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 0.9), 5))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]

    bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=0.8)

    # Highlight best bar.
    valid = [(v, i) for i, v in enumerate(values) if not np.isnan(v)]
    if valid:
        best_val, best_idx = (min if lower_is_better else max)(valid)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or f"{metric_key} — {dataset_name}")
    ax.set_xticklabels(methods, rotation=30, ha="right")

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1.5),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    filename = f"{metric_key}_{dataset_name}.png"
    return _savefig(fig, filename, fig_dir)


def plot_all_metric_bars(
    results: Dict[str, Dict[str, float]],
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> List[str]:
    """Generate all four metric bar charts in one call."""
    configs = [
        ("avg_accuracy",      "Average Accuracy (%)",     "Avg Accuracy",    False),
        ("forgetting",        "Forgetting (%)",            "Forgetting",      True),
        ("backward_transfer", "Backward Transfer (%)",     "BWT",             False),
        ("forward_transfer",  "Forward Transfer (%)",      "FWT",             False),
    ]
    paths = []
    for key, ylabel, title, lib in configs:
        p = plot_metric_bars(
            results, key, dataset_name,
            ylabel=ylabel, title=f"{title} — {dataset_name}",
            lower_is_better=lib, fig_dir=fig_dir,
        )
        paths.append(p)
    return paths


# ===========================================================================
# 3. Sequential training curves
# ===========================================================================
def plot_training_curves(
    acc_matrix: np.ndarray,
    method_name: str,
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> str:
    """Line chart: accuracy on each past task as training progresses.

    Each line represents one task. After the task is learned, its line
    shows whether accuracy is retained or drops over subsequent tasks.
    """
    T   = acc_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(7, T), 5))

    for i in range(T):
        # Only show points where task i has been seen (j ≥ i).
        xs  = list(range(i, T))
        ys  = [acc_matrix[j][i] for j in range(i, T)]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                color=PALETTE[i % len(PALETTE)], label=f"Task {i}")

    ax.set_xlabel("Training Step (Task Index)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Training Curves — {method_name} on {dataset_name}")
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{i}" for i in range(T)])
    ax.set_ylim(-5, 105)
    ax.legend(
        loc="upper right" if T <= 10 else "upper left",
        fontsize=8, ncol=max(1, T // 10),
    )

    filename = f"curves_{dataset_name}_{method_name}.png"
    return _savefig(fig, filename, fig_dir)


# ===========================================================================
# 4. Pareto frontier (accuracy vs. memory)
# ===========================================================================
def plot_pareto_frontier(
    results: Dict[str, Dict],
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> str:
    """Scatter plot of final avg accuracy vs. replay buffer size.

    Methods with no buffer (EWC, LwF, P&C) appear at memory = 0.
    The Pareto-optimal frontier is highlighted.

    Args:
        results: {method → {"avg_accuracy": float, "buffer_size": int, …}}
    """
    methods  = list(results.keys())
    acc_vals = [results[m].get("avg_accuracy", 0)  for m in methods]
    mem_vals = [results[m].get("buffer_size",  0)  for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (m, acc, mem) in enumerate(zip(methods, acc_vals, mem_vals)):
        ax.scatter(mem, acc, s=120, color=PALETTE[i % len(PALETTE)],
                   zorder=3, label=m)
        ax.annotate(m, (mem, acc), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)

    # Draw Pareto frontier (non-dominated points: higher acc AND lower mem).
    pts = sorted(zip(mem_vals, acc_vals))
    pareto = []
    best_acc = -np.inf
    for mem, acc in sorted(pts, key=lambda x: (x[0], -x[1])):
        if acc > best_acc:
            pareto.append((mem, acc))
            best_acc = acc
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.step(px, py, where="post", linestyle="--", color="grey",
                linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Replay Buffer Size (samples)")
    ax.set_ylabel("Final Average Accuracy (%)")
    ax.set_title(f"Accuracy vs. Memory Trade-off — {dataset_name}")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    filename = f"pareto_{dataset_name}.png"
    return _savefig(fig, filename, fig_dir)


# ===========================================================================
# 5. Compute cost chart (wall-clock time per task)
# ===========================================================================
def plot_compute_cost(
    task_times: Dict[str, List[float]],   # {method → [time_task0, time_task1, …]}
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> str:
    """Grouped bar chart: training time (seconds) per task, per method."""
    methods  = list(task_times.keys())
    n_tasks  = max(len(v) for v in task_times.values())

    x     = np.arange(n_tasks)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(8, n_tasks), 5))

    for idx, m in enumerate(methods):
        times  = task_times[m] + [0] * (n_tasks - len(task_times[m]))
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=m,
               color=PALETTE[idx % len(PALETTE)], alpha=0.85)

    ax.set_xlabel("Task Index")
    ax.set_ylabel("Wall-clock Time (seconds)")
    ax.set_title(f"Compute Cost per Task — {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)])
    ax.legend(fontsize=9)

    filename = f"compute_{dataset_name}.png"
    return _savefig(fig, filename, fig_dir)


# ===========================================================================
# 6. Combined summary grid (4 metrics + heatmap side-by-side)
# ===========================================================================
def plot_summary_grid(
    acc_matrix: np.ndarray,
    metrics: Dict[str, float],
    method_name: str,
    dataset_name: str,
    fig_dir: str = _FIG_DIR,
) -> str:
    """2×2 panel: heatmap | training curves | metric text | forgetting bars."""
    T = acc_matrix.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: heatmap ────────────────────────────────────────────────
    ax = axes[0]
    masked = np.where(np.isnan(acc_matrix), -1, acc_matrix)
    im = ax.imshow(masked, vmin=0, vmax=100,
                   cmap=sns.color_palette("RdYlGn", as_cmap=True), aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{i}" for i in range(T)], fontsize=7, rotation=45)
    ax.set_yticklabels([f"T{i}" for i in range(T)], fontsize=7)
    ax.set_title("Accuracy Matrix")
    ax.set_xlabel("Task Evaluated")
    ax.set_ylabel("Trained Up To")

    # ── Right: training curves ────────────────────────────────────────
    ax2 = axes[1]
    for i in range(T):
        xs = list(range(i, T))
        ys = [acc_matrix[j][i] for j in range(i, T)]
        ax2.plot(xs, ys, marker="o", markersize=3, linewidth=1.5,
                 color=PALETTE[i % len(PALETTE)], label=f"T{i}")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=7, ncol=max(1, T // 8))

    # ── Sup-title with metric summary ────────────────────────────────
    sup = (
        f"{method_name} | {dataset_name}   "
        f"AA={metrics['avg_accuracy']:.1f}%  "
        f"F={metrics['forgetting']:.1f}%  "
        f"BWT={metrics['backward_transfer']:.1f}%  "
        f"FWT={metrics['forward_transfer']:.1f}%"
    )
    fig.suptitle(sup, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    filename = f"summary_{dataset_name}_{method_name}.png"
    return _savefig(fig, filename, fig_dir)
