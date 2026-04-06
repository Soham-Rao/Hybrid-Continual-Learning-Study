"""Phase 1 end-to-end verification tests.

Run from the project root:
    pytest tests/test_phase1.py -v

Tests cover:
  1. All imports resolve correctly
  2. Metric math (known expected values)
  3. PermutedMNIST dataset (domain-IL, correct label range)
  4. SplitCIFAR10 dataset (class-IL, per-task label correctness)
  5. ResNet8 + expandable head (shape, weight preservation)
  6. SlimResNet18 (shape and param count)
  7. All implemented methods instantiate and basic observe()/2-task flows work
  8. Replay buffer reservoir sampling (capacity enforcement)
  9. Visualization pipeline (file is actually written to disk)
 10. End-to-end mini-run (PermutedMNIST, FineTune, 3 tasks, CPU)
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make src importable when running from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets import get_dataset
from src.metrics.continual_metrics import compute_all_metrics
from src.methods import get_method
from src.methods.base_method import ReplayBuffer
from src.models import get_model
from src.utils import RunLogger, seed_everything
from src.visualization.plots import plot_summary_grid

# Prefer local dataset mirror if provided; fall back to ./data for portability.
DATA_ROOT = os.environ.get("DATA_ROOT", "data")


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mnist_ds():
    seed_everything(42)
    return get_dataset("permuted_mnist", root=DATA_ROOT, n_tasks=3, batch_size=64, num_workers=0)


@pytest.fixture(scope="session")
def cifar10_ds():
    return get_dataset("split_cifar10", root=DATA_ROOT, batch_size=64, num_workers=0)


@pytest.fixture(scope="session")
def mnist_batch(mnist_ds):
    train_l, _ = mnist_ds.get_task_loaders(0)
    x, y = next(iter(train_l))
    return x, y


@pytest.fixture(scope="session")
def cifar10_batch(cifar10_ds):
    train_l, _ = cifar10_ds.get_task_loaders(0)
    x, y = next(iter(train_l))
    return x, y


# ── Test 1: Imports ────────────────────────────────────────────────────────

def test_imports():
    from src.datasets import get_dataset
    from src.methods import get_method
    from src.metrics.continual_metrics import compute_all_metrics
    from src.models import get_model
    from src.trainers.cl_trainer import CLTrainer
    from src.utils import RunLogger, seed_everything
    from src.visualization.plots import plot_summary_grid


# ── Test 2: Metric correctness ─────────────────────────────────────────────

def test_metric_values():
    A = np.array([
        [90.0, np.nan, np.nan],
        [70.0, 85.0,   np.nan],
        [60.0, 75.0,   88.0],
    ])
    m = compute_all_metrics(A, baseline_acc=[10.0, 10.0, 10.0], n_tasks=3)

    # Forgetting: ((90-60) + (85-75)) / 2 = 20.0
    assert abs(m["forgetting"] - 20.0) < 0.01, f"Forgetting={m['forgetting']}"

    # BWT: ((60-90) + (75-85)) / 2 = -20.0
    assert abs(m["backward_transfer"] + 20.0) < 0.01, f"BWT={m['backward_transfer']}"

    # AA: mean of last row [60, 75, 88] = 74.333...
    assert abs(m["avg_accuracy"] - 74.333) < 0.1, f"AA={m['avg_accuracy']}"

    # FWT uses only non-NaN zero-shot entries in the current implementation.
    assert isinstance(m["forward_transfer"], float)


# ── Test 3: PermutedMNIST ─────────────────────────────────────────────────

def test_permuted_mnist_meta(mnist_ds):
    assert mnist_ds.n_tasks == 3
    assert mnist_ds.scenario == "domain-il"
    assert mnist_ds.n_classes_total == 10
    assert mnist_ds.input_size == (1, 28, 28)


def test_permuted_mnist_batch(mnist_batch):
    x, y = mnist_batch
    assert x.shape == (64, 1, 28, 28)
    assert y.min() >= 0 and y.max() <= 9


def test_permuted_mnist_tasks_differ(mnist_ds):
    """Different tasks should have different pixel arrangements."""
    t0_loader, _ = mnist_ds.get_task_loaders(0)
    t1_loader, _ = mnist_ds.get_task_loaders(1)
    x0, _ = next(iter(t0_loader))
    x1, _ = next(iter(t1_loader))
    # Same raw images, different permutations → should not be identical.
    assert not torch.allclose(x0, x1)


# ── Test 4: Split CIFAR-10 ────────────────────────────────────────────────

def test_cifar10_meta(cifar10_ds):
    assert cifar10_ds.n_tasks == 5
    assert cifar10_ds.n_classes_per_task == 2
    assert cifar10_ds.scenario == "class-il"
    assert cifar10_ds.n_classes_total == 10


def test_cifar10_task_labels(cifar10_ds):
    for task_id in range(cifar10_ds.n_tasks):
        tr, _ = cifar10_ds.get_task_loaders(task_id)
        x, y = next(iter(tr))
        expected = set(cifar10_ds.task_classes(task_id))
        actual   = set(y.unique().tolist())
        assert actual.issubset(expected), f"Task {task_id}: {actual} not in {expected}"


# ── Test 5: ResNet8 + expandable head ─────────────────────────────────────

def test_resnet8_shape(mnist_batch):
    x, _ = mnist_batch
    model = get_model("resnet8", in_channels=1)
    model.expand(10)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (64, 10)


def test_resnet8_features(mnist_batch):
    x, _ = mnist_batch
    model = get_model("resnet8", in_channels=1)
    model.expand(10)
    feats = model.get_features(x)
    assert feats.shape == (64, 64)


def test_head_expansion_preserves_weights():
    """Expanding the head must not change previously learned weights."""
    model = get_model("resnet8", in_channels=3)
    model.expand(2)
    w_before = model.head.weight.data[:2].clone()
    model.expand(2)
    w_after  = model.head.weight.data[:2]
    assert torch.allclose(w_before, w_after), "Head expansion corrupted old weights"


# ── Test 6: SlimResNet18 ──────────────────────────────────────────────────

def test_slim_resnet18(cifar10_batch):
    x, _ = cifar10_batch
    model = get_model("slim_resnet18", in_channels=3)
    model.expand(10)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (64, 10)
    n_params = sum(p.numel() for p in model.parameters())
    # SlimResNet18 should be under 3M parameters
    assert n_params < 3_000_000, f"Too many params: {n_params:,}"


# ── Test 7: All methods observe() ────────────────────────────────────────

METHODS_CFG = [
    ("fine_tune",       {}),
    ("joint_training",  {"joint_replay_epochs": 1}),
    ("ewc",             {"ewc_lambda": 1.0, "fisher_samples": 10}),
    ("lwf",             {"lwf_lambda": 0.5, "lwf_temp": 2.0}),
    ("agem",            {"buffer_size": 20}),
    ("der",             {"buffer_size": 20, "der_alpha": 0.5}),
    ("xder",            {"buffer_size": 20, "der_alpha": 0.5, "xder_beta": 0.5}),
    ("icarl",           {"buffer_size": 20, "icarl_temp": 2.0, "use_nmc": True}),
    ("er_ewc",          {"buffer_size": 20, "ewc_lambda": 1.0}),
    ("progress_compress", {"pc_distill_w": 0.5, "pc_ewc_lambda": 1.0, "pc_compress_epochs": 1}),
    ("agem_distill",    {"buffer_size": 20, "distill_lambda": 0.5}),
    ("si_der",          {"buffer_size": 20, "der_alpha": 0.5, "si_lambda": 0.5}),
]


@pytest.mark.parametrize("method_name,cfg_override", METHODS_CFG)
def test_method_observe(method_name, cfg_override, cifar10_batch):
    x, y = cifar10_batch
    x, y = x[:8], y[:8]              # tiny batch for speed
    device = torch.device("cpu")

    model  = get_model("slim_resnet18", in_channels=3).to(device)
    method = get_method(method_name, model, cfg_override, device)
    method.before_task(0, None, 2)    # expand for task-0 (2 classes)

    loss = method.observe(x, y, task_id=0)
    assert isinstance(loss, float), f"{method_name} returned {type(loss)}"
    assert np.isfinite(loss),        f"{method_name} loss is not finite: {loss}"


@pytest.mark.parametrize("method_name,cfg_override", METHODS_CFG)
def test_method_two_tasks(method_name, cfg_override, cifar10_batch):
    """Basic sanity: method survives training on 2 tasks without error."""
    x0, y0 = cifar10_batch
    device = torch.device("cpu")
    model  = get_model("slim_resnet18", in_channels=3).to(device)
    method = get_method(method_name, model, cfg_override, device)
    ds = get_dataset("split_cifar10", root=DATA_ROOT, batch_size=8, num_workers=0)

    # Task 0
    train0, _ = ds.get_task_loaders(0)
    method.before_task(0, train0, 2)
    for _ in range(2):
        method.observe(x0[:8], y0[:8], task_id=0)
    method.after_task(0, train0)

    # Task 1 — different labels
    train1, _ = ds.get_task_loaders(1)
    x1, y1 = next(iter(train1))
    x1, y1 = x1.to(device), y1.to(device)
    method.before_task(1, train1, 2)
    loss = method.observe(x1[:8], y1[:8], task_id=1)
    assert np.isfinite(loss)


# ── Test 8: Replay buffer ────────────────────────────────────────────────

def test_reservoir_capacity():
    buf = ReplayBuffer(capacity=50, strategy="reservoir")
    for _ in range(100):
        buf.add(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)), task_id=0)
    assert len(buf) == 50


def test_reservoir_sample_shape():
    buf = ReplayBuffer(capacity=100, strategy="reservoir")
    buf.add(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)), task_id=0)
    s = buf.sample(16, torch.device("cpu"))
    assert s["x"].shape == (16, 3, 32, 32)
    assert s["y"].shape == (16,)


# ── Test 9: Visualization saves file ─────────────────────────────────────

def test_plot_summary_grid():
    A = np.array([
        [80.0, np.nan],
        [60.0, 75.0],
    ])
    m = compute_all_metrics(A, baseline_acc=[10.0, 10.0], n_tasks=2)
    from pathlib import Path
    out_dir = Path(__file__).parent / "_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = plot_summary_grid(A, m, "fine_tune", "test_ds", fig_dir=str(out_dir))
    assert os.path.exists(path), f"Plot not created at {path}"
    assert os.path.getsize(path) > 1000, "Plot file is suspiciously small"


# ── Test 10: End-to-end mini-run ─────────────────────────────────────────

def test_end_to_end_mini_run():
    """Full pipeline: PermutedMNIST × FineTune × 3 tasks × 1 epoch on CPU."""
    seed_everything(42)
    device = torch.device("cpu")

    from pathlib import Path
    out_dir = Path(__file__).parent / "_artifacts" / "mini_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_epochs":       1,
        "batch_size":     256,       # large batch = fewer steps = faster
        "lr":             0.03,
        "fp16":           False,
        "checkpoint_dir": str(out_dir / "checkpoints"),
        "figure_dir":     str(out_dir / "figures"),
        "log_dir":        str(out_dir / "logs"),
        "run_name":       "test_run",
    }

    ds     = get_dataset("permuted_mnist", root=DATA_ROOT, n_tasks=3, batch_size=256, num_workers=0)
    model  = get_model("resnet8", in_channels=1).to(device)
    method = get_method("fine_tune", model, cfg, device)
    logger = RunLogger(cfg["log_dir"], "test_run")

    from src.trainers.cl_trainer import CLTrainer
    trainer = CLTrainer(method, ds, cfg, logger, device)
    results = trainer.train()

    acc_matrix = results["acc_matrix"]
    metrics    = results["metrics"]

    # All 3 diagonal entries must be filled and > 0
    for i in range(3):
        assert not np.isnan(acc_matrix[i, i]), f"A[{i},{i}] is NaN"
        assert acc_matrix[i, i] > 0

    # Fine-tune must forget: Task 0 accuracy must fall after Task 1+2
    assert acc_matrix[0, 0] > acc_matrix[2, 0], \
        "Fine-tune should forget task 0 after training on tasks 1 and 2"

    # Metrics must be finite floats
    for k, v in metrics.items():
        assert np.isfinite(v), f"Metric {k} = {v}"

    print(f"\nMini-run metrics: {metrics}")
    print(f"Acc matrix:\n{acc_matrix}")
