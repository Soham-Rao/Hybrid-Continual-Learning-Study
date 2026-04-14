"""Phase 3 method reimplementation and smoke verification tests.

This suite is the v2 gate for all continual-learning methods. It focuses on:
- all 12 method implementations instantiating correctly
- tiny Class-IL smoke passes on Split CIFAR-10
- tiny Domain-IL smoke passes on Permuted MNIST
- method state round-trips for methods that keep extra internal state
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import get_dataset
from src.methods import get_method
from src.models import get_model
from src.utils import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_local"


METHODS_CFG = [
    ("fine_tune", {}),
    ("joint_training", {"joint_replay_epochs": 1}),
    ("ewc", {"ewc_lambda": 1.0, "fisher_samples": 10}),
    ("agem", {"buffer_size": 20, "agem_mem_batch": 8}),
    ("lwf", {"lwf_lambda": 0.5, "lwf_temp": 2.0}),
    ("der", {"buffer_size": 20, "der_alpha": 0.5}),
    ("xder", {"buffer_size": 20, "der_alpha": 0.5, "xder_beta": 0.5}),
    ("icarl", {"buffer_size": 20, "icarl_temp": 2.0, "use_nmc": True}),
    ("er_ewc", {"buffer_size": 20, "ewc_lambda": 1.0}),
    (
        "progress_compress",
        {"pc_distill_w": 0.5, "pc_ewc_lambda": 1.0, "pc_compress_epochs": 1},
    ),
    ("agem_distill", {"buffer_size": 20, "distill_lambda": 0.5, "agem_mem_batch": 8}),
    ("si_der", {"buffer_size": 20, "der_alpha": 0.5, "si_lambda": 0.5}),
]

STATEFUL_METHODS = {
    "joint_training",
    "ewc",
    "agem",
    "der",
    "xder",
    "icarl",
    "er_ewc",
    "progress_compress",
    "agem_distill",
    "si_der",
}


def build_cfg(overrides: dict) -> dict:
    base = {
        "lr": 0.03,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "fp16": False,
        "batch_size": 4,
    }
    return {**base, **overrides}


def cuda_device() -> torch.device:
    assert torch.cuda.is_available(), "CUDA is required for Phase 3 smoke tests."
    return torch.device("cuda")


def tiny_loader(loader: DataLoader, max_samples: int = 16, batch_size: int = 4) -> DataLoader:
    dataset = loader.dataset
    indices = list(range(min(len(dataset), max_samples)))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@pytest.fixture(scope="session")
def cifar10_ds():
    seed_everything(42)
    return get_dataset("split_cifar10", root=str(DATA_ROOT), batch_size=8, num_workers=0)


@pytest.fixture(scope="session")
def pmnist_ds():
    seed_everything(42)
    return get_dataset(
        "permuted_mnist",
        root=str(DATA_ROOT),
        n_tasks=2,
        batch_size=8,
        num_workers=0,
    )


@pytest.fixture(scope="session")
def cifar10_task0_batch(cifar10_ds):
    train_loader, _ = cifar10_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    return x[:8], y[:8]


@pytest.fixture(scope="session")
def pmnist_task0_batch(pmnist_ds):
    train_loader, _ = pmnist_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    return x[:8], y[:8]


def test_method_registry_has_all_expected_entries() -> None:
    expected = {name for name, _ in METHODS_CFG}
    from src.methods import __all__ as exported

    for method_name in expected:
        assert method_name in exported


@pytest.mark.parametrize("method_name,cfg_override", METHODS_CFG)
def test_method_observe_class_il(method_name, cfg_override, cifar10_task0_batch):
    x, y = cifar10_task0_batch
    device = cuda_device()
    model = get_model("slim_resnet18", in_channels=3).to(device)
    method = get_method(method_name, model, build_cfg(cfg_override), device)

    method.before_task(0, None, 2, n_classes_total=10)
    loss = method.observe(x.to(device), y.to(device), task_id=0)

    assert isinstance(loss, float)
    assert np.isfinite(loss), f"{method_name} produced non-finite loss {loss}"


@pytest.mark.parametrize("method_name,cfg_override", METHODS_CFG)
def test_method_two_task_class_il_smoke(method_name, cfg_override, cifar10_ds):
    device = cuda_device()
    model = get_model("slim_resnet18", in_channels=3).to(device)
    method = get_method(method_name, model, build_cfg(cfg_override), device)

    train0_full, _ = cifar10_ds.get_task_loaders(0)
    train0 = tiny_loader(train0_full)
    method.before_task(0, train0, 2, n_classes_total=cifar10_ds.n_classes_total)
    x0, y0 = next(iter(train0))
    loss0 = method.observe(x0.to(device), y0.to(device), task_id=0)
    assert np.isfinite(loss0)
    method.after_task(0, train0)

    train1_full, _ = cifar10_ds.get_task_loaders(1)
    train1 = tiny_loader(train1_full)
    method.before_task(1, train1, 2, n_classes_total=cifar10_ds.n_classes_total)
    x1, y1 = next(iter(train1))
    loss1 = method.observe(x1.to(device), y1.to(device), task_id=1)
    assert np.isfinite(loss1), f"{method_name} failed on second task"


@pytest.mark.parametrize("method_name,cfg_override", METHODS_CFG)
def test_method_domain_il_smoke(method_name, cfg_override, pmnist_ds, pmnist_task0_batch):
    x, y = pmnist_task0_batch
    device = cuda_device()
    model = get_model("resnet8", in_channels=1).to(device)
    method = get_method(method_name, model, build_cfg(cfg_override), device)

    train0_full, _ = pmnist_ds.get_task_loaders(0)
    train0 = tiny_loader(train0_full)
    method.before_task(0, train0, 10, n_classes_total=pmnist_ds.n_classes_total)
    loss = method.observe(x.to(device), y.to(device), task_id=0)
    assert np.isfinite(loss), f"{method_name} failed on Domain-IL smoke path"


@pytest.mark.parametrize(
    "method_name,cfg_override",
    [(name, cfg) for name, cfg in METHODS_CFG if name in STATEFUL_METHODS],
)
def test_stateful_method_state_roundtrip(method_name, cfg_override, cifar10_ds):
    device = cuda_device()
    cfg = build_cfg(cfg_override)

    model = get_model("slim_resnet18", in_channels=3).to(device)
    method = get_method(method_name, model, cfg, device)

    train0_full, _ = cifar10_ds.get_task_loaders(0)
    train0 = tiny_loader(train0_full)
    method.before_task(0, train0, 2, n_classes_total=cifar10_ds.n_classes_total)
    x0, y0 = next(iter(train0))
    method.observe(x0.to(device), y0.to(device), task_id=0)
    method.after_task(0, train0)

    state = method.state_dict()

    clone_model = get_model("slim_resnet18", in_channels=3).to(device)
    clone_method = get_method(method_name, clone_model, cfg, device)
    clone_method.before_task(0, train0, 2, n_classes_total=cifar10_ds.n_classes_total)
    clone_method.load_state_dict(state)

    original_buffer = method.get_buffer()
    cloned_buffer = clone_method.get_buffer()
    if original_buffer is not None:
        assert cloned_buffer is not None
        assert len(cloned_buffer) == len(original_buffer)

    if method_name == "joint_training":
        assert len(clone_method._all_x) == len(method._all_x)
    if method_name == "ewc":
        assert len(clone_method._old_params) == len(method._old_params)
    if method_name == "er_ewc":
        assert len(clone_method._old_params) == len(method._old_params)
    if method_name == "icarl":
        assert len(clone_method.class_means) == len(method.class_means)
    if method_name == "progress_compress":
        assert clone_method.kb.n_classes == method.kb.n_classes
    if method_name == "si_der":
        assert set(clone_method._si_omega.keys()) == set(method._si_omega.keys())
