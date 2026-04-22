"""Phase 2 dataset and model validation for the v2 workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import get_dataset
from src.models import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_local"
MINI_ROOT = DATA_ROOT / "mini-imagenet"


def cuda_device() -> torch.device:
    assert torch.cuda.is_available(), "CUDA is required for Phase 2 validation."
    return torch.device("cuda")


@pytest.fixture(scope="session")
def mnist_ds():
    return get_dataset("permuted_mnist", root=str(DATA_ROOT), n_tasks=3, batch_size=8, num_workers=0)


@pytest.fixture(scope="session")
def cifar10_ds():
    return get_dataset("split_cifar10", root=str(DATA_ROOT), batch_size=8, num_workers=0)


@pytest.fixture(scope="session")
def cifar100_ds():
    return get_dataset("split_cifar100", root=str(DATA_ROOT), batch_size=8, num_workers=0)


@pytest.fixture(scope="session")
def mini_ds():
    if not MINI_ROOT.exists():
        pytest.fail(f"Mini-ImageNet root missing at {MINI_ROOT}")
    return get_dataset("split_mini_imagenet", root=str(MINI_ROOT), batch_size=4, num_workers=0)


def test_mandatory_dataset_roots_exist() -> None:
    assert (DATA_ROOT / "MNIST").exists()
    assert (DATA_ROOT / "cifar-10-batches-py").exists()
    assert (DATA_ROOT / "cifar-100-python").exists()
    assert MINI_ROOT.exists()


def test_permuted_mnist_loader(mnist_ds) -> None:
    train_loader, _ = mnist_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    assert mnist_ds.scenario == "domain-il"
    assert x.shape == (8, 1, 28, 28)
    assert int(y.min()) >= 0 and int(y.max()) <= 9

    t0_loader, _ = mnist_ds.get_task_loaders(0)
    t1_loader, _ = mnist_ds.get_task_loaders(1)
    x0, _ = next(iter(t0_loader))
    x1, _ = next(iter(t1_loader))
    assert not torch.allclose(x0, x1)


def test_permuted_mnist_rgb_resize_path() -> None:
    ds = get_dataset(
        "permuted_mnist",
        root=str(DATA_ROOT),
        n_tasks=1,
        batch_size=1,
        num_workers=0,
        image_size=224,
        out_channels=3,
    )
    train_loader, _ = ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    assert x.shape == (1, 3, 224, 224)
    assert int(y.min()) >= 0 and int(y.max()) <= 9


def test_split_cifar10_loader(cifar10_ds) -> None:
    train_loader, _ = cifar10_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    assert cifar10_ds.n_tasks == 5
    assert cifar10_ds.n_classes_per_task == 2
    assert x.shape == (8, 3, 32, 32)
    assert set(y.unique().tolist()).issubset(set(cifar10_ds.task_classes(0)))


def test_split_cifar100_loader(cifar100_ds) -> None:
    train_loader, _ = cifar100_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    assert cifar100_ds.n_tasks == 20
    assert cifar100_ds.n_classes_per_task == 5
    assert x.shape == (8, 3, 32, 32)
    assert set(y.unique().tolist()).issubset(set(cifar100_ds.task_classes(0)))


def test_split_mini_imagenet_loader(mini_ds) -> None:
    train_loader, _ = mini_ds.get_task_loaders(0)
    x, y = next(iter(train_loader))
    assert mini_ds.n_tasks == 20
    assert mini_ds.n_classes_per_task == 5
    assert x.shape[1:] == (3, 84, 84)
    assert set(y.unique().tolist()).issubset(set(mini_ds.task_classes(0)))


def test_slim_resnet18_forward_on_cifar_and_mini(mini_ds) -> None:
    device = cuda_device()
    model = get_model("slim_resnet18", in_channels=3, pretrained=False).to(device)
    model.expand(10)

    cifar_x = torch.randn(2, 3, 32, 32, device=device)
    mini_x = torch.randn(2, 3, 84, 84, device=device)
    with torch.no_grad():
        cifar_out = model(cifar_x)
        mini_out = model(mini_x)
    assert cifar_out.shape == (2, 10)
    assert mini_out.shape == (2, 10)


def test_slim_resnet18_forward_on_resized_permuted_mnist() -> None:
    device = cuda_device()
    ds = get_dataset(
        "permuted_mnist",
        root=str(DATA_ROOT),
        n_tasks=1,
        batch_size=2,
        num_workers=0,
        image_size=32,
    )
    train_loader, _ = ds.get_task_loaders(0)
    x, _ = next(iter(train_loader))
    assert x.shape == (2, 1, 32, 32)

    model = get_model("slim_resnet18", in_channels=1, pretrained=False).to(device)
    model.expand(10)
    with torch.no_grad():
        out = model(x.to(device))
    assert out.shape == (2, 10)


def test_expandable_head_preserves_weights() -> None:
    device = cuda_device()
    model = get_model("slim_resnet18", in_channels=3, pretrained=False).to(device)
    model.expand(5)
    weights_before = model.head.weight.data[:5].clone()
    model.expand(5)
    weights_after = model.head.weight.data[:5]
    assert torch.allclose(weights_before, weights_after)


def test_vit_small_build_and_mini_resize_path() -> None:
    pytest.importorskip("timm")
    device = cuda_device()
    ds = get_dataset(
        "split_mini_imagenet",
        root=str(MINI_ROOT),
        batch_size=1,
        num_workers=0,
        image_size=224,
    )
    train_loader, _ = ds.get_task_loaders(0)
    x, _ = next(iter(train_loader))
    assert x.shape[1:] == (3, 224, 224)
    x = x.to(device)

    model = get_model(
        "vit_small_patch16_224",
        in_channels=3,
        pretrained=False,
        image_size=224,
    ).to(device)
    model.expand(5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_convnext_tiny_build_and_forward() -> None:
    pytest.importorskip("timm")
    device = cuda_device()
    model = get_model("convnext_tiny", in_channels=3, pretrained=False).to(device)
    model.expand(5)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)
