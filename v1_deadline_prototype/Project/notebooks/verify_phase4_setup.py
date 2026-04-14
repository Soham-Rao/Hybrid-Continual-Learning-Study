"""Quick local validation for the Phase 4 setup."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _check_path(label: str, path: Path) -> bool:
    ok = path.exists()
    status = "OK" if ok else "MISSING"
    print(f"{label}: {status} -> {path}")
    return ok


def main() -> None:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("device_name=cpu")

    checks = []
    checks.append(_check_path("mini_root", ROOT / "Project" / "data_local" / "mini-imagenet"))
    checks.append(_check_path("tiny_root", ROOT / "Project" / "data_local" / "tiny-imagenet-200"))
    checks.append(_check_path("mini_runner", ROOT / "Project" / "notebooks" / "local_mini_imagenet.py"))
    checks.append(_check_path("tiny_runner", ROOT / "Project" / "notebooks" / "local_tiny_imagenet.py"))
    checks.append(_check_path("mini_cfg_resnet", ROOT / "Project" / "experiments" / "configs" / "phase4" / "mini_imagenet_resnet18_local.yaml"))
    checks.append(_check_path("mini_cfg_vit", ROOT / "Project" / "experiments" / "configs" / "phase4" / "mini_imagenet_vit_small_local.yaml"))
    checks.append(_check_path("tiny_cfg_resnet", ROOT / "Project" / "experiments" / "configs" / "phase4" / "tiny_imagenet_resnet18_local.yaml"))
    checks.append(_check_path("tiny_cfg_vit", ROOT / "Project" / "experiments" / "configs" / "phase4" / "tiny_imagenet_vit_small_local.yaml"))

    if all(checks):
        print("phase4_setup_status=ready")
    else:
        print("phase4_setup_status=partial")


if __name__ == "__main__":
    main()
