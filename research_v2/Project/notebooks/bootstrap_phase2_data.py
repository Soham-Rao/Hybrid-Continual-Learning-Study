"""Bootstrap v2 local datasets from archived mirrors or lightweight downloads.

Primary policy:
- prefer linking existing v1 local dataset mirrors into v2
- download torchvision-managed small datasets only if they are missing
- keep Mini-ImageNet / Tiny-ImageNet local-first and never silently fetch them
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_V1_DATA_ROOT = WORKSPACE_ROOT.parent / "v1_deadline_prototype" / "Project" / "data_local"
V2_DATA_ROOT = PROJECT_ROOT / "data_local"

DATASETS = [
    "MNIST",
    "cifar-10-batches-py",
    "cifar-100-python",
    "mini-imagenet",
    "tiny-imagenet-200",
]


def _create_link_or_copy(src: Path, dst: Path, prefer_links: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    if prefer_links:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                os.symlink(src, dst, target_is_directory=True)
            return "linked"
        except Exception:
            pass

    shutil.copytree(src, dst)
    return "copied"


def _download_small_datasets(root: Path) -> list[str]:
    downloaded: list[str] = []
    from torchvision import datasets

    if not (root / "MNIST").exists():
        datasets.MNIST(str(root), train=True, download=True)
        datasets.MNIST(str(root), train=False, download=True)
        downloaded.append("MNIST")

    if not (root / "cifar-10-batches-py").exists():
        datasets.CIFAR10(str(root), train=True, download=True)
        datasets.CIFAR10(str(root), train=False, download=True)
        downloaded.append("CIFAR10")

    if not (root / "cifar-100-python").exists():
        datasets.CIFAR100(str(root), train=True, download=True)
        datasets.CIFAR100(str(root), train=False, download=True)
        downloaded.append("CIFAR100")

    return downloaded


def bootstrap(v1_data_root: Path, prefer_links: bool, download_small: bool) -> None:
    V2_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"v2 data root: {V2_DATA_ROOT}")
    print(f"v1 data root: {v1_data_root}")

    for name in DATASETS:
        src = v1_data_root / name
        dst = V2_DATA_ROOT / name
        if src.exists():
            mode = _create_link_or_copy(src, dst, prefer_links=prefer_links)
            print(f"{name}: {mode}")
        else:
            print(f"{name}: source missing in v1 mirror")

    if download_small:
        downloaded = _download_small_datasets(V2_DATA_ROOT)
        if downloaded:
            print("Downloaded small datasets:", ", ".join(downloaded))
        else:
            print("Small torchvision datasets already present.")

    missing_large = [
        name for name in ("mini-imagenet", "tiny-imagenet-200")
        if not (V2_DATA_ROOT / name).exists()
    ]
    if missing_large:
        print("Large local-only datasets still missing:", ", ".join(missing_large))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-data-root",
        default=str(DEFAULT_V1_DATA_ROOT),
        help="Archived v1 data_local directory to reuse when present.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy directories instead of linking/junctioning them.",
    )
    parser.add_argument(
        "--no-download-small",
        action="store_true",
        help="Do not attempt torchvision downloads for MNIST/CIFAR if missing.",
    )
    args = parser.parse_args()

    bootstrap(
        Path(args.v1_data_root),
        prefer_links=not args.copy,
        download_small=not args.no_download_small,
    )


if __name__ == "__main__":
    main()
