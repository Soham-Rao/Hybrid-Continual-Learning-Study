"""Download Tiny-ImageNet from Hugging Face and export loader-friendly folders.

Exports to:

    Project/data_local/tiny-imagenet-200/
        train/<wnid>/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt

Default source dataset:
    zh-plus/tiny-imagenet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "Project" / "data_local" / "tiny-imagenet-200"


def _ensure_clean_target(out_dir: Path, force: bool) -> None:
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    if (train_dir.exists() or val_dir.exists()) and not force:
        raise SystemExit(
            f"Target already looks populated: {out_dir}\n"
            "Use --force if you want to overwrite exported files."
        )
    train_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)


def _export_train(split, label_names: list[str], out_dir: Path) -> None:
    total = len(split)
    for idx, row in enumerate(split):
        wnid = label_names[row["label"]]
        img_dir = out_dir / "train" / wnid / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{wnid}_{idx:05d}.JPEG"
        row["image"].save(img_path, format="JPEG", quality=95)
        if (idx + 1) % 5000 == 0 or idx + 1 == total:
            print(f"train_exported={idx + 1}/{total}")


def _export_val(split, label_names: list[str], out_dir: Path) -> None:
    ann_path = out_dir / "val" / "val_annotations.txt"
    total = len(split)
    with ann_path.open("w", encoding="utf-8") as ann_fh:
        for idx, row in enumerate(split):
            wnid = label_names[row["label"]]
            filename = f"val_{idx:05d}.JPEG"
            img_path = out_dir / "val" / "images" / filename
            row["image"].save(img_path, format="JPEG", quality=95)
            ann_fh.write(f"{filename}\t{wnid}\t0\t0\t64\t64\n")
            if (idx + 1) % 2000 == 0 or idx + 1 == total:
                print(f"val_exported={idx + 1}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Tiny-ImageNet from Hugging Face and export to folder layout."
    )
    parser.add_argument(
        "--repo-id",
        default="zh-plus/tiny-imagenet",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="Export root directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow export into an existing target directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_clean_target(out_dir, force=args.force)

    print(f"repo_id={args.repo_id}")
    print(f"out_dir={out_dir}")

    ds = load_dataset(args.repo_id)
    if "train" not in ds or "valid" not in ds:
        raise SystemExit(
            f"Expected train/valid splits in {args.repo_id}, found: {list(ds.keys())}"
        )

    label_feature = ds["train"].features["label"]
    label_names = list(label_feature.names)
    print(f"n_classes={len(label_names)}")
    print(f"train_rows={len(ds['train'])}")
    print(f"valid_rows={len(ds['valid'])}")

    _export_train(ds["train"], label_names, out_dir)
    _export_val(ds["valid"], label_names, out_dir)

    meta = {
        "source": args.repo_id,
        "train_rows": len(ds["train"]),
        "valid_rows": len(ds["valid"]),
        "n_classes": len(label_names),
    }
    (out_dir / "hf_export_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print("tiny_imagenet_export_status=complete")


if __name__ == "__main__":
    main()
