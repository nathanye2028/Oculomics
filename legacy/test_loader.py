"""
test_loader.py
==============
Smoke test for :class:`dataset.MBRSETDataset` against the public Kaggle
"Retinal Disease Detection" dataset
(``mohamedabdalkader/retinal-disease-detection``).

It downloads the dataset (cached after the first run), instantiates the custom
Dataset, pulls **exactly one batch** through a ``DataLoader``, and prints the
tensor shapes. Exits non-zero on any failure.

Run:
    python test_loader.py                 # train split, dr_grade task
    python test_loader.py --split valid --task dr_referable --batch-size 16
"""
from __future__ import annotations
# LEGACY: moved to legacy/ on 2026-09-01 (superseded; kept for reference only).
# The shim below lets it still import the repo modules from the parent dir.
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the local dataset.py is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset  # noqa: E402

KAGGLE_SLUG = "mohamedabdalkader/retinal-disease-detection"
SUBDIR = "Diabetic Retinopathy"  # top-level folder inside the archive

# This dataset's CSV columns -> the clinical-label schema dataset.py expects.
COLUMN_MAP = {
    "Image name": "file",
    "Retinopathy grade": "final_icdr",     # 0-4 ICDR severity grade
    "Risk of macular edema": "final_edema",  # 0/1 macular-edema risk
}


def resolve_split_dir(split: str) -> str:
    """Download (or reuse cached) dataset and return the chosen split dir."""
    import kagglehub

    root = kagglehub.dataset_download(KAGGLE_SLUG)
    split_dir = os.path.join(root, SUBDIR, split)
    if not os.path.isdir(split_dir):
        available = (
            os.listdir(os.path.join(root, SUBDIR))
            if os.path.isdir(os.path.join(root, SUBDIR))
            else os.listdir(root)
        )
        raise FileNotFoundError(
            f"Split {split!r} not found under {os.path.join(root, SUBDIR)!r}. "
            f"Available: {available}"
        )
    return split_dir


def main() -> int:
    p = argparse.ArgumentParser(description="One-batch smoke test for MBRSETDataset.")
    p.add_argument("--split", default="train", choices=["train", "valid", "test"])
    p.add_argument("--task", default="dr_grade",
                   help="Label task key (e.g. dr_grade, dr_referable, edema).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    split_dir = resolve_split_dir(args.split)
    csv_path = os.path.join(split_dir, "annotations.csv")
    images_dir = os.path.join(split_dir, "images")
    print(f"[info] split dir : {split_dir}")
    print(f"[info] images dir: {images_dir}")

    # Load annotations and remap columns onto the expected schema.
    df = pd.read_csv(csv_path).rename(columns=COLUMN_MAP)
    print(f"[info] annotations: {len(df)} rows, columns={list(df.columns)}")

    # MBRSETDataset 'train' split applies the artifact-resilient augmentations;
    # other splits use the deterministic resize + center-crop pipeline.
    ds = MBRSETDataset(
        csv=df,
        images_dir=images_dir,
        task=args.task,
        split=args.split if args.split == "train" else "val",
        file_col="file",
        image_size=args.image_size,
        drop_missing_files=True,
    )
    print(f"[info] dataset    : {ds}")
    counts = ds.class_counts()
    if counts is not None:
        print(f"[info] class counts: {counts.tolist()}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Pull exactly one batch.
    batch = next(iter(loader))

    print("\n=== one batch ===")
    print(f"image : shape={tuple(batch['image'].shape)} dtype={batch['image'].dtype}")
    print(f"label : shape={tuple(batch['label'].shape)} dtype={batch['label'].dtype}")
    print(f"file  : {len(batch['file'])} names, e.g. {batch['file'][0]}")
    print(f"label values: {batch['label'].tolist()}")

    # Sanity assertions.
    b = min(args.batch_size, len(ds))
    assert batch["image"].shape == (b, 3, args.image_size, args.image_size), batch["image"].shape
    assert batch["image"].dtype == torch.float32
    assert batch["label"].shape[0] == b
    print("\nOK: pulled one batch successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
