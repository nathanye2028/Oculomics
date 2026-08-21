"""
rfmid_dataset.py
================
RFMiD (Retinal Fundus Multi-Disease Image Dataset) **multi-label classification**
loader — 1,920 train / 640 val / 640 test fundus images, 45 disease labels plus
a binary ``Disease_Risk``.

IMPORTANT: RFMiD ships **image-level labels only — no segmentation masks**. It
therefore cannot train lesion segmentation. Its value here is *in-domain
pretraining*: pretrain the MobileNetV3 encoder on these 1,920 retinas
(:mod:`pretrain_encoder`), then fine-tune on IDRiD's 54 masked images. Transfer
from fundus photographs beats ImageNet transfer for a 54-image target task.

Works with either Kaggle mirror (layouts differ slightly); the split dirs and
label CSVs are located by globbing:
    kagglehub.dataset_download("andrewmvd/retinal-disease-classification")
    kagglehub.dataset_download("ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid")
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import crop_to_fov, make_rng

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Kaggle slugs known to carry RFMiD (either works).
KAGGLE_SLUGS = (
    "ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid",
    "andrewmvd/retinal-disease-classification",
)

# Substrings identifying each split in file/dir names, in priority order.
_SPLIT_KEYS = {
    "train": ("train",),
    "val": ("valid", "eval", "val"),
    "test": ("test",),
}
_IMG_EXTS = (".png", ".jpg", ".jpeg")


def download_rfmid(slug: Optional[str] = None) -> str:
    """Download (or reuse cached) RFMiD and return the dataset root."""
    import kagglehub
    slugs = (slug,) if slug else KAGGLE_SLUGS
    last = None
    for s in slugs:
        try:
            return kagglehub.dataset_download(s)
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"Could not download RFMiD from {slugs}: {last}")


def _find_split(root: str, split: str) -> Tuple[str, str]:
    """Locate (labels_csv, images_dir) for a split, mirror-agnostically."""
    keys = _SPLIT_KEYS[split]
    csvs = [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
            if any(k in os.path.basename(p).lower() for k in keys)]
    if not csvs:
        raise FileNotFoundError(f"No {split!r} label CSV under {root!r}")
    csv_path = sorted(csvs, key=len)[0]

    # Prefer an image dir that sits near the CSV and matches the split name.
    cand = [d for d in glob.glob(os.path.join(root, "**", ""), recursive=True)
            if any(k in d.lower() for k in keys)
            and any(f.lower().endswith(_IMG_EXTS) for f in os.listdir(d)[:50] or [])]
    if not cand:
        raise FileNotFoundError(f"No {split!r} image directory under {root!r}")
    # Deepest match is usually the leaf folder holding the images.
    images_dir = sorted(cand, key=lambda d: (-d.count(os.sep), len(d)))[0]
    return csv_path, images_dir.rstrip(os.sep)


class RFMiDDataset(Dataset):
    """RFMiD multi-label classification dataset.

    Returns ``(image[3,H,W], labels[C])`` where labels are binary (BCE-ready).

    Parameters
    ----------
    root : RFMiD download root (from :func:`download_rfmid`).
    split : 'train' | 'val' | 'test'.
    image_size : square resize.
    labels : which columns to predict. Default = all disease columns + Disease_Risk.
    fov_crop / augment : standard fundus preprocessing + light flips.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        labels: Optional[Sequence[str]] = None,
        fov_crop: bool = True,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        csv_path, images_dir = _find_split(root, split)
        self.images_dir = images_dir
        self.image_size = image_size
        self.fov_crop = fov_crop
        self.augment = augment
        self.seed = seed

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        id_col = df.columns[0]                       # 'ID'
        self.label_cols: List[str] = list(labels) if labels else [
            c for c in df.columns if c != id_col]

        on_disk = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)
                   if f.lower().endswith(_IMG_EXTS)}
        df["_file"] = df[id_col].astype(str).map(on_disk)
        missing = int(df["_file"].isna().sum())
        if missing:
            df = df.dropna(subset=["_file"])
        self.missing = missing

        self.files: List[str] = df["_file"].tolist()
        self.labels = torch.as_tensor(
            df[self.label_cols].to_numpy(dtype=np.float32))   # [N, C]

    @property
    def num_classes(self) -> int:
        return len(self.label_cols)

    def pos_weight(self) -> torch.Tensor:
        """Per-label pos_weight = #neg/#pos, for imbalance-aware BCE."""
        pos = self.labels.sum(0).clamp(min=1.0)
        neg = self.labels.shape[0] - pos
        return (neg / pos).clamp(max=50.0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rng = make_rng(self.seed, idx)
        img = Image.open(os.path.join(self.images_dir, self.files[idx])).convert("RGB")
        if self.fov_crop:
            arr, _ = crop_to_fov(np.asarray(img))
            img = Image.fromarray(arr)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.augment and bool(rng.integers(0, 2)):
            arr = arr[:, ::-1, :].copy()
        x = torch.from_numpy(arr.transpose(2, 0, 1) / 255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, self.labels[idx]
