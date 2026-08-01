"""
ddr_dataset.py
==============
DDR **DR-grading** loader (the Kaggle mirror's ``DR_grading`` subset):
12,521 fundus images labelled with an ICDR-style grade in ``DR_grading.csv``
(``id_code, diagnosis``).

This mirror ships **no lesion masks**, so DDR cannot train segmentation. Its
value is scale: 12.5k in-domain retinas for encoder pretraining
(:mod:`pretrain_encoder`), ~7x more than RFMiD alone.

Grade 5 appears in DDR as "ungradable"; pass ``drop_ungradable=True`` (default)
to exclude those images from the pretraining corpus.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import crop_to_fov, make_rng

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

KAGGLE_SLUG = "mariaherrerot/ddrdataset"
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def download_ddr(slug: str = KAGGLE_SLUG) -> str:
    import kagglehub
    return kagglehub.dataset_download(slug)


def _resolve(root: str):
    """Locate (csv_path, images_dir) inside the DDR download."""
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No DDR csv under {root!r}")
    csv_path = sorted(csvs, key=len)[0]
    dirs = [d for d in glob.glob(os.path.join(root, "**", ""), recursive=True)
            if any(f.lower().endswith(_IMG_EXTS) for f in (os.listdir(d)[:30] or []))]
    if not dirs:
        raise FileNotFoundError(f"No DDR image directory under {root!r}")
    images_dir = sorted(dirs, key=lambda d: (-d.count(os.sep), len(d)))[0]
    return csv_path, images_dir.rstrip(os.sep)


class DDRGradingDataset(Dataset):
    """DDR grading images -> (image[3,H,W], grade:int).

    Used as a *pretraining* corpus; ``num_classes`` is 5 (ICDR 0-4).
    """

    def __init__(self, root: str, image_size: int = 224, fov_crop: bool = True,
                 augment: bool = False, drop_ungradable: bool = True,
                 seed: int = 42, limit: Optional[int] = None) -> None:
        super().__init__()
        csv_path, self.images_dir = _resolve(root)
        self.image_size = image_size
        self.fov_crop = fov_crop
        self.augment = augment
        self.seed = seed

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        id_col, lab_col = df.columns[0], df.columns[1]
        if drop_ungradable:
            df = df[df[lab_col] <= 4]

        on_disk = {f: f for f in os.listdir(self.images_dir)
                   if f.lower().endswith(_IMG_EXTS)}
        stems = {os.path.splitext(f)[0]: f for f in on_disk}
        def _match(v):
            v = str(v)
            return on_disk.get(v) or stems.get(os.path.splitext(v)[0])
        df["_file"] = df[id_col].map(_match)
        self.missing = int(df["_file"].isna().sum())
        df = df.dropna(subset=["_file"])
        if limit:
            df = df.iloc[:limit]

        self.files: List[str] = df["_file"].tolist()
        self.labels = torch.as_tensor(df[lab_col].to_numpy(), dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return 5

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(self.labels, minlength=self.num_classes)

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
