"""
vessel_dataset.py
=================
Retinal **vessel** segmentation dataset (800 paired images, 2048x2048):

    train/Original + train/Ground truth   600 pairs
    test/Original  + test/Ground truth    200 pairs

Why a vessel set matters for a *lesion* project
-----------------------------------------------
Vessels are NOT lesions, so this cannot be merged into MA/HE/EX/SE training.
Its value is as the strongest available **pretraining task**, for three reasons:

1. It is a *dense segmentation* task, so it pretrains the **decoder as well as
   the encoder** — unlike DDR/RFMiD classification, which only pretrains the
   encoder. Nearly the whole GCG-U-Net gets initialised from real data.
2. Vessels are thin, faint, high-frequency structures — the same visual regime
   as microaneurysms. Features tuned to fine vasculature should transfer to
   tiny-lesion detection far better than natural-image features.
3. 800 images vs IDRiD's 54 (≈15x), all densely annotated.

Clinically this is also the classic MA confounder: microaneurysms are easily
mistaken for vessel cross-sections and bifurcations, so a network that already
models the vascular tree starts from a better place.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import fov_bbox, make_rng, random_patch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

KAGGLE_SLUG = "nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation"
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_SPLIT_DIR = {"train": "train", "test": "test"}


def download_vessels(slug: str = KAGGLE_SLUG) -> str:
    import kagglehub
    return kagglehub.dataset_download(slug)


class VesselSegDataset(Dataset):
    """Paired fundus / vessel-mask dataset -> (image[3,H,W], mask[1,H,W]).

    Parameters
    ----------
    root : dataset root (contains ``train/`` and ``test/``).
    split : 'train' | 'test'.
    image_size : output resolution when ``patch_size`` is None.
    patch_size : if set, take native-resolution crops instead of resizing.
    fov_crop : trim the black border first.
    augment : joint flips on image + mask.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 512,
        patch_size: Optional[int] = None,
        fov_crop: bool = True,
        augment: bool = False,
        fg_bias: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if split not in _SPLIT_DIR:
            raise ValueError(f"split must be train/test, got {split!r}")
        base = os.path.join(root, _SPLIT_DIR[split])
        self.img_dir = os.path.join(base, "Original")
        self.msk_dir = os.path.join(base, "Ground truth")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Missing {self.img_dir!r}")

        imgs = {f for f in os.listdir(self.img_dir) if f.lower().endswith(_IMG_EXTS)}
        msks = {f for f in os.listdir(self.msk_dir) if f.lower().endswith(_IMG_EXTS)}
        self.files: List[str] = sorted(imgs & msks)     # names pair exactly

        self.image_size = image_size
        self.patch_size = patch_size
        self.fov_crop = fov_crop
        self.augment = augment
        self.fg_bias = fg_bias
        self.seed = seed

    @property
    def num_classes(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.files)

    def load_full(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        f = self.files[idx]
        img = np.asarray(Image.open(os.path.join(self.img_dir, f)).convert("RGB"))
        mk = np.asarray(Image.open(os.path.join(self.msk_dir, f)).convert("L"))
        masks = (mk > 127).astype(np.uint8)[None]        # [1,H,W]
        if self.fov_crop:
            r0, r1, c0, c1 = fov_bbox(img)
            img = img[r0:r1, c0:c1]
            masks = masks[:, r0:r1, c0:c1]
        return img, masks

    def __getitem__(self, idx: int):
        rng = make_rng(self.seed, idx)
        img_np, masks = self.load_full(idx)

        if self.patch_size:
            img_np, masks = random_patch(img_np, masks, self.patch_size, rng, self.fg_bias)
        else:
            size = (self.image_size, self.image_size)
            img_np = np.asarray(Image.fromarray(img_np).resize(size, Image.BILINEAR))
            masks = np.stack([np.asarray(Image.fromarray(masks[c]).resize(size, Image.NEAREST))
                              for c in range(masks.shape[0])], axis=0)

        if self.augment:
            if bool(rng.integers(0, 2)):
                img_np = img_np[:, ::-1, :].copy(); masks = masks[:, :, ::-1].copy()
            if bool(rng.integers(0, 2)):
                img_np = img_np[::-1, :, :].copy(); masks = masks[:, ::-1, :].copy()

        x = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.from_numpy(masks.astype(np.float32))
