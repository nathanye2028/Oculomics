"""
seg_dataset.py
==============
A segmentation-style ``Dataset`` for retinal fundus images.

Built to exercise the U-Net / MobileNetV3 + GCG segmentation model
(:mod:`model`) on real fundus images. It yields ``(image, mask)`` pairs:

    image : FloatTensor [3, H, W]  (ImageNet-normalised)
    mask  : FloatTensor [1, H, W]  (binary lesion-vs-background target)

IMPORTANT — placeholder masks
-----------------------------
The RFMiD dataset (``andrewmvd/retinal-disease-classification``) ships only
**image-level** multi-label disease annotations — it has **no pixel-level
lesion masks**. Real lesion segmentation needs per-pixel ground truth, which
this dataset does not provide.

To still run and validate the full segmentation pipeline end-to-end on these
real images, this loader synthesises a *deterministic placeholder* mask
(``mask_mode``):
  * ``"fov"``    : a circular field-of-view proxy (bright retinal disc vs.
                   dark border) derived by thresholding image luminance — a
                   coarse, reproducible target so the loss/optimiser have a
                   real signal to fit during a smoke test.
  * ``"zeros"``  : all-background mask (pure plumbing check).

These are NOT lesion annotations. To train genuine lesion segmentation, point
``masks_dir`` at a dataset that provides masks (e.g. IDRiD / e-ophtha) — the
class will load ``<stem>`` masks from there instead of synthesising them.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class RFMiDSegDataset(Dataset):
    """Fundus segmentation dataset over a directory of images.

    Parameters
    ----------
    images_dir : directory of fundus images (e.g. RFMiD ``Training``).
    image_size : square output resolution.
    fraction   : keep only this fraction of images (deterministic head slice).
                 Default 1/6 per request ("1/2 of 1/3").
    masks_dir  : if given, load real masks ``<stem>.png`` from here; otherwise
                 synthesise placeholder masks (see module docstring).
    mask_mode  : "fov" | "zeros" when synthesising placeholder masks.
    augment    : light geometric augmentation applied jointly to image+mask.
    """

    def __init__(
        self,
        images_dir: str,
        image_size: int = 256,
        fraction: float = 1.0 / 6.0,
        masks_dir: Optional[str] = None,
        mask_mode: str = "fov",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        self.images_dir = images_dir
        self.image_size = image_size
        self.masks_dir = masks_dir
        self.mask_mode = mask_mode
        self.augment = augment

        files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith(_IMG_EXTS)
        )
        # Deterministic numeric sort when filenames are integer IDs (RFMiD).
        try:
            files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        except ValueError:
            pass

        if 0.0 < fraction < 1.0:
            keep = max(1, int(round(len(files) * fraction)))
            files = files[:keep]
        self.files: List[str] = files
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    # -- mask synthesis ---------------------------------------------------- #
    def _placeholder_mask(self, img: Image.Image) -> Image.Image:
        if self.mask_mode == "zeros":
            return Image.new("L", img.size, 0)
        # "fov": threshold luminance to get the bright retinal field of view.
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        thr = gray.mean() * 0.5
        m = (gray > thr).astype(np.uint8) * 255
        return Image.fromarray(m, mode="L")

    def _load_mask(self, fname: str, img: Image.Image) -> Image.Image:
        if self.masks_dir:
            stem = os.path.splitext(fname)[0]
            for ext in _IMG_EXTS:
                p = os.path.join(self.masks_dir, stem + ext)
                if os.path.exists(p):
                    return Image.open(p).convert("L")
        return self._placeholder_mask(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        mask = self._load_mask(fname, img)

        size = (self.image_size, self.image_size)
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        if self.augment and bool(self._rng.integers(0, 2)):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        mask_t = torch.from_numpy((np.asarray(mask, dtype=np.float32) / 255.0)[None] > 0.5).float()
        return img_t, mask_t
