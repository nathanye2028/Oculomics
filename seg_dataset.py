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

from fundus_utils import crop_to_fov, make_rng

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
        fov_crop: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        self.images_dir = images_dir
        self.image_size = image_size
        self.masks_dir = masks_dir
        self._missing_masks: set = set()
        self.mask_mode = mask_mode
        self.augment = augment
        self.fov_crop = fov_crop
        self.seed = seed

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

    def _load_mask(self, fname: str, img: Image.Image, bbox=None) -> Image.Image:
        if self.masks_dir:
            stem = os.path.splitext(fname)[0]
            for ext in _IMG_EXTS:
                p = os.path.join(self.masks_dir, stem + ext)
                if os.path.exists(p):
                    m = Image.open(p).convert("L")
                    if bbox is not None:                  # keep aligned with FOV crop
                        r0, r1, c0, c1 = bbox
                        m = Image.fromarray(np.asarray(m)[r0:r1, c0:c1])
                    return m
            # A silent placeholder here would mean "real masks requested, synthetic
            # masks trained on" with no sign anything went wrong (e.g. IDRiD-style
            # '<stem>_MA.tif' names never match '<stem>.<ext>').
            if fname not in self._missing_masks:
                self._missing_masks.add(fname)
                print(f"[warn] seg_dataset: no mask for {fname!r} under {self.masks_dir} "
                      f"({len(self._missing_masks)} missing so far) — using PLACEHOLDER mask")
        return self._placeholder_mask(img)                # generated from cropped img -> aligned

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = make_rng(self.seed, idx)        # per-call: diverse across workers, reproducible
        fname = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")

        bbox = None
        if self.fov_crop:
            arr, bbox = crop_to_fov(np.asarray(img))
            img = Image.fromarray(arr)

        mask = self._load_mask(fname, img, bbox)

        size = (self.image_size, self.image_size)
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        if self.augment and bool(rng.integers(0, 2)):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        # `> 0` (as in idrid_dataset), NOT `/255 > 0.5`: masks encoded {0,1} or as
        # red-on-black RGB (L-value ~76) binarise to all-zero under the old rule.
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)[None] > 0).float()
        return img_t, mask_t
