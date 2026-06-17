"""
idrid_dataset.py
================
Real lesion-segmentation dataset for **IDRiD** (Indian Diabetic Retinopathy
Image Dataset), Segmentation subset.

Yields genuine per-pixel lesion ground truth as a **multi-label mask tensor**
``[C, H, W]`` (one binary channel per lesion type) paired with the fundus
image — the supervision the GCG blocks need (one channel per distinct lesion).

IDRiD lesion classes (binary .tif masks, named ``IDRiD_NN_<suffix>.tif``):
    MA  microaneurysms      MA   |  HE  haemorrhages   HE
    EX  hard exudates       EX   |  SE  soft exudates  SE   (present in ~half)
    OD  optic disc          OD   (anatomical, not a lesion)

A missing mask file for an image means that lesion is absent -> all-zero
channel (correct supervision).

Resolution & tiny lesions
-------------------------
Images are 4288x2848; microaneurysms are ~0.05% of pixels and vanish under a
global downscale. Two modes:
  * ``patch_size`` set  -> random crop at NATIVE resolution (lesion-biased),
    preserving small lesions. Use this for training.
  * ``patch_size=None`` -> FOV-crop then resize to ``image_size`` (whole image).
    Use this for visualisation / global eval; for held-out scoring prefer
    ``fundus_utils.tiled_predict`` at native resolution.

``fov_crop`` strips the black border first (shared with the rest of the
pipeline via :mod:`fundus_utils`).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import fov_bbox, make_rng, random_patch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# lesion code -> (groundtruth subfolder, filename suffix)
LESION_FOLDERS: Dict[str, Tuple[str, str]] = {
    "MA": ("1. Microaneurysms", "MA"),
    "HE": ("2. Haemorrhages", "HE"),
    "EX": ("3. Hard Exudates", "EX"),
    "SE": ("4. Soft Exudates", "SE"),
    "OD": ("5. Optic Disc", "OD"),
}
DEFAULT_LESIONS = ("MA", "HE", "EX", "SE")   # the 4 DR lesions; OD excluded by default

_SPLIT_DIR = {"train": "a. Training Set", "test": "b. Testing Set"}


def resolve_seg_base(root: str) -> str:
    """Locate the 'A. Segmentation' base dir under an IDRiD download root.

    Robust to the kagglehub unzip quirk that leaves a literal 'A.%20Segmentation'
    outer folder. Returns the dir that directly contains '1. Original Images'
    and '2. All Segmentation Groundtruths'.
    """
    hits = glob.glob(os.path.join(root, "**", "1. Original Images"), recursive=True)
    for h in hits:
        base = os.path.dirname(h)
        if os.path.isdir(os.path.join(base, "2. All Segmentation Groundtruths")):
            return base
    raise FileNotFoundError(f"Could not find IDRiD 'A. Segmentation' base under {root!r}")


class IDRiDSegDataset(Dataset):
    """IDRiD lesion-segmentation dataset (multi-label per-lesion masks).

    Parameters
    ----------
    root_or_base : IDRiD download root or the 'A. Segmentation' dir.
    split : 'train' | 'test'.
    image_size : square resolution used in whole-image mode (patch_size=None).
    lesions : ordered lesion codes -> output channels (default 4 DR lesions).
    fov_crop : crop the black border to the retinal field of view first.
    patch_size : if set, return a native-resolution random crop of this size
                 (training); else resize the whole FOV image to image_size.
    fg_bias : prob. a training patch is centred on a lesion pixel (vs uniform).
    augment : light joint geometric augmentation (flips).
    seed : base seed for reproducible-yet-diverse per-sample augmentation.
    """

    def __init__(
        self,
        root_or_base: str,
        split: str = "train",
        image_size: int = 512,
        lesions: Sequence[str] = DEFAULT_LESIONS,
        fov_crop: bool = True,
        patch_size: Optional[int] = None,
        fg_bias: float = 0.7,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if os.path.isdir(os.path.join(root_or_base, "2. All Segmentation Groundtruths")):
            self.base = root_or_base
        else:
            self.base = resolve_seg_base(root_or_base)

        if split not in _SPLIT_DIR:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.image_size = image_size
        self.lesions = list(lesions)
        for code in self.lesions:
            if code not in LESION_FOLDERS:
                raise KeyError(f"Unknown lesion code {code!r}; choices: {list(LESION_FOLDERS)}")
        self.fov_crop = fov_crop
        self.patch_size = patch_size
        self.fg_bias = fg_bias
        self.augment = augment
        self.seed = seed

        self.img_dir = os.path.join(self.base, "1. Original Images", _SPLIT_DIR[split])
        self.gt_dir = os.path.join(self.base, "2. All Segmentation Groundtruths", _SPLIT_DIR[split])

        self.files: List[str] = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    @property
    def num_classes(self) -> int:
        return len(self.lesions)

    def __len__(self) -> int:
        return len(self.files)

    # -- helpers ----------------------------------------------------------- #
    def _mask_path(self, stem: str, code: str) -> Optional[str]:
        folder, suffix = LESION_FOLDERS[code]
        for ext in (".tif", ".tiff", ".png"):
            p = os.path.join(self.gt_dir, folder, f"{stem}_{suffix}{ext}")
            if os.path.exists(p):
                return p
        return None

    def load_full(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load FOV-cropped native-resolution (image[H,W,3], masks[C,H,W])."""
        fname = self.files[idx]
        stem = os.path.splitext(fname)[0]
        img = np.asarray(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        H, W = img.shape[:2]
        masks = np.zeros((len(self.lesions), H, W), dtype=np.uint8)
        for c, code in enumerate(self.lesions):
            p = self._mask_path(stem, code)
            if p is not None:
                m = np.asarray(Image.open(p).convert("L"))
                masks[c] = (m > 0).astype(np.uint8)
        if self.fov_crop:
            r0, r1, c0, c1 = fov_bbox(img)
            img = img[r0:r1, c0:c1]
            masks = masks[:, r0:r1, c0:c1]
        return img, masks

    def _to_tensors(self, img_np: np.ndarray, masks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        mask_t = torch.from_numpy(masks.astype(np.float32))
        return img_t, mask_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = make_rng(self.seed, idx)                 # diverse per worker+epoch, reproducible
        img_np, masks = self.load_full(idx)

        if self.patch_size:
            # Native-resolution lesion-biased crop -> preserves tiny lesions.
            img_np, masks = random_patch(img_np, masks, self.patch_size, rng, self.fg_bias)
        else:
            size = (self.image_size, self.image_size)
            img_np = np.asarray(Image.fromarray(img_np).resize(size, Image.BILINEAR))
            masks = np.stack([
                np.asarray(Image.fromarray(masks[c]).resize(size, Image.NEAREST))
                for c in range(masks.shape[0])
            ], axis=0)

        if self.augment and bool(rng.integers(0, 2)):
            img_np = img_np[:, ::-1, :].copy()
            masks = masks[:, :, ::-1].copy()

        return self._to_tensors(img_np, masks)
