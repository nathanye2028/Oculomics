"""
idrid_dataset.py
================
Real lesion-segmentation dataset for **IDRiD** (Indian Diabetic Retinopathy
Image Dataset), Segmentation subset.

Unlike the placeholder masks in :mod:`seg_dataset`, this yields *genuine*
per-pixel lesion ground truth: each fundus image is paired with a
**multi-label mask tensor** ``[C, H, W]`` where each channel is a binary mask
for one lesion type. This is the supervision the GCG blocks need — one output
channel per distinct lesion, so gating can be tied to (and measured against)
specific pathologies.

IDRiD lesion classes (binary .tif masks, named ``IDRiD_NN_<suffix>.tif``):
    MA  microaneurysms      (1. Microaneurysms)
    HE  haemorrhages        (2. Haemorrhages)
    EX  hard exudates       (3. Hard Exudates)
    SE  soft exudates       (4. Soft Exudates)   -- present in only ~half
    OD  optic disc          (5. Optic Disc)      -- anatomical, not a lesion

A missing mask file for an image means that lesion is absent -> an all-zero
channel (correct supervision, not missing data).

Images are 4288x2848; lesions (esp. microaneurysms ~0.05% of pixels) are tiny,
so segmentation at low resolution is lossy. ``fov_crop`` removes the black
border first; ``image_size`` controls the working resolution.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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
    root_or_base : either the IDRiD download root or the 'A. Segmentation' dir.
    split : 'train' | 'test'.
    image_size : square working resolution.
    lesions : ordered lesion codes -> output channels (default 4 DR lesions).
    fov_crop : crop the black border to the retinal field of view first.
    augment : light joint geometric augmentation (flips).
    """

    def __init__(
        self,
        root_or_base: str,
        split: str = "train",
        image_size: int = 512,
        lesions: Sequence[str] = DEFAULT_LESIONS,
        fov_crop: bool = True,
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
        self.augment = augment

        self.img_dir = os.path.join(self.base, "1. Original Images", _SPLIT_DIR[split])
        self.gt_dir = os.path.join(self.base, "2. All Segmentation Groundtruths", _SPLIT_DIR[split])

        self.files: List[str] = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self._rng = np.random.default_rng(seed)

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

    @staticmethod
    def _fov_bbox(rgb: np.ndarray, tol: int = 12) -> Tuple[int, int, int, int]:
        """Bounding box of the non-black field of view (row0,row1,col0,col1)."""
        gray = rgb.max(axis=2)
        rows = np.where(gray.max(axis=1) > tol)[0]
        cols = np.where(gray.max(axis=0) > tol)[0]
        if len(rows) == 0 or len(cols) == 0:
            return 0, rgb.shape[0], 0, rgb.shape[1]
        return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        stem = os.path.splitext(fname)[0]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        W, H = img.size
        img_np = np.asarray(img)

        # Build full-res mask stack [C, H, W] from the per-lesion tifs.
        masks = np.zeros((len(self.lesions), H, W), dtype=np.uint8)
        for c, code in enumerate(self.lesions):
            p = self._mask_path(stem, code)
            if p is not None:
                m = np.asarray(Image.open(p).convert("L"))
                masks[c] = (m > 0).astype(np.uint8)

        # FOV crop (identical box for image + every mask).
        if self.fov_crop:
            r0, r1, c0, c1 = self._fov_bbox(img_np)
            img_np = img_np[r0:r1, c0:c1]
            masks = masks[:, r0:r1, c0:c1]

        # Resize: image bilinear, masks nearest (preserve binary lesion pixels).
        size = (self.image_size, self.image_size)
        img = Image.fromarray(img_np).resize(size, Image.BILINEAR)
        mask_resized = np.stack([
            np.asarray(Image.fromarray(masks[c]).resize(size, Image.NEAREST))
            for c in range(masks.shape[0])
        ], axis=0).astype(np.float32)

        # Joint augmentation.
        if self.augment and bool(self._rng.integers(0, 2)):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask_resized = mask_resized[:, :, ::-1].copy()

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        mask_t = torch.from_numpy(mask_resized)
        return img_t, mask_t
