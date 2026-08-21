"""
retlesion_dataset.py
====================
Large-scale lesion-mask source: 1,472 EyePACS-derived fundus images with
per-pixel annotations for **microaneurysms** and **cotton-wool spots**
(= soft exudates).

    Original-Microaneurysm / Mask-Microaneurysm    1,435 pairs   -> MA
    Original-ischemic      / Mask-ischemic           471 pairs   -> SE (CWS)
    (434 images carry BOTH annotations)

Why this matters: IDRiD supplies only ~43 training images, which is why the
segmentation GCG comparison came back null with +/-0.07 run-to-run variance.
This adds ~33x more MA-annotated data -- MA being the hardest, weakest lesion.

Two things this loader has to get right
--------------------------------------
1. **Per-channel validity.** Only MA and SE are annotated here; HE and EX are
   *unknown*, not absent. Each sample therefore carries ``valid[C]`` so the loss
   ignores unannotated channels (see ``fundus_utils.focal_tversky_loss``).
   Without this we would teach the model "these images contain no haemorrhages".

2. **Scale matching.** These images are pre-downsampled to 896x896 with the FOV
   filling the frame, whereas IDRiD is ~3400px wide. A 512px patch here covers
   ~57% of the retina but only ~15% in IDRiD, so a microaneurysm spans ~3.8x
   fewer pixels. Mixing the two naively shows the network the same lesion at
   incompatible scales. With ``scale_match=True`` (default) we crop the same
   *fraction of retina* as an IDRiD patch and upsample it to ``patch_size``, so
   lesions occupy comparable pixel extents.

   Caveat worth stating in any writeup: upsampling equalises apparent scale but
   cannot restore detail lost when the source images were downsampled to 896.

Empty masks (an image filed under a lesion but with an all-zero mask) are
dropped by default -- they are annotation gaps, not confirmed negatives.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import dihedral_jitter, fov_bbox, make_rng
from idrid_dataset import DEFAULT_LESIONS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

KAGGLE_SLUG = "ajithdari/retinal-lesion-segmentation-dataset"

# (image dir, mask dir, mask filename suffix, lesion code in DEFAULT_LESIONS)
SOURCES = [
    ("Original-Microaneurysm", "Mask-Microaneurysm", "__microaneurysm", "MA"),
    ("Original-ischemic", "Mask-ischemic", "__cotton_wool_spots", "SE"),
]
# IDRiD reference: a 512px patch spans ~15% of a ~3400px-wide FOV.
IDRID_PATCH_FRAC = 512.0 / 3400.0


def download_retlesion(slug: str = KAGGLE_SLUG) -> str:
    import kagglehub
    return kagglehub.dataset_download(slug)


def resolve_base(root: str) -> str:
    """Return the dir containing the Original-*/Mask-* folders."""
    if os.path.isdir(os.path.join(root, SOURCES[0][0])):
        return root
    cand = os.path.join(root, "Dataset")
    if os.path.isdir(os.path.join(cand, SOURCES[0][0])):
        return cand
    for dirpath, dirnames, _ in os.walk(root):
        if SOURCES[0][0] in dirnames:
            return dirpath
    raise FileNotFoundError(f"Could not locate Original-Microaneurysm under {root!r}")


class RetLesionDataset(Dataset):
    """(image[3,H,W], masks[C,H,W], valid[C]) with C = len(lesions).

    Parameters
    ----------
    root : dataset root (or the dir holding Original-*/Mask-*).
    lesions : ordered lesion codes -> channels (default IDRiD's MA/HE/EX/SE).
    patch_size : output crop size.
    scale_match : crop the IDRiD-equivalent retina fraction then upsample.
    drop_empty : skip images whose only annotation is an all-zero mask.
    fg_bias : probability a crop is centred on a lesion pixel.
    """

    def __init__(
        self,
        root: str,
        lesions: Sequence[str] = DEFAULT_LESIONS,
        patch_size: int = 512,
        scale_match: bool = True,
        drop_empty: bool = True,
        fg_bias: float = 0.7,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base = resolve_base(root)
        self.lesions = list(lesions)
        self.patch_size = patch_size
        self.scale_match = scale_match
        self.fg_bias = fg_bias
        self.augment = augment
        self.seed = seed

        # stem -> {lesion_code: (image_path, mask_path)}
        index: Dict[str, Dict[str, Tuple[str, str]]] = {}
        for odir, mdir, suf, code in SOURCES:
            if code not in self.lesions:
                continue
            op, mp = os.path.join(self.base, odir), os.path.join(self.base, mdir)
            if not os.path.isdir(op):
                continue
            masks = {os.path.splitext(f)[0].replace(suf, ""): f for f in os.listdir(mp)}
            for f in os.listdir(op):
                stem = os.path.splitext(f)[0]
                if stem in masks:
                    index.setdefault(stem, {})[code] = (
                        os.path.join(op, f), os.path.join(mp, masks[stem]))

        self.items: List[Tuple[str, Dict[str, Tuple[str, str]]]] = []
        self.n_dropped = 0
        for stem, ann in sorted(index.items()):
            if drop_empty:
                # keep only lesions whose mask actually contains positives
                ann = {c: p for c, p in ann.items() if self._mask_nonempty(p[1])}
                if not ann:
                    self.n_dropped += 1
                    continue
            self.items.append((stem, ann))

    @staticmethod
    def _mask_nonempty(path: str) -> bool:
        try:
            return bool((np.asarray(Image.open(path).convert("L")) > 127).any())
        except Exception:  # noqa
            return False

    @property
    def num_classes(self) -> int:
        return len(self.lesions)

    def counts(self) -> Dict[str, int]:
        out = {c: 0 for c in self.lesions}
        for _, ann in self.items:
            for c in ann:
                out[c] += 1
        return out

    def __len__(self) -> int:
        return len(self.items)

    def load_full(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """FOV-cropped native-resolution image + mask stack + validity vector."""
        _, ann = self.items[idx]
        img_path = next(iter(ann.values()))[0]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        masks = np.zeros((len(self.lesions), H, W), dtype=np.uint8)
        valid = torch.zeros(len(self.lesions))
        for code, (_, mpath) in ann.items():
            ci = self.lesions.index(code)
            m = np.asarray(Image.open(mpath).convert("L"))
            masks[ci] = (m > 127).astype(np.uint8)
            valid[ci] = 1.0                      # this lesion IS annotated
        r0, r1, c0, c1 = fov_bbox(img)
        return img[r0:r1, c0:c1], masks[:, r0:r1, c0:c1], valid

    def __getitem__(self, idx: int):
        rng = make_rng(self.seed, idx)
        img, masks, valid = self.load_full(idx)
        H, W = img.shape[:2]

        # Crop size: match IDRiD's retina-fraction, then upsample to patch_size.
        crop = int(round(min(H, W) * IDRID_PATCH_FRAC)) if self.scale_match else self.patch_size
        crop = max(32, min(crop, H, W))

        # Unannotated channels are all-zero by construction, so masks.any() only
        # fires on annotated lesions. (The previous `& valid...any()` reduced to
        # a scalar `& True` — it looked per-channel but wasn't.)
        fg = masks.any(axis=0)
        if self.fg_bias > 0 and fg.sum() > 0 and rng.random() < self.fg_bias:
            ys, xs = np.where(fg)
            k = int(rng.integers(len(ys)))
            top = int(np.clip(ys[k] - crop // 2, 0, H - crop))
            left = int(np.clip(xs[k] - crop // 2, 0, W - crop))
        else:
            top = int(rng.integers(0, H - crop + 1))
            left = int(rng.integers(0, W - crop + 1))
        img = img[top:top + crop, left:left + crop]
        masks = masks[:, top:top + crop, left:left + crop]

        if crop != self.patch_size:              # scale to the shared patch size
            size = (self.patch_size, self.patch_size)
            img = np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))
            masks = np.stack([np.asarray(Image.fromarray(masks[c]).resize(size, Image.NEAREST))
                              for c in range(masks.shape[0])], axis=0)

        if self.augment:
            # Shared dihedral+jitter (fundus_utils), same as every other lesion
            # loader — rot90 applies since crops are square by construction.
            img, masks = dihedral_jitter(img, masks, rng)

        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.from_numpy(masks.astype(np.float32)), valid.clone()
