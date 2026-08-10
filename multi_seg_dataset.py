"""
multi_seg_dataset.py
====================
Merge several lesion-segmentation datasets into one training set, with
**per-channel annotation validity**.

Why validity matters (this is the subtle, important part)
---------------------------------------------------------
IDRiD annotates MA, HE, EX, SE. e-ophtha (this mirror) annotates **EX only**.
If we loaded e-ophtha with MA/HE/SE as all-zero masks, we would be teaching the
model *"these images contain no microaneurysms"* — which is false; they are
simply **unannotated**. That injects systematically wrong supervision and would
degrade exactly the lesions we care most about.

So every sample carries a ``valid`` vector [C]: 1 = this lesion is annotated in
this image, 0 = unknown. The loss (``fundus_utils.focal_tversky_loss(..., valid=)``)
skips unannotated channels rather than treating them as negatives.

Current sources
---------------
* **IDRiD**    54 train / 27 test, masks for MA, HE, EX, SE   (valid = all)
* **e-ophtha** 47 images, masks for EX only                   (valid = EX)
* **FGADR**    1474 train / 368 test, masks for MA, HE, EX, SE (valid = all)

FGADR dominates the mixture by ~18x, so treat it as the primary training signal
and IDRiD/e-ophtha as held-out-distribution checks rather than equal partners.

Note both DDR and RFMiD (as mirrored on Kaggle) are *classification* sets with
no lesion masks — they are used for encoder pretraining (:mod:`pretrain_rfmid`),
not here.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import fov_bbox, make_rng, random_patch
from idrid_dataset import DEFAULT_LESIONS, IDRiDSegDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

EOPHTHA_SLUG = "diveshthakker/eoptha-diabetic-retinopathy"
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _index_of(fname: str) -> Optional[str]:
    m = re.search(r"(\d+)", os.path.splitext(os.path.basename(fname))[0])
    return m.group(1) if m else None


class EOphthaEXDataset(Dataset):
    """e-ophtha exudate subset: images in ``EX/``, masks in ``Annotation_EX/``.

    Files pair by numeric index (``EX12.jpg`` <-> ``EX_GT12.png``), not by stem.
    Only the EX channel is annotated; all other lesion channels are marked
    invalid so the loss ignores them.
    """

    def __init__(self, root: str, lesions: Sequence[str] = DEFAULT_LESIONS,
                 fov_crop: bool = True) -> None:
        self.img_dir = os.path.join(root, "EX")
        self.msk_dir = os.path.join(root, "Annotation_EX")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"e-ophtha 'EX' dir not found under {root!r}")
        self.lesions = list(lesions)
        self.fov_crop = fov_crop

        imgs = {_index_of(f): f for f in os.listdir(self.img_dir)
                if f.lower().endswith(_IMG_EXTS)}
        msks = {_index_of(f): f for f in os.listdir(self.msk_dir)
                if f.lower().endswith(_IMG_EXTS)}
        keys = sorted(set(imgs) & set(msks) - {None}, key=lambda k: int(k))
        self.pairs: List[Tuple[str, str]] = [(imgs[k], msks[k]) for k in keys]

        # Only EX is annotated here.
        self.valid = torch.tensor(
            [1.0 if c == "EX" else 0.0 for c in self.lesions], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def load_full(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        imf, mkf = self.pairs[idx]
        img = np.asarray(Image.open(os.path.join(self.img_dir, imf)).convert("RGB"))
        ex = np.asarray(Image.open(os.path.join(self.msk_dir, mkf)).convert("L"))
        H, W = img.shape[:2]
        masks = np.zeros((len(self.lesions), H, W), dtype=np.uint8)
        if "EX" in self.lesions:
            masks[self.lesions.index("EX")] = (ex > 0).astype(np.uint8)
        if self.fov_crop:
            r0, r1, c0, c1 = fov_bbox(img)
            img = img[r0:r1, c0:c1]
            masks = masks[:, r0:r1, c0:c1]
        return img, masks


class MultiLesionSegDataset(Dataset):
    """Concatenate lesion-segmentation sources, yielding (image, masks, valid).

    Parameters
    ----------
    idrid_root / eophtha_root / fgadr_root : dataset roots; None skips a source.
                FGADR auto-detects under data/ when passed the string "auto".
    split : 'train' | 'test'  (e-ophtha has no official split -> train only).
    image_size / patch_size / fg_bias / augment : as in :class:`IDRiDSegDataset`.
    """

    def __init__(
        self,
        idrid_root: Optional[str] = None,
        eophtha_root: Optional[str] = None,
        fgadr_root: Optional[str] = None,
        split: str = "train",
        image_size: int = 512,
        lesions: Sequence[str] = DEFAULT_LESIONS,
        patch_size: Optional[int] = None,
        fg_bias: float = 0.7,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.lesions = list(lesions)
        self.image_size = image_size
        self.patch_size = patch_size
        self.fg_bias = fg_bias
        self.augment = augment
        self.seed = seed

        self.sources: List[Tuple[str, object, torch.Tensor]] = []
        if idrid_root:
            ds = IDRiDSegDataset(idrid_root, split=split, image_size=image_size,
                                 lesions=self.lesions, fov_crop=True,
                                 patch_size=None, augment=False, seed=seed)
            self.sources.append(("idrid", ds, torch.ones(len(self.lesions))))
        if eophtha_root and split == "train":       # e-ophtha: train-only source
            ds = EOphthaEXDataset(eophtha_root, lesions=self.lesions)
            self.sources.append(("eophtha", ds, ds.valid))
        if fgadr_root:
            from fgadr_dataset import FGADRSegDataset
            ds = FGADRSegDataset(
                None if fgadr_root == "auto" else fgadr_root,
                split=split, image_size=image_size, lesions=self.lesions,
                fov_crop=True, patch_size=None, augment=False, seed=seed)
            self.sources.append(("fgadr", ds, ds.valid))

        self.index: List[Tuple[int, int]] = []      # (source_i, local_i)
        for si, (_, ds, _) in enumerate(self.sources):
            self.index.extend((si, i) for i in range(len(ds)))

    @property
    def num_classes(self) -> int:
        return len(self.lesions)

    def source_counts(self) -> dict:
        return {name: len(ds) for name, ds, _ in self.sources}

    def __len__(self) -> int:
        return len(self.index)

    def _to_tensors(self, img_np, masks):
        x = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.from_numpy(masks.astype(np.float32))

    def __getitem__(self, i: int):
        si, li = self.index[i]
        _, ds, valid = self.sources[si]
        rng = make_rng(self.seed, i)
        img_np, masks = ds.load_full(li)

        if self.patch_size:
            img_np, masks = random_patch(img_np, masks, self.patch_size, rng, self.fg_bias)
        else:
            size = (self.image_size, self.image_size)
            img_np = np.asarray(Image.fromarray(img_np).resize(size, Image.BILINEAR))
            masks = np.stack([np.asarray(Image.fromarray(masks[c]).resize(size, Image.NEAREST))
                              for c in range(masks.shape[0])], axis=0)

        if self.augment and bool(rng.integers(0, 2)):
            img_np = img_np[:, ::-1, :].copy()
            masks = masks[:, :, ::-1].copy()

        x, m = self._to_tensors(img_np, masks)
        return x, m, valid.clone()
