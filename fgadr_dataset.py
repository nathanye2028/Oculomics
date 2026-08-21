"""
fgadr_dataset.py
================
FGADR **Seg-set** lesion-segmentation source — 1,842 fundus images at
1280x1280 with pixel-level masks for MA / HE / EX / SE.

Why this matters
----------------
Before FGADR the segmentation sources were IDRiD (54 train / 27 test, all four
lesions) and e-ophtha (47 images, EX only). FGADR adds **1,842 fully-annotated
images** — roughly an 18x increase in four-lesion supervision — and it is the
first source large enough to train on rather than merely fine-tune against.

Interface follows :class:`idrid_dataset.IDRiDSegDataset` exactly, so it drops
into :class:`multi_seg_dataset.MultiLesionSegDataset` as a third source:
``load_full(idx) -> (image[H,W,3] uint8, masks[C,H,W] uint8)`` at native
resolution, plus a ``valid`` [C] vector.

Annotation validity
-------------------
All four default lesions (MA, HE, EX, SE) are annotated for **every** image, so
``valid`` is all-ones — unlike e-ophtha, FGADR needs no masking for the default
lesion set.

The release also ships IRMA (159 masks) and NV (49 masks). These are **not** in
``DEFAULT_LESIONS`` and should not be enabled casually: a missing IRMA file
cannot be distinguished from "annotator saw no IRMA", so loading them for all
1,842 images would teach ~1,700 false negatives — precisely the error that
``valid`` exists to prevent. :meth:`valid_for` returns correct per-image
validity if you do enable them, but note that ``MultiLesionSegDataset`` uses a
single per-source ``valid`` vector and would need extending to consume it.

Measured properties (all 7,576 masks in the release)
----------------------------------------------------
    lesion   masks   w/ lesion   median instance   median area*
    MA        1842        1424           84 px          0.063%
    HE        1842        1456          166 px          0.601%
    EX        1842        1279           72 px          0.350%
    SE        1842         627         1506 px          0.259%
    *fraction of image pixels, among images containing that lesion

Two release quirks handled here
-------------------------------
1. **Mixed mask modes.** ~2/3 of masks are RGB, ~1/3 single-channel L, with
   stray values (1..17) alongside 0/255 from compression. We take channel 0 and
   threshold at >127.
2. **The filename suffix is not the grade.** ``0003_3.png`` splits 1000/254/588
   across _1/_2/_3, whereas real DR grades in ``DR_Seg_Grading_Label.csv`` are
   0-4 splitting 101/212/595/647/287. Splits stratify on the CSV.

On resolution: FGADR annotations are robust to downscaling — measured instance
survival at 512 is >99% for every lesion, and even naive NEAREST at 256 keeps
93-98%. The binding constraint is not annotation loss but *learnability*: a
microaneurysm reduced to ~3 px at 256 (vs 14 px at 512, 33 px at 768) gives the
network little to key on. Prefer ``patch_size`` (native-resolution crops via
:func:`fundus_utils.random_patch`) over a global downscale.

Setup:
    unzip -q data/FGADR-Seg-set_Release.zip 'Seg-set/*' -d data/fgadr

Run:
    python fgadr_dataset.py                # stats + one-sample self-check
    python fgadr_dataset.py --retention    # what downscaling costs, per lesion
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_utils import dihedral_jitter, fov_bbox, make_rng, random_patch
from idrid_dataset import DEFAULT_LESIONS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Lesion code -> mask directory. Note the release's "Hemohedge" spelling.
LESION_FOLDERS: Dict[str, str] = {
    "MA": "Microaneurysms_Masks",
    "HE": "Hemohedge_Masks",
    "EX": "HardExudate_Masks",
    "SE": "SoftExudate_Masks",
    "IRMA": "IRMA_Masks",
    "NV": "Neovascularization_Masks",
}
# Annotated for all 1,842 images; the rest are partial (see module docstring).
FULL_COVERAGE = ("MA", "HE", "EX", "SE")
GRADING_CSV = "DR_Seg_Grading_Label.csv"


def resolve_seg_root(root: Optional[str] = None) -> str:
    """Locate the ``Seg-set`` dir, tolerating a couple of nesting variants."""
    candidates = [root] if root else ["data/fgadr/Seg-set", "data/Seg-set", "Seg-set"]
    for c in candidates:
        if not c:
            continue
        if os.path.isdir(os.path.join(c, "Original_Images")):
            return c
        nested = os.path.join(c, "Seg-set")
        if os.path.isdir(os.path.join(nested, "Original_Images")):
            return nested
    raise FileNotFoundError(
        "Could not find FGADR Seg-set (expected data/fgadr/Seg-set/Original_Images/). "
        "Extract with:\n"
        "  unzip -q data/FGADR-Seg-set_Release.zip 'Seg-set/*' -d data/fgadr"
    )


def read_grades(seg_root: str) -> Dict[str, int]:
    """Map ``filename -> DR grade (0-4)`` from the release CSV (headerless)."""
    path = os.path.join(seg_root, GRADING_CSV)
    grades: Dict[str, int] = {}
    if not os.path.isfile(path):
        return grades
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                try:
                    grades[row[0].strip()] = int(row[1])
                except ValueError:
                    continue                       # tolerate a stray header
    return grades


def stratified_split(
    files: Sequence[str],
    grades: Dict[str, int],
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split filenames into (train, val, test), stratified on DR grade.

    FGADR ships no official split. Stratifying matters because grades are far
    from uniform (grade 0 is only ~5% of the set), so an unstratified draw can
    leave a held-out set with almost no grade-0 or grade-4 cases and make
    metrics jump between seeds for reasons unrelated to the model.

    A dedicated **val** split exists so model selection has a matched-distribution
    signal. Selecting the best checkpoint on a small out-of-distribution set (e.g.
    IDRiD's 27 images) is noisy enough to save a worse checkpoint than a better one.

    Splits are disjoint and deterministic given ``seed``: test is drawn first,
    then val from what remains, so changing ``val_frac`` never moves an image
    into or out of test.
    """
    rng = np.random.default_rng(seed)
    by_grade: Dict[int, List[str]] = {}
    for f in files:
        by_grade.setdefault(grades.get(f, -1), []).append(f)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    for g in sorted(by_grade):
        group = sorted(by_grade[g])
        idx = rng.permutation(len(group))
        n_test = int(round(len(group) * test_frac))
        n_val = int(round(len(group) * val_frac))
        test.extend(group[i] for i in idx[:n_test])
        val.extend(group[i] for i in idx[n_test:n_test + n_val])
        train.extend(group[i] for i in idx[n_test + n_val:])
    return sorted(train), sorted(val), sorted(test)


def _load_mask(path: str) -> np.ndarray:
    """Read a mask as uint8 {0,1}, handling RGB / L modes and stray values."""
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 127).astype(np.uint8)


class FGADRSegDataset(Dataset):
    """FGADR Seg-set, shaped like :class:`idrid_dataset.IDRiDSegDataset`.

    Parameters
    ----------
    root       : path to ``Seg-set`` (auto-detected under data/ if None).
    split      : 'train' | 'val' | 'test' | 'all'. Grade-stratified and disjoint;
                 FGADR ships no official split. Use 'val' for model selection.
    image_size : square resolution in whole-image mode (patch_size=None).
    lesions    : ordered lesion codes -> output channels.
    fov_crop   : crop the black border to the retinal field of view first.
    patch_size : if set, return a native-resolution random crop of this size.
    fg_bias    : prob. a training patch is centred on a lesion pixel.
    augment    : light joint geometric augmentation (flips).
    test_frac  : held-out fraction for the stratified split.
    seed       : base seed for reproducible per-sample augmentation ONLY.
    split_seed : seed for the train/val/test partition. Deliberately separate
                 from ``seed`` and fixed by default: if the partition followed
                 the run seed, every ``--seeds 0 1 2`` sweep would score each
                 run on a different "test" set, and ``eval_fgadr.py`` (which
                 rebuilds the default partition) would evaluate non-default-seed
                 checkpoints on images they were trained on.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        image_size: int = 512,
        lesions: Sequence[str] = DEFAULT_LESIONS,
        fov_crop: bool = True,
        patch_size: Optional[int] = None,
        fg_bias: float = 0.7,
        augment: bool = False,
        val_frac: float = 0.1,
        test_frac: float = 0.2,
        seed: int = 42,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        for code in lesions:
            if code not in LESION_FOLDERS:
                raise KeyError(f"Unknown lesion code {code!r}; choices: {list(LESION_FOLDERS)}")
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be 'train', 'val', 'test' or 'all', got {split!r}")

        self.root = resolve_seg_root(root)
        self.split = split
        self.image_size = image_size
        self.lesions = list(lesions)
        self.fov_crop = fov_crop
        self.patch_size = patch_size
        self.fg_bias = fg_bias
        self.augment = augment
        self.seed = seed

        self.img_dir = os.path.join(self.root, "Original_Images")
        all_files = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(".png"))
        if not all_files:
            raise FileNotFoundError(f"no images in {self.img_dir}")

        self.grades = read_grades(self.root)
        if not self.grades:
            # Every image would land in one stratum and the split becomes a
            # plain random draw. Recoverable, but it must not pass silently.
            print(f"[warn] {GRADING_CSV} not found under {self.root}; split will NOT be "
                  "grade-stratified. Extract it with:\n"
                  f"       unzip -q -o data/FGADR-Seg-set_Release.zip 'Seg-set/{GRADING_CSV}'"
                  " -d data/fgadr")
        # NB: split_seed, not seed — the partition must be identical across runs
        # (see the parameter docstring; a run-seed-coupled split leaks test data
        # into eval_fgadr.py and makes cross-seed "test" numbers incomparable).
        train, val, test = stratified_split(all_files, self.grades, val_frac, test_frac, split_seed)
        self.files: List[str] = {
            "train": train, "val": val, "test": test, "all": all_files,
        }[split]

        # All four default lesions are annotated for every image.
        self.valid = torch.tensor(
            [1.0 if c in FULL_COVERAGE else 0.0 for c in self.lesions], dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        return len(self.lesions)

    def __len__(self) -> int:
        return len(self.files)

    def grade_of(self, fname: str) -> int:
        return self.grades.get(fname, -1)

    def _mask_path(self, code: str, fname: str) -> Optional[str]:
        p = os.path.join(self.root, LESION_FOLDERS[code], fname)
        return p if os.path.isfile(p) else None

    def valid_for(self, idx: int) -> torch.Tensor:
        """Per-image annotation validity [C].

        Identical to ``self.valid`` for the four full-coverage lesions. Only
        differs when IRMA/NV are enabled, where a mask file exists for a small
        subset and absence means *unannotated*, not *absent*.
        """
        fname = self.files[idx]
        return torch.tensor(
            [1.0 if (c in FULL_COVERAGE or self._mask_path(c, fname)) else 0.0
             for c in self.lesions], dtype=torch.float32)

    def load_full(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load FOV-cropped native-resolution (image[H,W,3], masks[C,H,W])."""
        fname = self.files[idx]
        img = np.asarray(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        H, W = img.shape[:2]
        masks = np.zeros((len(self.lesions), H, W), dtype=np.uint8)
        for c, code in enumerate(self.lesions):
            p = self._mask_path(code, fname)
            if p is not None:
                masks[c] = _load_mask(p)
        if self.fov_crop:
            r0, r1, c0, c1 = fov_bbox(img)
            img = img[r0:r1, c0:c1]
            masks = masks[:, r0:r1, c0:c1]
        return img, masks

    def _to_tensors(self, img_np: np.ndarray, masks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        return img_t, torch.from_numpy(masks.astype(np.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # Shared dihedral+jitter (fundus_utils): the same augmentation as
            # IDRiD, so no source in a mixed training set is under-augmented.
            img_np, masks = dihedral_jitter(img_np, masks, rng)

        x, m = self._to_tensors(img_np, masks)
        return x, m, self.valid_for(idx)


# --------------------------------------------------------------------------- #
# Diagnostic: what does downscaling actually cost?
# --------------------------------------------------------------------------- #
def lesion_retention(
    root: Optional[str] = None,
    lesions: Sequence[str] = FULL_COVERAGE,
    sizes: Sequence[int] = (256, 512, 768, 1024),
    n: int = 60,
    seed: int = 0,
) -> None:
    """Report three quantities per lesion and target size.

    * **area** — lesion pixels relative to native. NEAREST scores ~100% here
      *by construction* (point sampling is unbiased in expectation), so area
      alone says nothing about whether small lesions survived. Judging
      resolution on this number alone is how you reach the wrong conclusion.
    * **inst** — fraction of connected lesion components retaining at least one
      positive pixel after a downsample/upsample round trip. This is what
      answers "did we delete lesions?".
    * **med px/inst** — median surviving size of a single lesion, i.e. how much
      the network has left to learn from. This is the real constraint.

    Needs scipy for connected components (already a scikit-learn dependency).
    """
    from scipy import ndimage

    seg_root = resolve_seg_root(root)
    rng = np.random.default_rng(seed)

    for code in lesions:
        d = os.path.join(seg_root, LESION_FOLDERS[code])
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        rng.shuffle(files)
        masks: List[np.ndarray] = []
        for f in files:
            m = _load_mask(os.path.join(d, f)).astype(bool)
            if m.any():
                masks.append(m)
            if len(masks) >= n:
                break
        if not masks:
            continue

        native_px: List[int] = []
        for m in masks:
            lab, _ = ndimage.label(m)
            native_px += list(np.bincount(lab.ravel())[1:])
        print(f"\n{code}  ({len(masks)} lesion-bearing masks, native 1280, "
              f"median instance {np.median(native_px):.0f} px)")
        print(f"  {'size':>6}  {'area':>7}  {'inst':>7}  {'med px/inst':>12}")

        for s in sizes:
            area: List[float] = []
            surv_px: List[int] = []
            kept = total = 0
            for m in masks:
                lab, k = ndimage.label(m)
                if k == 0:
                    continue
                total += k
                small = np.asarray(
                    Image.fromarray(m.astype(np.uint8) * 255).resize((s, s), Image.NEAREST)
                ) > 127
                area.append(small.sum() * (m.shape[0] / s) ** 2 / m.sum())
                up = np.asarray(
                    Image.fromarray(small.astype(np.uint8) * 255)
                    .resize(m.shape[::-1], Image.NEAREST)
                ) > 127
                kept += len(np.unique(lab[up & (lab > 0)]))
                sl, sk = ndimage.label(small)
                if sk:
                    surv_px += list(np.bincount(sl.ravel())[1:])
            print(f"  {s:>6}  {np.mean(area)*100:>6.1f}%  {kept/total*100:>6.1f}%"
                  f"  {np.median(surv_px):>12.0f}")


def main() -> int:
    p = argparse.ArgumentParser(description="FGADR Seg-set stats / self-check.")
    p.add_argument("--root", default=None)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--retention", action="store_true",
                   help="Measure what downscaling costs instead of running the self-check.")
    args = p.parse_args()

    if args.retention:
        lesion_retention(args.root)
        return 0

    tr = FGADRSegDataset(args.root, split="train", image_size=args.image_size, augment=True)
    va = FGADRSegDataset(args.root, split="val", image_size=args.image_size)
    te = FGADRSegDataset(args.root, split="test", image_size=args.image_size)
    print(f"[info] root    : {tr.root}")
    print(f"[info] lesions : {tr.lesions}   valid={tr.valid.tolist()}")
    print(f"[info] split   : {len(tr)} train / {len(va)} val / {len(te)} test")
    overlap = (set(tr.files) & set(va.files)) | (set(tr.files) & set(te.files)) \
        | (set(va.files) & set(te.files))
    print(f"[info] disjoint: {'YES' if not overlap else f'NO — {len(overlap)} leaked!'}")

    from collections import Counter
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        c = Counter(ds.grade_of(f) for f in ds.files)
        print(f"[info] {name:5} grades: " + "  ".join(f"{g}:{c[g]}" for g in sorted(c)))

    img, msk, valid = tr[0]
    print(f"\n=== one sample ({tr.files[0]}, grade {tr.grade_of(tr.files[0])}) ===")
    print(f"image : shape={tuple(img.shape)} range=[{img.min():.2f}, {img.max():.2f}]")
    print(f"mask  : shape={tuple(msk.shape)}   valid={valid.tolist()}")
    for i, c in enumerate(tr.lesions):
        print(f"  {c:<5} positive fraction: {msk[i].mean().item()*100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
