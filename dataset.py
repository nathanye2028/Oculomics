"""
dataset.py
==========
A lightweight, artifact-resilient PyTorch ``Dataset`` for the **mBRSET**
(mobile Brazilian Retinal Image Dataset) of smartphone-captured fundus photographs.

Design goals (per project spec):
  * Strictly modular & production-ready.
  * Memory efficient  -> labels are pre-vectorised once in ``__init__``; the
    pandas ``DataFrame`` is *not* retained; images are decoded lazily, with an
    optional bounded LRU cache and JPEG draft-mode fast-decoding.
  * Artifact resilient -> augmentation pipeline simulates the acquisition
    artifacts typical of *mobile* fundus capture (uneven illumination, blur,
    compression, occlusion), and the dataset can expose per-image quality /
    artifact flags for sample weighting or curriculum filtering.

The label schema follows ``dataframe_brsetmobile.csv``:
    file, laterality, sex, age, educational_level, insurance, ...,
    final_icdr (0-4, NaN for ungradable), final_edema (yes/no),
    final_quality (yes/no), final_artifacts (yes/no).

Typical use
-----------
    from dataset import MBRSETDataset, build_transforms

    train_ds = MBRSETDataset(
        csv="dataframe_brsetmobile.csv",
        images_dir="data/images",
        task="dr_referable",          # referable DR  (ICDR >= 2)
        split="train",                # -> training augmentations
        image_size=384,
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    batch = next(iter(loader))
    batch["image"]   # FloatTensor [B, 3, H, W]
    batch["label"]   # LongTensor  [B]   (or FloatTensor for multilabel/regression)
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

try:  # torchvision v2 transforms are faster; fall back to v1 cleanly.
    from torchvision.transforms import v2 as T

    _V2 = True
except ImportError:  # pragma: no cover
    import torchvision.transforms as T

    _V2 = False

# Truncated JPEGs are common in mobile captures; tolerate rather than crash.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- #
# Normalisation constants
# ----------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Channel statistics computed on mBRSET fundus crops (see src/data_loader.py).
# Prefer these over ImageNet stats when training from scratch.
MBRSET_MEAN = (0.5896, 0.2989, 0.1108)
MBRSET_STD = (0.2854, 0.1591, 0.0701)


# ----------------------------------------------------------------------------- #
# Label specifications
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelSpec:
    """Declarative description of how to derive a target from a CSV row.

    Attributes
    ----------
    source_cols : columns required from the CSV.
    fn          : maps a row-slice (dict of column -> value) to a python scalar
                  or 1-D sequence. Return ``np.nan`` to mark the row unusable.
    dtype       : torch dtype of the emitted label tensor.
    num_classes : number of classes (None for regression / multilabel vector).
    multilabel  : True if ``fn`` returns a vector to be used with BCE.
    """

    source_cols: Tuple[str, ...]
    fn: Callable[[Dict[str, object]], Union[float, int, Sequence[float]]]
    dtype: torch.dtype
    num_classes: Optional[int] = None
    multilabel: bool = False
    description: str = ""


def _yes_no(value, positive: str = "yes") -> float:
    """Map a yes/no (or 1/0) cell to 1.0 / 0.0, NaN-safe."""
    if isinstance(value, str):
        return 1.0 if value.strip().lower() == positive else 0.0
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float(value)


def _icdr_grade(row: Dict[str, object]) -> float:
    v = row["final_icdr"]
    return np.nan if pd.isna(v) else float(int(v))


# Registry of ready-made tasks. ``label_col``/``label_map`` kwargs on the
# dataset override this for ad-hoc targets.
LABEL_REGISTRY: Dict[str, LabelSpec] = {
    # Diabetic retinopathy --------------------------------------------------- #
    "dr_grade": LabelSpec(
        source_cols=("final_icdr",),
        fn=_icdr_grade,
        dtype=torch.long,
        num_classes=5,
        description="ICDR severity grade 0-4 (multiclass).",
    ),
    "dr_binary": LabelSpec(  # any DR
        source_cols=("final_icdr",),
        fn=lambda r: np.nan if pd.isna(r["final_icdr"]) else float(int(r["final_icdr"]) >= 1),
        dtype=torch.long,
        num_classes=2,
        description="Any DR present (ICDR >= 1).",
    ),
    "dr_referable": LabelSpec(  # clinically referable DR
        source_cols=("final_icdr",),
        fn=lambda r: np.nan if pd.isna(r["final_icdr"]) else float(int(r["final_icdr"]) >= 2),
        dtype=torch.long,
        num_classes=2,
        description="Referable DR (ICDR >= 2: moderate NPDR or worse).",
    ),
    # Macular edema ---------------------------------------------------------- #
    "edema": LabelSpec(
        source_cols=("final_edema",),
        fn=lambda r: _yes_no(r["final_edema"]),
        dtype=torch.long,
        num_classes=2,
        description="Diabetic macular edema present (yes/no).",
    ),
    # Image-quality control heads ------------------------------------------- #
    "quality": LabelSpec(
        source_cols=("final_quality",),
        fn=lambda r: _yes_no(r["final_quality"]),
        dtype=torch.long,
        num_classes=2,
        description="Gradable quality (yes/no).",
    ),
    "artifacts": LabelSpec(
        source_cols=("final_artifacts",),
        fn=lambda r: _yes_no(r["final_artifacts"]),
        dtype=torch.long,
        num_classes=2,
        description="Acquisition artifacts present (yes/no).",
    ),
    # Demographics ----------------------------------------------------------- #
    "sex": LabelSpec(
        source_cols=("sex",),
        fn=lambda r: np.nan if pd.isna(r["sex"]) else float(int(r["sex"])),
        dtype=torch.long,
        num_classes=2,
        description="Patient sex as encoded in the CSV (0/1).",
    ),
    "age": LabelSpec(
        source_cols=("age",),
        fn=lambda r: np.nan if pd.isna(r["age"]) else float(r["age"]),
        dtype=torch.float32,
        num_classes=None,
        description="Patient age in years (regression).",
    ),
}


# ----------------------------------------------------------------------------- #
# Transform factory
# ----------------------------------------------------------------------------- #
def build_transforms(
    split: str = "train",
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    artifact_resilient: bool = True,
) -> Callable:
    """Construct a torchvision transform pipeline.

    ``split='train'`` returns an augmentation chain that, when
    ``artifact_resilient`` is set, injects perturbations mimicking real mobile
    fundus acquisition faults so the model learns to ignore them:
        * geometric: random resized crop, flips, mild rotation;
        * photometric: brightness/contrast/saturation jitter (uneven flash);
        * optical: gaussian blur (defocus / motion);
        * occlusion: random erasing (dust, eyelash, reflections).
    ``split`` in {'val', 'test'} returns a deterministic resize + center crop.

    The pipeline always ends in a normalised ``float32`` CHW tensor.

    Note on ordering: the PIL image is converted to a tensor *first*, then every
    augmentation runs on the tensor. This avoids torchvision's PIL ``adjust_hue``
    path (which overflows under NumPy >= 2) and uses the faster tensor kernels.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    is_train = split == "train"

    # Convert PIL -> float CHW tensor in [0, 1] up front, then augment tensors.
    if _V2:
        to_tensor: List[Callable] = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    else:  # torchvision v1
        to_tensor = [T.ToTensor()]

    normalize = T.Normalize(mean=list(mean), std=list(std))

    if is_train:
        pipeline: List[Callable] = list(to_tensor)
        pipeline += [
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
        if artifact_resilient:
            pipeline += [
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            ]
        pipeline.append(normalize)
        if artifact_resilient:
            # Occlusion (dust / eyelash / reflections); operates on tensors.
            pipeline.append(T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)))
    else:
        resize_to = (int(round(size[0] * 1.14)), int(round(size[1] * 1.14)))
        pipeline = list(to_tensor) + [
            T.Resize(resize_to, antialias=True),
            T.CenterCrop(size),
            normalize,
        ]

    return T.Compose(pipeline)


# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
class MBRSETDataset(Dataset):
    """Memory-efficient mBRSET fundus dataset.

    Parameters
    ----------
    csv : str | pandas.DataFrame
        Path to ``dataframe_brsetmobile.csv`` (or a pre-filtered DataFrame,
        e.g. one split of a stratified train/val partition).
    images_dir : str
        Directory holding the fundus ``.jpg`` files named by the CSV ``file``
        column (e.g. ``1.1.jpg``).
    task : str, optional
        Key into :data:`LABEL_REGISTRY` (``dr_grade``, ``dr_referable``,
        ``dr_binary``, ``edema``, ``quality``, ``artifacts``, ``sex``,
        ``age``). Mutually exclusive with ``label_col``.
    label_col : str, optional
        Use a raw CSV column directly as the label, optionally remapped via
        ``label_map``. Overrides ``task``.
    label_map : dict, optional
        Value -> int/float mapping applied to ``label_col``.
    split : str
        ``'train'`` -> augmentations; ``'val'``/``'test'`` -> deterministic.
        Ignored when an explicit ``transform`` is supplied.
    image_size, mean, std, artifact_resilient
        Forwarded to :func:`build_transforms` when ``transform`` is None.
    transform : callable, optional
        Custom transform; overrides the built-in pipeline.
    file_col : str
        CSV column with image filenames (default ``'file'``).
    drop_missing_labels : bool
        Drop rows whose derived label is NaN (default True).
    drop_missing_files : bool
        Pre-scan ``images_dir`` and drop rows whose image is absent
        (default False; the scan touches the filesystem once).
    return_metadata : bool
        Include demographic / quality fields in each sample dict.
    cache_size : int
        Max decoded PIL images held in an in-process LRU cache
        (0 disables caching; caches are per-worker).
    draft_decode : bool
        Use Pillow JPEG draft mode to decode at a reduced scale near the
        target size — markedly faster and lower-memory for large captures.
    """

    METADATA_COLS = (
        "patient", "age", "sex", "laterality", "educational_level", "insurance",
        "dm_time", "insulin", "systemic_hypertension",
        "final_icdr", "final_edema", "final_quality", "final_artifacts",
    )

    def __init__(
        self,
        csv: Union[str, pd.DataFrame],
        images_dir: str,
        task: Optional[str] = "dr_referable",
        label_col: Optional[str] = None,
        label_map: Optional[Dict[object, Union[int, float]]] = None,
        split: str = "train",
        image_size: Union[int, Tuple[int, int]] = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        artifact_resilient: bool = True,
        transform: Optional[Callable] = None,
        file_col: str = "file",
        drop_missing_labels: bool = True,
        drop_missing_files: bool = False,
        return_metadata: bool = False,
        cache_size: int = 0,
        draft_decode: bool = True,
    ) -> None:
        super().__init__()

        if not os.path.isdir(images_dir):
            logger.warning("images_dir does not exist: %s", images_dir)
        self.images_dir = images_dir
        self.split = split
        self.file_col = file_col
        self.return_metadata = return_metadata
        self.draft_decode = draft_decode
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

        # --- load frame ----------------------------------------------------- #
        df = pd.read_csv(csv) if isinstance(csv, str) else csv.copy()
        if file_col not in df.columns:
            raise KeyError(f"file column {file_col!r} not in CSV columns: {list(df.columns)}")

        # --- resolve label spec -------------------------------------------- #
        # NB: keep ``spec`` local. Its ``fn`` is a lambda/closure that cannot be
        # pickled; retaining it on ``self`` would break DataLoader workers under
        # the ``spawn`` start method (macOS / Windows), where the dataset is
        # pickled into each worker. Labels are fully vectorised below, so the
        # spec's callable is not needed past ``__init__``.
        spec, label_values = self._resolve_labels(df, task, label_col, label_map)
        self.task = label_col or task
        self.num_classes = spec.num_classes
        self.multilabel = spec.multilabel
        self._label_dtype = spec.dtype

        files = df[file_col].astype(str).to_numpy()

        # --- filter rows ---------------------------------------------------- #
        keep = np.ones(len(df), dtype=bool)
        if drop_missing_labels:
            valid = ~_isnan_vector(label_values)
            keep &= valid
        if drop_missing_files:
            existing = {f for f in os.listdir(images_dir)} if os.path.isdir(images_dir) else set()
            keep &= np.array([f in existing for f in files], dtype=bool)

        n_drop = int((~keep).sum())
        if n_drop:
            logger.info("Dropping %d/%d rows (missing labels/files).", n_drop, len(df))

        self.files = files[keep]
        labels = label_values[keep]

        # --- vectorise labels once (no DataFrame retained) ------------------ #
        self.labels = torch.as_tensor(labels, dtype=self._label_dtype)

        # Optional metadata snapshot kept as plain dict-of-arrays (cheap).
        self._meta: Dict[str, np.ndarray] = {}
        if return_metadata:
            for c in self.METADATA_COLS:
                if c in df.columns:
                    self._meta[c] = df[c].to_numpy()[keep]

        # --- transforms ----------------------------------------------------- #
        self.transform = transform or build_transforms(
            split=split, image_size=image_size, mean=mean, std=std,
            artifact_resilient=artifact_resilient,
        )

        # --- per-worker LRU image cache ------------------------------------- #
        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[int, Image.Image]" = OrderedDict()

        if len(self.files) == 0:
            logger.warning("MBRSETDataset initialised with 0 usable samples.")

    # ------------------------------------------------------------------ #
    # Label resolution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_labels(
        df: pd.DataFrame,
        task: Optional[str],
        label_col: Optional[str],
        label_map: Optional[Dict[object, Union[int, float]]],
    ) -> Tuple[LabelSpec, np.ndarray]:
        if label_col is not None:
            if label_col not in df.columns:
                raise KeyError(f"label_col {label_col!r} not in CSV.")
            raw = df[label_col]
            if label_map is not None:
                mapped = raw.map(label_map)
                is_float = any(isinstance(v, float) for v in label_map.values())
                dtype = torch.float32 if is_float else torch.long
                n_cls = None if is_float else len(set(label_map.values()))
                spec = LabelSpec((label_col,), lambda r: r[label_col], dtype, n_cls,
                                 description=f"custom map on {label_col}")
                return spec, mapped.to_numpy(dtype=np.float64)
            # No map: assume already numeric.
            arr = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
            uniq = np.unique(arr[~np.isnan(arr)])
            is_int = np.allclose(uniq, uniq.astype(int))
            spec = LabelSpec((label_col,), lambda r: r[label_col],
                             torch.long if is_int else torch.float32,
                             int(uniq.max()) + 1 if is_int else None,
                             description=f"raw {label_col}")
            return spec, arr

        if task is None:
            raise ValueError("Provide either `task` or `label_col`.")
        if task not in LABEL_REGISTRY:
            raise KeyError(f"Unknown task {task!r}. Available: {sorted(LABEL_REGISTRY)}")
        spec = LABEL_REGISTRY[task]
        missing = [c for c in spec.source_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Task {task!r} needs columns {missing} not present in CSV.")

        sub = df[list(spec.source_cols)]
        # Row-wise application; vectorised enough for label derivation (cheap).
        values = np.array(
            [spec.fn(dict(zip(spec.source_cols, row))) for row in sub.to_numpy()],
            dtype=np.float64,
        )
        return spec, values

    # ------------------------------------------------------------------ #
    # Image IO
    # ------------------------------------------------------------------ #
    def _load_image(self, idx: int) -> Image.Image:
        if self._cache_size and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        path = os.path.join(self.images_dir, self.files[idx])
        with Image.open(path) as img:
            if self.draft_decode:
                # Decode JPEG at the smallest scale >= target -> faster/cheaper.
                img.draft("RGB", self.image_size)
            img = img.convert("RGB")

        if self._cache_size:
            self._cache[idx] = img
            self._cache.move_to_end(idx)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return img

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image = self.transform(self._load_image(idx))
        sample: Dict[str, object] = {
            "image": image,
            "label": self.labels[idx],
            "file": self.files[idx],
        }
        if self.return_metadata and self._meta:
            sample["metadata"] = {k: v[idx] for k, v in self._meta.items()}
        return sample

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def class_counts(self) -> Optional[torch.Tensor]:
        """Per-class sample counts (None for regression)."""
        if self.num_classes is None or self.multilabel:
            return None
        return torch.bincount(self.labels.long(), minlength=self.num_classes)

    def class_weights(self) -> Optional[torch.Tensor]:
        """Inverse-frequency class weights for imbalanced CE/Focal loss."""
        counts = self.class_counts()
        if counts is None:
            return None
        counts = counts.clamp(min=1)
        w = counts.sum() / (len(counts) * counts.float())
        return w

    def sample_weights(self) -> Optional[torch.Tensor]:
        """Per-sample weights for ``WeightedRandomSampler`` (balanced sampling)."""
        cw = self.class_weights()
        if cw is None:
            return None
        return cw[self.labels.long()]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MBRSETDataset(task={self.task!r}, n={len(self)}, "
            f"num_classes={self.num_classes}, split={self.split!r}, "
            f"image_size={self.image_size})"
        )


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #
def _isnan_vector(arr: np.ndarray) -> np.ndarray:
    """NaN mask that tolerates object/non-float arrays."""
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    return pd.isna(arr)


def stratified_split(
    csv: Union[str, pd.DataFrame],
    task: str = "dr_referable",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    group_col: Optional[str] = "patient",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Patient-grouped, label-stratified train/val/test split.

    Splitting by ``group_col='patient'`` prevents leakage from a patient's
    multiple images straddling splits — essential for honest evaluation.
    Returns a dict of DataFrames keyed ``'train' | 'val' | 'test'`` ready to
    pass to :class:`MBRSETDataset`.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    df = pd.read_csv(csv) if isinstance(csv, str) else csv.copy()
    spec = LABEL_REGISTRY[task]
    y = np.array(
        [spec.fn(dict(zip(spec.source_cols, row)))
         for row in df[list(spec.source_cols)].to_numpy()],
        dtype=np.float64,
    )
    df = df.loc[~_isnan_vector(y)].reset_index(drop=True)
    y = y[~np.isnan(y)]

    groups = df[group_col].to_numpy() if group_col and group_col in df.columns else np.arange(len(df))

    if group_col and group_col in df.columns:
        # Hold out test by patient, then val by patient from the remainder.
        n_splits = max(2, int(round(1.0 / max(test_frac, 1e-6))))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(sgkf.split(df, y, groups))
        df_tv, y_tv, g_tv = df.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]

        rel_val = val_frac / (1.0 - test_frac)
        n_splits_v = max(2, int(round(1.0 / max(rel_val, 1e-6))))
        sgkf_v = StratifiedGroupKFold(n_splits=n_splits_v, shuffle=True, random_state=seed)
        tr_idx, val_idx = next(sgkf_v.split(df_tv, y_tv, g_tv))
        return {
            "train": df_tv.iloc[tr_idx].reset_index(drop=True),
            "val": df_tv.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True),
        }

    # No grouping -> plain stratified splits.
    df_tv, df_test = train_test_split(df, test_size=test_frac, stratify=y, random_state=seed)
    y_tv = y[df_tv.index.to_numpy()] if False else None  # noqa: keep simple below
    df_train, df_val = train_test_split(
        df_tv, test_size=val_frac / (1.0 - test_frac),
        stratify=df_tv[list(spec.source_cols)[0]], random_state=seed,
    )
    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }


# ----------------------------------------------------------------------------- #
# Smoke test
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="mBRSET dataset smoke test.")
    p.add_argument("--csv", default="dataframe_brsetmobile.csv")
    p.add_argument("--images", default="data/images")
    p.add_argument("--task", default="dr_referable")
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    ds = MBRSETDataset(
        csv=args.csv, images_dir=args.images, task=args.task,
        split="train", image_size=args.image_size,
        return_metadata=True, drop_missing_files=False,
    )
    print(ds)
    print("class counts :", ds.class_counts())
    print("class weights:", ds.class_weights())
    if len(ds):
        try:
            s = ds[0]
            print("sample image :", s["image"].shape, s["image"].dtype)
            print("sample label :", s["label"], "file:", s["file"])
        except FileNotFoundError as e:
            print("(labels OK; image file not present in this checkout)", e)
