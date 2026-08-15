"""
brset_dataset.py
================
**BRSET (tabletop) -> the mBRSET schema**, so one loader serves both datasets.

BRSET is ~16k fundus images from a tabletop camera; mBRSET is ~5k from a
smartphone (Phelcom Eyer). Training on the first and testing on the second is
the domain-shift experiment this project's hypothesis is actually about:
does a model trained on clean, constrained captures survive unconstrained
mobile ones?

Why this is an adapter and not a second ``Dataset``
--------------------------------------------------
It would be easy to write ``BRSETDataset`` alongside :class:`dataset.MBRSETDataset`.
It would also quietly destroy the experiment. A domain-shift result is a claim
that *the images* differ; if the two datasets travel through two different
loaders, any measured gap is confounded by whatever the loaders do differently —
a different FOV crop, a different resize, a different normalisation. The number
would be uninterpretable and the confound invisible.

So this module does exactly one thing: rename and re-encode BRSET's label CSV
into the column names ``dataset.py`` already expects. The returned DataFrame
goes straight into ``MBRSETDataset(csv=df, ...)`` and ``stratified_split(...)``,
which means **both datasets are then decoded, FOV-cropped, resized and
normalised by the same code**. The only thing that differs is the pixels, which
is the entire point.

    from brset_dataset import load_brset
    from dataset import MBRSETDataset, stratified_split

    df = load_brset("<BRSET>/labels.csv")          # mBRSET-schema DataFrame
    splits = stratified_split(df, task="dr_referable", group_col="patient")
    train = MBRSETDataset(csv=splits["train"], images_dir="<BRSET>/fundus_photos",
                          task="dr_referable", split="train")

Schema mapping
--------------
Taken from the published BRSET column spec, and **verified at load time against
the CSV you actually have** — a rename that silently produced an all-NaN label
column would show up as a model scoring at chance, so a missing column is a hard
error naming the columns that were present instead.

    BRSET                 -> mBRSET            note
    image_id              -> file              ``.jpg`` appended if absent
    patient_id            -> patient           the grouping key; no patient leakage
    DR_ICDR      (0-4)    -> final_icdr        same ICDR scale in both datasets
    macular_edema(0/1)    -> final_edema       dataset.py's _yes_no handles 0/1
    patient_age           -> age
    patient_sex  (1/2)    -> sex     (0/1)     RE-ENCODED, see below
    exam_eye     (1/2)    -> laterality
    quality               -> final_quality     optional; absent in some releases
    artifacts             -> final_artifacts   optional

``patient_sex`` is re-encoded because BRSET uses 1/2 while ``LABEL_REGISTRY["sex"]``
declares ``num_classes=2`` and passes the raw value through. Feeding it a 2 would
index past the end of a 2-logit head and crash inside CrossEntropyLoss — or, with
a wider head, train silently against a class that never occurs.

Run this first, on the real CSV, before trusting any of the above:

    python brset_dataset.py --csv <BRSET>/labels.csv --inspect

It prints which mapped columns were found, how many rows survive per task, and
the label distribution next to mBRSET's, which is the thing that decides whether
a cross-dataset AUROC is even comparable.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# BRSET column -> mBRSET column. Split into columns we cannot proceed without and
# columns that are nice to have, because BRSET releases differ in the quality
# fields and none of the DR/edema tasks need them.
REQUIRED_MAP: Dict[str, str] = {
    "image_id": "file",
    "patient_id": "patient",
    "DR_ICDR": "final_icdr",
}
OPTIONAL_MAP: Dict[str, str] = {
    "macular_edema": "final_edema",
    "patient_age": "age",
    "patient_sex": "sex",
    "exam_eye": "laterality",
    "quality": "final_quality",
    "artifacts": "final_artifacts",
    "comorbidities": "comorbidities",
    "diabetes_time_y": "dm_time",
    "insuline": "insulin",
    "camera": "camera",
}

# Tasks in dataset.LABEL_REGISTRY and the mBRSET column each one needs, so
# --inspect can say which tasks are actually runnable on this CSV.
TASK_REQUIREMENTS: Dict[str, str] = {
    "dr_grade": "final_icdr",
    "dr_binary": "final_icdr",
    "dr_referable": "final_icdr",
    "edema": "final_edema",
    "quality": "final_quality",
    "artifacts": "final_artifacts",
    "sex": "sex",
    "age": "age",
}


def _normalise_sex(s: pd.Series) -> pd.Series:
    """BRSET encodes sex as 1/2; dataset.py's ``sex`` task expects 0/1.

    Left as-is this is not a crash you would enjoy debugging: a label of 2 against
    a 2-logit head raises deep inside CrossEntropyLoss with no mention of the CSV.
    Values already in {0,1} pass through, so re-running on adapted data is safe.
    """
    vals = set(pd.unique(s.dropna()))
    if vals and vals <= {1, 2, 1.0, 2.0}:
        return s - 1
    return s


def _ensure_ext(files: pd.Series, ext: str) -> pd.Series:
    """Append ``ext`` to filenames that carry no extension.

    BRSET's ``image_id`` is bare (``img00001``) in some releases and already
    suffixed in others; guessing wrong makes every file 'missing' at load.
    """
    if not ext:
        return files.astype(str)
    return files.astype(str).map(lambda f: f if os.path.splitext(f)[1] else f + ext)


def load_brset(
    csv: Union[str, pd.DataFrame],
    image_ext: str = ".jpg",
    strict: bool = True,
) -> pd.DataFrame:
    """Load BRSET's label CSV and return it in the mBRSET column schema.

    Parameters
    ----------
    csv : path to BRSET's ``labels.csv`` (or a DataFrame already read).
    image_ext : appended to ``file`` entries that lack an extension.
    strict : raise if a :data:`REQUIRED_MAP` column is missing. Set False to
        adapt whatever is present and let the caller decide — useful when
        inspecting an unfamiliar release.

    Returns
    -------
    A DataFrame with mBRSET column names, ready for
    :class:`dataset.MBRSETDataset` and :func:`dataset.stratified_split`.
    """
    df = pd.read_csv(csv) if isinstance(csv, str) else csv.copy()

    missing = [c for c in REQUIRED_MAP if c not in df.columns]
    if missing and strict:
        raise KeyError(
            f"BRSET CSV is missing required column(s) {missing}. "
            f"Columns present: {sorted(df.columns)}. "
            f"If this release names them differently, update REQUIRED_MAP in "
            f"brset_dataset.py rather than renaming the CSV — the mapping is "
            f"meant to be the auditable record of what was joined to what."
        )

    rename = {src: dst for src, dst in {**REQUIRED_MAP, **OPTIONAL_MAP}.items()
              if src in df.columns}
    out = df.rename(columns=rename).copy()

    # Keep only the mBRSET-schema columns; carrying BRSET's other 15 label
    # columns through would let a caller reach for e.g. 'drusens' and get a
    # column that has no mBRSET counterpart to test against.
    keep = [c for c in rename.values() if c in out.columns]
    out = out[keep]

    if "file" in out.columns:
        out["file"] = _ensure_ext(out["file"], image_ext)
    if "sex" in out.columns:
        out["sex"] = _normalise_sex(out["sex"])
    if "final_icdr" in out.columns:
        # dataset.py's _icdr_grade does int() on this; a stray string grade would
        # otherwise raise per-row deep inside label vectorisation.
        out["final_icdr"] = pd.to_numeric(out["final_icdr"], errors="coerce")

    out.attrs["source"] = "brset"
    return out


def load_any(root: str, dataset: str, image_ext: str = ".jpg") -> Dict[str, str]:
    """Resolve ``root`` to a (DataFrame, images_dir) pair for either dataset.

    Lets a trainer accept ``--dataset {mbrset,brset}`` without branching on the
    schema everywhere. Returns a dict so the caller can log what it resolved.
    """
    if dataset == "mbrset":
        csv_path = os.path.join(root, "labels_mbrset.csv")
        img_dir = os.path.join(root, "images")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"expected {csv_path}")
        return {"df": pd.read_csv(csv_path), "images_dir": img_dir,
                "csv": csv_path, "source": "mbrset"}
    if dataset == "brset":
        csv_path = os.path.join(root, "labels.csv")
        # BRSET's image directory is named differently across releases.
        candidates = ["fundus_photos", "images", "photos"]
        img_dir = next((os.path.join(root, c) for c in candidates
                        if os.path.isdir(os.path.join(root, c))), os.path.join(root, "fundus_photos"))
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"expected {csv_path}")
        return {"df": load_brset(csv_path, image_ext=image_ext), "images_dir": img_dir,
                "csv": csv_path, "source": "brset"}
    raise ValueError(f"unknown dataset {dataset!r}; expected 'mbrset' or 'brset'")


# --------------------------------------------------------------------------- #
# Inspector — run this on the real CSV before trusting the mapping above
# --------------------------------------------------------------------------- #
def _distribution(df: pd.DataFrame, task: str) -> Optional[Dict[int, int]]:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import LABEL_REGISTRY, _isnan_vector          # noqa: E402

    spec = LABEL_REGISTRY[task]
    if any(c not in df.columns for c in spec.source_cols):
        return None
    y = np.array([spec.fn(dict(zip(spec.source_cols, row)))
                  for row in df[list(spec.source_cols)].to_numpy()], dtype=np.float64)
    y = y[~_isnan_vector(y)]
    if spec.dtype is not None and spec.num_classes:
        vals, counts = np.unique(y.astype(int), return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}
    return {"n": len(y)}


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a BRSET CSV against the mBRSET schema.")
    p.add_argument("--csv", required=True, help="BRSET labels.csv")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--images-dir", default=None,
                   help="If given, check how many mapped filenames actually exist on disk.")
    p.add_argument("--compare-mbrset", default=None,
                   help="Path to labels_mbrset.csv, to print both distributions side by side.")
    p.add_argument("--inspect", action="store_true", help="(default behaviour)")
    args = p.parse_args()

    raw = pd.read_csv(args.csv)
    print(f"\n{'='*74}\nBRSET CSV: {args.csv}\n  {len(raw)} rows, {len(raw.columns)} columns\n{'='*74}")

    print("\nCOLUMN MAPPING")
    for src, dst in {**REQUIRED_MAP, **OPTIONAL_MAP}.items():
        req = "required" if src in REQUIRED_MAP else "optional"
        mark = "OK  " if src in raw.columns else ("MISSING" if req == "required" else "absent ")
        print(f"  {mark:8s} {src:<18} -> {dst:<18} ({req})")
    unmapped = [c for c in raw.columns if c not in {**REQUIRED_MAP, **OPTIONAL_MAP}]
    if unmapped:
        print(f"\n  not mapped ({len(unmapped)}): {', '.join(unmapped)}")
        print("  ^ BRSET-only labels with no mBRSET counterpart; they cannot be")
        print("    cross-tested, so they are dropped rather than carried through.")

    try:
        df = load_brset(raw, image_ext=args.image_ext)
    except KeyError as e:
        print(f"\n[FAIL] {e}")
        return 1
    print(f"\nADAPTED: {len(df)} rows, columns = {list(df.columns)}")
    if "file" in df.columns:
        print(f"  file sample: {list(df['file'].head(3))}")
    if "patient" in df.columns:
        print(f"  patients: {df['patient'].nunique()} "
              f"({len(df)/max(df['patient'].nunique(),1):.2f} images/patient)")
    if "sex" in df.columns:
        print(f"  sex values after re-encode: {sorted(pd.unique(df['sex'].dropna()))}")

    if args.images_dir:
        if os.path.isdir(args.images_dir):
            on_disk = set(os.listdir(args.images_dir))
            hit = df["file"].isin(on_disk).sum()
            print(f"\n  files found on disk: {hit}/{len(df)}")
            if hit == 0:
                print("  ^ ZERO matches. --image-ext is probably wrong, or the images")
                print("    live in subdirectories. Check a real filename before training.")
            elif hit < len(df):
                miss = df.loc[~df["file"].isin(on_disk), "file"].head(3).tolist()
                print(f"  missing examples: {miss}  (MBRSETDataset(drop_missing_files=True) drops these)")
        else:
            print(f"\n  [warn] images_dir does not exist: {args.images_dir}")

    print("\nTASK AVAILABILITY (rows with a usable label)")
    other = pd.read_csv(args.compare_mbrset) if args.compare_mbrset else None
    for task, need in TASK_REQUIREMENTS.items():
        if need not in df.columns:
            print(f"  {task:<14} unavailable (needs {need})")
            continue
        d = _distribution(df, task)
        line = f"  {task:<14} {d}"
        if other is not None:
            od = _distribution(other, task)
            line += f"\n  {'':<14} mBRSET: {od}"
        print(line)

    if other is not None:
        print("\nNB: compare the class balances above. A large prevalence gap between")
        print("    BRSET and mBRSET moves AUROC on its own, independently of any domain")
        print("    shift in the pixels — report prevalence alongside the transfer result.")

    print(f"\n{'='*74}")
    print("If the mapping above is right, train with:")
    print("  python train_mbrset.py --dataset brset --root <BRSET> \\")
    print("      --external-test-root <mBRSET> --external-test-dataset mbrset \\")
    print("      --task dr_referable --seeds ...")
    print(f"{'='*74}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
