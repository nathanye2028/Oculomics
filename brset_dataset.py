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

Value re-encoding (verified against a real BRSET CSV, n=1619)
-------------------------------------------------------------
Renaming the columns is not enough — three of them carry values that mean
something different from their mBRSET counterparts. All three were caught by
reading an actual CSV, not the spec, and all three fail *silently* or crash far
from the cause:

``patient_sex``  1/2 -> 0/1
    ``LABEL_REGISTRY["sex"]`` declares ``num_classes=2`` and passes the raw value
    through. A 2 indexes past the end of a 2-logit head and raises inside
    CrossEntropyLoss with no mention of the CSV.

``quality``  Adequate/Inadequate -> 1/0
    mBRSET stores yes/no here, and ``dataset._yes_no`` returns 1.0 only for the
    literal string "yes". Left alone, every BRSET row — gradable or not —
    becomes 0.0, and the quality head trains against a constant.

``artifacts``  1/2 -> 0/1, **polarity inverted**
    BRSET's 1/2 follows its focus/illumination/image_field convention where
    1 = adequate and 2 = inadequate, so 2 means artifacts ARE present. mBRSET's
    ``final_artifacts="yes"`` also means present. The crosstab that settled it:
    ``artifacts=2`` occurs only within ``quality=Inadequate`` (7/7). Passed
    through raw this both crashes on the label 2 and, for every other row, trains
    "clean image" as "artifacts present".

Only ``sex`` affects the DR tasks; ``quality``/``artifacts`` matter if you use
the image-quality heads, which is precisely the artifact-resilience angle this
project cares about, so they are fixed rather than dropped. ``DR_ICDR`` (0-4)
and ``macular_edema`` (0/1) needed no re-encoding — they already match.

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


def _normalise_quality(s: pd.Series) -> pd.Series:
    """BRSET 'Adequate'/'Inadequate' -> 1/0, matching mBRSET's yes/no polarity.

    ``dataset._yes_no`` only recognises the literal "yes", so without this every
    row lands on 0.0 and the head trains against a constant label — no error, no
    warning, just an AUROC of 0.5 that looks like a modelling problem.
    """
    if s.dtype == object:
        m = s.astype(str).str.strip().str.lower()
        return m.map({"adequate": 1.0, "inadequate": 0.0,
                      "yes": 1.0, "no": 0.0}).astype(float)
    return _flip_12(s)          # some releases use the 1/2 convention here too


def _flip_12(s: pd.Series) -> pd.Series:
    """BRSET quality-parameter convention 1=adequate / 2=inadequate -> 1/0.

    A dict ``map`` rather than ``(s == 1)``: the comparison turns a missing cell
    into False -> 0.0, i.e. an ungraded image silently becomes a confident
    "inadequate" label. ``map`` leaves NaN as NaN, so ``MBRSETDataset``'s
    ``drop_missing_labels`` removes the row instead.
    """
    vals = set(pd.unique(s.dropna()))
    if vals and vals <= {1, 2, 1.0, 2.0}:
        return s.map({1: 1.0, 2: 0.0}).astype(float)
    return s


def _normalise_artifacts(s: pd.Series) -> pd.Series:
    """BRSET artifacts 1/2 -> 0/1 with the polarity INVERTED.

    BRSET follows its focus/illumination convention: 1 = adequate, 2 = inadequate,
    so 2 is the row that HAS artifacts. mBRSET's ``final_artifacts="yes"`` also
    means present, so 2 -> 1 and 1 -> 0. Verified on a real CSV: artifacts=2
    occurred only among quality=Inadequate rows (7/7).

    Get this backwards and the label is not merely wrong, it is anti-correlated
    with the truth — the model learns to call clean images artifacted.

    NaN stays NaN (see :func:`_flip_12`): ``(s == 2)`` would turn every
    unlabelled row into a confident "no artifacts".
    """
    vals = set(pd.unique(s.dropna()))
    if vals and vals <= {1, 2, 1.0, 2.0}:
        return s.map({1: 0.0, 2: 1.0}).astype(float)
    if s.dtype == object:
        m = s.astype(str).str.strip().str.lower()
        return m.map({"yes": 1.0, "no": 0.0}).astype(float)
    return s


# mBRSET column -> re-encoder, applied after the rename. Keyed on the *destination*
# name so it reads against the schema dataset.py consumes.
VALUE_NORMALISERS = {
    "sex": _normalise_sex,
    "final_quality": _normalise_quality,
    "final_artifacts": _normalise_artifacts,
}


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
    for col, fn in VALUE_NORMALISERS.items():
        if col in out.columns:
            out[col] = fn(out[col])
    if "final_icdr" in out.columns:
        # dataset.py's _icdr_grade does int() on this; a stray string grade would
        # otherwise raise per-row deep inside label vectorisation.
        out["final_icdr"] = pd.to_numeric(out["final_icdr"], errors="coerce")

    out.attrs["source"] = "brset"
    return out


# Label CSV names per dataset. The name varies by release (PhysioNet 1.0.1 ships
# labels_brset.csv, the Kaggle BRSET mirror labels.csv), and PhysioNet downloads
# drop a version directory (mBRSET/1.0/labels_mbrset.csv), so a root is resolved
# by SEARCHING a few levels for the CSV rather than by hardcoding one path.
LABEL_CSV_NAMES: Dict[str, List[str]] = {
    "mbrset": ["labels_mbrset.csv", "dataframe_brsetmobile.csv"],
    "brset": ["labels_brset.csv", "labels.csv", "dataframe_brset.csv"],
}


def resolve_root(root: str, dataset: str, max_depth: int = 3) -> str:
    """Return the directory that actually holds ``dataset``'s label CSV, searching
    up to ``max_depth`` levels below ``root``. Raises FileNotFoundError naming
    every CSV that WAS found, so a wrong root is fixed from the message."""
    names = LABEL_CSV_NAMES[dataset]
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{dataset}: root does not exist: {root}")
    if any(os.path.isfile(os.path.join(root, n)) for n in names):
        return root
    hits, csvs = [], []
    for dirpath, dirnames, files in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in ("images", "fundus_photos", "photos")]
        if any(n in files for n in names):
            hits.append(dirpath)
        csvs += [os.path.join(dirpath, f) for f in files if f.lower().endswith(".csv")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(f"{dataset}: label CSV found in several places under {root}: {hits}. "
                                f"Pass the one you mean.")
    raise FileNotFoundError(
        f"{dataset}: none of {names} within {max_depth} levels of {root}. "
        f"CSVs found there: {sorted(csvs)[:12] or 'none'}. Pass the directory that holds "
        f"the label CSV and the images/ folder (e.g. the PhysioNet version dir mBRSET/1.0).")


def load_any(root: str, dataset: str, image_ext: str = ".jpg") -> Dict[str, str]:
    """Resolve ``root`` to a (DataFrame, images_dir) pair for either dataset.

    Lets a trainer accept ``--dataset {mbrset,brset}`` without branching on the
    schema everywhere. Returns a dict so the caller can log what it resolved.
    ``root`` may be a parent of the real dataset directory (see resolve_root).
    """
    if dataset == "mbrset":
        root = resolve_root(root, dataset)
        csv_path = next(os.path.join(root, n) for n in LABEL_CSV_NAMES["mbrset"]
                        if os.path.isfile(os.path.join(root, n)))
        img_dir = os.path.join(root, "images")
        return {"df": pd.read_csv(csv_path), "images_dir": img_dir,
                "csv": csv_path, "source": "mbrset"}
    if dataset == "brset":
        root = resolve_root(root, dataset)
        csv_names = LABEL_CSV_NAMES["brset"]
        csv_path = next(os.path.join(root, n) for n in csv_names
                        if os.path.isfile(os.path.join(root, n)))
        # BRSET's image directory is named differently across releases too.
        candidates = ["fundus_photos", "images", "photos"]
        img_dir = next((os.path.join(root, c) for c in candidates
                        if os.path.isdir(os.path.join(root, c))), os.path.join(root, "fundus_photos"))
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
    # Show every re-encoded column as raw -> adapted. These are the mappings most
    # likely to be wrong on an unfamiliar release, and the ones that fail quietly.
    inv = {v: k for k, v in {**REQUIRED_MAP, **OPTIONAL_MAP}.items()}
    print("\n  VALUE RE-ENCODING (raw -> adapted)")
    for col in VALUE_NORMALISERS:
        if col not in df.columns:
            continue
        src = inv.get(col, col)
        raw_vals = sorted(map(str, pd.unique(raw[src].dropna()))) if src in raw.columns else ["?"]
        new_vals = sorted(pd.unique(df[col].dropna()))
        print(f"    {src:<14} {str(raw_vals):<34} -> {new_vals}")
        if len(new_vals) and set(new_vals) - {0.0, 1.0, 0, 1}:
            print(f"      ^ NOT 0/1 after re-encode. A 2 here crashes CrossEntropyLoss;")
            print(f"        add a rule to VALUE_NORMALISERS for this release.")
    print("    ^ verify the POLARITY, not just the range. BRSET's artifacts column")
    print("      is inverted relative to mBRSET's (2 = present); a silent flip trains")
    print("      the model to call clean images artifacted.")

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
