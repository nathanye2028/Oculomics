"""
public_fundus.py
================
Adapters that load the public glaucoma / AMD fundus datasets into the mBRSET
column schema, so every one of them flows through ``dataset.MBRSETDataset``,
``train_mbrset.py`` and ``evaluate_deploy.py`` unchanged (via
``brset_dataset.load_any``). Same philosophy as ``brset_dataset.py``: the mapping
is explicit and auditable, values that cannot be trusted become NaN (dropped),
and ``--inspect`` shows what was actually joined before any GPU time is spent.

    --dataset airogs   Rotterdam EyePACS AIROGS. Two layouts: the Zenodo release
                       (train_labels.csv with challenge_id + class RG / NRG,
                       ~101k images) and the Kaggle "EyePACS AIROGS Light" subsets
                       (deathtrooper/eyepacs-airogs-light, ...-light-v2): balanced
                       RG/ and NRG/ class folders under train/validation/test, in
                       release-raw|pad|crop standardisations at 256 px -- no CSV.
                       The folder layout is used when no CSV is found; ONE release
                       directory is read (``airogs_release``, default release-crop)
                       so an image is never counted three times. No patient id is
                       published in either, so ``patient`` is the image id: assume
                       one image per subject (the split cannot group what it
                       cannot see). The light subsets are 50 % RG by construction.
    --dataset refuge   REFUGE: images in Glaucoma/ and Non-Glaucoma/ folders
                       (train) and/or a label table (ImgName, Label) for the
                       validation/test releases; mask folders are skipped.
    --dataset papila   PAPILA: ClinicalData/patient_data_od|os.xlsx (ID, Age,
                       Gender, Diagnosis 0 healthy / 1 glaucoma / 2 suspect) and
                       FundusImages/RET<ID>OD|OS.jpg; both eyes share a patient.
                       Suspects are EXCLUDED by default (``papila_suspect``).
    --dataset odir     ODIR-5K: full_df.csv (Kaggle, one row per eye) or the
                       per-patient annotation sheet (expanded to two rows); labels
                       ``glaucoma`` and ``amd`` come from the per-eye diagnostic
                       keywords; eyes flagged low-quality / no-fundus become NaN.

Every adapter returns ``{"df", "images_dir", "csv", "source"}`` with columns
``file`` (relative to ``images_dir``, may contain sub-directories), ``patient``,
the task columns ``glaucoma`` / ``amd`` (0/1/NaN), and ``age`` / ``sex`` /
``laterality`` when the release has them. ``.xlsx`` metadata needs ``openpyxl``
(in requirements.txt); a CSV export next to the sheet works too.

    python public_fundus.py --root <ODIR-5K> --dataset odir --inspect
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PUBLIC_DATASETS: Tuple[str, ...] = ("airogs", "refuge", "papila", "odir")
IMG_EXTS = (".jpg", ".jpeg", ".png")
_SKIP_DIR_TOKENS = ("mask", "annotation", "segmentation", "disc_cup", "cup_disc", "gt", "label")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _read_table(path: str, header="infer") -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path, header=(0 if header == "infer" else header))
        except ImportError as e:                      # pragma: no cover
            raise ImportError(f"{path}: reading .xlsx needs openpyxl "
                              f"(.venv/bin/pip install openpyxl), or export the sheet to CSV "
                              f"next to it") from e
    return pd.read_csv(path, header=(0 if header == "infer" else header))


def _read_table_with_header_row(path: str, key: str) -> pd.DataFrame:
    """Some releases put a title row above the header. Find the row whose first
    cells contain ``key`` (e.g. 'ID') and use it as the header."""
    raw = _read_table(path, header=None)
    for i in range(min(len(raw), 10)):
        cells = [str(c).strip() for c in raw.iloc[i].tolist()]
        if key in cells:
            df = raw.iloc[i + 1:].copy()
            df.columns = cells
            return df.reset_index(drop=True)
    return _read_table(path)


def _find_table(root: str, stems: Iterable[str], subdirs: Iterable[str] = ("",)) -> Optional[str]:
    for sub in subdirs:
        for stem in stems:
            for ext in (".csv", ".xlsx", ".xls"):
                p = os.path.join(root, sub, stem + ext)
                if os.path.isfile(p):
                    return p
    return None


def _first_dir(root: str, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return None


def _walk_images(images_dir: str) -> List[str]:
    """Relative paths of every image under ``images_dir``, skipping mask/annotation dirs."""
    out = []
    for dirpath, dirnames, files in os.walk(images_dir):
        dirnames[:] = [d for d in dirnames if not any(t in d.lower() for t in _SKIP_DIR_TOKENS)]
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.relpath(os.path.join(dirpath, f), images_dir))
    return sorted(out)


def _stem_index(images_dir: str) -> Dict[str, str]:
    """{lower-case stem: relative path} for every image under images_dir."""
    idx = {}
    for rel in _walk_images(images_dir):
        idx.setdefault(os.path.splitext(os.path.basename(rel))[0].lower(), rel)
    return idx


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """First column whose lower-cased, space/underscore-insensitive name matches."""
    norm = {re.sub(r"[\s_\-]", "", c.lower()): c for c in df.columns}
    for n in names:
        k = re.sub(r"[\s_\-]", "", n.lower())
        if k in norm:
            return norm[k]
    return None


def _finalise(df: pd.DataFrame, images_dir: str, dataset: str, csv: Optional[str]) -> Dict[str, object]:
    df = df.reset_index(drop=True)
    for c in ("glaucoma", "amd", "age", "sex"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["source"] = dataset
    return {"df": df, "images_dir": images_dir, "csv": csv, "source": dataset}


# --------------------------------------------------------------------------- #
# AIROGS
# --------------------------------------------------------------------------- #
def _airogs_from_folders(root: str, release: Optional[str] = None) -> Dict[str, object]:
    """Kaggle AIROGS-light layout: <release-*>/<train|validation|test>/<RG|NRG>/*.jpg."""
    releases = sorted(d for d in os.listdir(root)
                      if d.lower().startswith("release") and os.path.isdir(os.path.join(root, d)))
    images_dir = root
    if releases:
        pick = release or ("release-crop" if "release-crop" in releases else releases[0])
        if pick not in releases:
            raise FileNotFoundError(f"AIROGS: release {pick!r} not under {root}; found {releases}")
        images_dir = os.path.join(root, pick)
    elif release:
        raise FileNotFoundError(f"AIROGS: airogs_release={release!r} given but {root} has no release-* dirs")
    rel = _walk_images(images_dir)
    if not rel:
        raise FileNotFoundError(f"AIROGS: neither a label CSV nor images under {root}")
    rows = []
    for r in rel:
        parts = [p.lower() for p in r.split(os.sep)[:-1]]
        lab = 1.0 if "rg" in parts else (0.0 if "nrg" in parts else np.nan)
        split = next((p for p in parts if p in ("train", "training", "validation", "val", "test")), None)
        rows.append({"file": r, "patient": os.path.splitext(os.path.basename(r))[0],
                     "glaucoma": lab, "source_split": split})
    df = pd.DataFrame(rows)
    out = _finalise(df, images_dir, "airogs", None)
    out["release"] = os.path.basename(images_dir) if releases else None
    out["releases_available"] = releases
    return out


def load_airogs(root: str, image_ext: str = ".jpg", airogs_release: Optional[str] = None) -> Dict[str, object]:
    # Zenodo: train_labels.csv at the root. AIROGS-light v2: metadata.csv one level
    # down (file_name, label RG/NRG, folder). Light v1: class folders only.
    subdirs = ("",) + tuple(sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))))
    csv = _find_table(root, ("train_labels", "labels", "airogs_labels", "metadata"), subdirs=subdirs)
    if csv is None:                                 # class folders, no CSV
        return _airogs_from_folders(root, release=airogs_release)
    t = _read_table(csv)
    # Prefer a FILENAME column over a bare numeric id (v2's `id` is the source image
    # number, not the file), then fall back to the challenge id.
    idc = _col(t, "file_name", "filename", "file", "image_id", "challenge_id", "id")
    labc = _col(t, "class", "label", "label_binary", "glaucoma", "referable")
    if idc is None or labc is None:
        raise KeyError(f"AIROGS: need an id/filename column and a class column; found {list(t.columns)}")
    lab = t[labc].astype(str).str.strip().str.upper().map(
        {"RG": 1.0, "NRG": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
    images_dir = _first_dir(root, ("train", "images", "img")) or root
    idx = _stem_index(images_dir)
    ids = t[idc].astype(str).str.strip()
    stems = [os.path.splitext(os.path.basename(i))[0] for i in ids]
    files = [idx.get(st.lower(), (i if os.path.splitext(i)[1] else i + image_ext)) for st, i in zip(stems, ids)]
    df = pd.DataFrame({"file": files, "patient": stems, "glaucoma": lab.to_numpy()})
    fc = _col(t, "folder", "split")
    if fc:
        df["source_split"] = t[fc].astype(str).str.strip().str.lower().to_numpy()
    out = _finalise(df, images_dir, "airogs", csv)
    out["release"] = None
    out["releases_available"] = []
    return out


# --------------------------------------------------------------------------- #
# REFUGE
# --------------------------------------------------------------------------- #
def load_refuge(root: str, image_ext: str = ".jpg") -> Dict[str, object]:
    rel = _walk_images(root)
    if not rel:
        raise FileNotFoundError(f"REFUGE: no images under {root}")
    # Any label table under root (validation/test releases ship one).
    table: Dict[str, float] = {}
    tables = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".xlsx", ".xls", ".csv")):
                p = os.path.join(dirpath, f)
                try:
                    t = _read_table(p)
                except Exception:                     # pragma: no cover - unreadable sheet
                    continue
                ic, lc = _col(t, "ImgName", "image", "filename", "file"), _col(t, "Label", "Glaucoma Label", "glaucoma")
                if ic is None or lc is None:
                    continue
                tables.append(p)
                for img, lab in zip(t[ic].astype(str), pd.to_numeric(t[lc], errors="coerce")):
                    table[os.path.splitext(os.path.basename(img))[0].lower()] = float(lab) if lab == lab else np.nan
    rows = []
    for r in rel:
        parts = [p.lower() for p in r.split(os.sep)[:-1]]
        stem = os.path.splitext(os.path.basename(r))[0]
        if "non-glaucoma" in parts or "non_glaucoma" in parts or "normal" in parts:
            lab = 0.0
        elif "glaucoma" in parts:
            lab = 1.0
        else:
            lab = table.get(stem.lower(), np.nan)
        rows.append({"file": r, "patient": stem, "glaucoma": lab})
    df = pd.DataFrame(rows)
    return _finalise(df, root, "refuge", tables[0] if tables else None)


# --------------------------------------------------------------------------- #
# PAPILA
# --------------------------------------------------------------------------- #
def load_papila(root: str, image_ext: str = ".jpg", papila_suspect: str = "exclude") -> Dict[str, object]:
    if papila_suspect not in ("exclude", "positive", "negative"):
        raise ValueError("papila_suspect must be exclude | positive | negative")
    images_dir = _first_dir(root, ("FundusImages", "fundus_images", "images")) or root
    idx = _stem_index(images_dir)
    rows, csv = [], None
    for eye, lat in (("od", "R"), ("os", "L")):
        p = _find_table(root, (f"patient_data_{eye}",), subdirs=("ClinicalData", "clinical_data", ""))
        if p is None:
            raise FileNotFoundError(f"PAPILA: no ClinicalData/patient_data_{eye}.xlsx|csv under {root}")
        csv = csv or p
        t = _read_table_with_header_row(p, "ID")
        idc, dc = _col(t, "ID"), _col(t, "Diagnosis")
        if idc is None or dc is None:
            raise KeyError(f"PAPILA {p}: need ID and Diagnosis columns; found {list(t.columns)}")
        ac, gc = _col(t, "Age"), _col(t, "Gender", "Sex")
        for _, r in t.iterrows():
            digits = re.sub(r"\D", "", str(r[idc]))
            if not digits:
                continue
            stem = f"ret{digits.zfill(3)}{eye}"
            file = idx.get(stem, f"RET{digits.zfill(3)}{eye.upper()}{image_ext}")
            d = pd.to_numeric(r[dc], errors="coerce")
            lab = {0: 0.0, 1: 1.0}.get(int(d) if d == d else -1, np.nan)
            if d == 2:
                lab = {"exclude": np.nan, "positive": 1.0, "negative": 0.0}[papila_suspect]
            rows.append({"file": file, "patient": digits, "laterality": lat, "glaucoma": lab,
                         "age": r[ac] if ac else np.nan, "sex": r[gc] if gc else np.nan,
                         "papila_diagnosis": d})
    df = pd.DataFrame(rows)
    return _finalise(df, images_dir, "papila", csv)


# --------------------------------------------------------------------------- #
# ODIR-5K
# --------------------------------------------------------------------------- #
_LOW_QUALITY = ("low image quality", "lens dust", "optic disk photographically invisible",
                "image offset", "anterior segment image", "no fundus image")


def _odir_eye_labels(keywords) -> Tuple[float, float]:
    kw = str(keywords).lower() if keywords == keywords and keywords is not None else ""
    if not kw or any(q in kw for q in _LOW_QUALITY):
        return np.nan, np.nan
    glaucoma = 1.0 if "glaucoma" in kw else 0.0            # includes 'suspected glaucoma'
    amd = 1.0 if "macular degeneration" in kw else 0.0     # wet / dry age-related macular degeneration
    return glaucoma, amd


def load_odir(root: str, image_ext: str = ".jpg") -> Dict[str, object]:
    csv = _find_table(root, ("full_df", "data", "ODIR-5K_Training_Annotations(Updated)_V2", "odir"),
                      subdirs=("", "ODIR-5K", "ODIR-5K/ODIR-5K"))
    if csv is None:
        raise FileNotFoundError(f"ODIR-5K: no full_df.csv / data.xlsx under {root}")
    t = _read_table(csv)
    images_dir = _first_dir(root, ("preprocessed_images", "ODIR-5K/Training Images", "Training Images",
                                   "ODIR-5K/ODIR-5K/Training Images", "images")) or root
    idc, agec, sexc = _col(t, "ID"), _col(t, "Patient Age", "age"), _col(t, "Patient Sex", "sex")
    lkw, rkw = _col(t, "Left-Diagnostic Keywords"), _col(t, "Right-Diagnostic Keywords")
    if lkw is None or rkw is None:
        raise KeyError(f"ODIR-5K {csv}: need Left-/Right-Diagnostic Keywords; found {list(t.columns)}")
    sex_map = {"male": 0.0, "female": 1.0, "m": 0.0, "f": 1.0}
    rows = []
    fnc = _col(t, "filename")
    for _, r in t.iterrows():
        pid = str(r[idc]) if idc else None
        age = r[agec] if agec else np.nan
        sex = sex_map.get(str(r[sexc]).strip().lower(), np.nan) if sexc else np.nan
        if fnc:                                         # Kaggle full_df: one row per eye
            fn = str(r[fnc])
            side = "left" if "_left" in fn.lower() else "right"
            eyes = [(fn, side)]
        else:                                           # per-patient sheet: expand to two rows
            lf, rf = _col(t, "Left-Fundus"), _col(t, "Right-Fundus")
            eyes = [(str(r[lf]), "left"), (str(r[rf]), "right")]
        for fn, side in eyes:
            g, a = _odir_eye_labels(r[lkw] if side == "left" else r[rkw])
            if not os.path.splitext(fn)[1]:
                fn = fn + image_ext
            rows.append({"file": fn, "patient": pid if pid else os.path.splitext(fn)[0].split("_")[0],
                         "laterality": "L" if side == "left" else "R",
                         "glaucoma": g, "amd": a, "age": age, "sex": sex})
    df = pd.DataFrame(rows)
    return _finalise(df, images_dir, "odir", csv)


# --------------------------------------------------------------------------- #
# Dispatch + inspector
# --------------------------------------------------------------------------- #
_LOADERS = {"airogs": load_airogs, "refuge": load_refuge, "papila": load_papila, "odir": load_odir}


def load_public(root: str, dataset: str, image_ext: str = ".jpg", **kw) -> Dict[str, object]:
    if dataset not in _LOADERS:
        raise ValueError(f"unknown public dataset {dataset!r}; expected one of {PUBLIC_DATASETS}")
    return _LOADERS[dataset](root, image_ext=image_ext, **kw)


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a public fundus dataset against the mBRSET schema.")
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True, choices=PUBLIC_DATASETS)
    p.add_argument("--task", default=None, help="Task column to report (default: every one present).")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--papila-suspect", default="exclude", choices=["exclude", "positive", "negative"])
    p.add_argument("--airogs-release", default=None,
                   help="AIROGS-light only: which release-* directory to read (default release-crop).")
    p.add_argument("--inspect", action="store_true", help="(default behaviour)")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if the task column is missing/single-class or no image is on disk.")
    args = p.parse_args()

    kw = {"papila_suspect": args.papila_suspect} if args.dataset == "papila" else {}
    if args.dataset == "airogs" and args.airogs_release:
        kw["airogs_release"] = args.airogs_release
    src = load_public(args.root, args.dataset, image_ext=args.image_ext, **kw)
    df = src["df"]
    print(f"\n{'='*74}\n{args.dataset.upper()} @ {args.root}\n  table: {src['csv']}\n  images_dir: {src['images_dir']}"
          f"\n  {len(df)} rows, {df['patient'].nunique()} patients ({len(df)/max(df['patient'].nunique(),1):.2f} images/patient)"
          f"\n  columns: {list(df.columns)}\n{'='*74}")
    on_disk = sum(os.path.isfile(os.path.join(src["images_dir"], f)) for f in df["file"].head(2000))
    print(f"  files on disk (first {min(len(df), 2000)} rows): {on_disk}"
          f"   sample: {list(df['file'].head(3))}")
    failed = on_disk == 0
    tasks = [args.task] if args.task else [c for c in ("glaucoma", "amd") if c in df.columns]
    for task in tasks:
        if task not in df.columns:
            print(f"  {task:<10} unavailable in this dataset"); failed = True; continue
        y = df[task]
        n, pos = int(y.notna().sum()), int((y == 1).sum())
        ok = 0 < pos < n
        print(f"  {task:<10} n={n:<7} pos={pos:<6} prev={pos / n if n else 0:.2%}  "
              f"dropped(NaN)={int(y.isna().sum())}  {'ok' if ok else 'SINGLE-CLASS / EMPTY'}")
        failed |= not ok
    if args.dataset == "papila" and "papila_diagnosis" in df.columns:
        print(f"  PAPILA diagnosis counts (0 healthy / 1 glaucoma / 2 suspect): "
              f"{df['papila_diagnosis'].value_counts(dropna=False).to_dict()}  suspect={args.papila_suspect}")
    if args.dataset == "airogs":
        if src.get("release"):
            print(f"  AIROGS-light release read: {src['release']}  (available: {src['releases_available']}; "
                  f"--airogs-release to change)")
        if "source_split" in df.columns:
            print(f"  source folders: {df['source_split'].value_counts(dropna=False).to_dict()}  "
                  f"(informational -- the trainer makes its own patient-grouped split)")
        print("  NB: AIROGS publishes no patient id; the patient-grouped split treats every image")
        print("      as its own subject. Say so when reporting an in-domain AIROGS number.")
    print(f"{'='*74}\n")
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
