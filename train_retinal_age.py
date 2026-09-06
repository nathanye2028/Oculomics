#!/usr/bin/env python3
"""
train_retinal_age.py
====================
**Retinal age** from a single fundus photograph, and the **retinal age gap**
(predicted minus chronological age) that the disease-association step needs.

Why a separate trainer and not ``train_mbrset.py --task age``
-------------------------------------------------------------
``train_mbrset.py`` is a classifier through and through: cross-entropy,
AUROC-based checkpoint selection, class-balanced sampling, softmax evaluation,
class-count checks against the external set. Retinal age is a regression with
its own selection metric (val MAE in years), its own cohort rule and its own
deliverable (a per-image prediction table), so it gets its own trainer that
*reuses* the classifier's parts rather than forking them: the same
:class:`dataset.MBRSETDataset` (so BRSET and mBRSET get the identical FOV crop,
resize, augmentation and normalisation — see ``brset_dataset.py`` for why that
matters), the same :class:`model.MBRSETClassifier` in ``regression=True`` mode,
the same EMA / AMP / warmup-cosine recipe, and the same label-free AdaBN.

Cohort: learn *healthy* ageing
------------------------------
Disease shifts the retinal age gap — that is the reason to measure it — so the
model is fit on retinas that are ageing normally and the diseased ones are held
back to be *scored*, never trained on.

``--healthy`` (image quality is always required except for ``all``):

* ``nodm`` (default) — the patient has no diabetes (BRSET ``diabetes`` column)
  AND every image of the patient is DR grade 0. Patient-level on purpose: one
  eye with retinopathy disqualifies the fellow eye too.
* ``dr0`` — every image of the patient is DR grade 0; diabetics without
  retinopathy are kept. Use it when a release has no ``diabetes`` column.
* ``gradable`` — adequate quality only.
* ``all`` — no filter at all (ablation).

``--exclude-pathology`` additionally drops patients with any BRSET ophthalmic
flag set (AMD, drusen, increased cup-disc ratio, hypertensive retinopathy,
vascular occlusion, haemorrhage, myopic fundus, retinal detachment, scar,
nevus). Those columns are not part of the mBRSET schema, so they are read from
the raw BRSET CSV and joined on the image id.

Splits. The patient-grouped, age-stratified 70/10/20 split is drawn over ALL
patients first; the healthy filter is applied to train and val afterwards. So
every patient sits in exactly one split, the non-healthy patients of the train
and val partitions are simply never shown to the model, and every non-healthy
BRSET image can be scored without leakage (``cohort=nonhealthy`` in the
predictions table). Checkpoint selection uses the healthy val MAE.

What is reported
----------------
* MAE, RMSE, Pearson r, R², mean gap, and **MAE by age bin** — BRSET's age
  distribution skews 40-70, so the tails are where it is weak and the table
  says by how much. Patient-level MAE averages both eyes.
* mBRSET zero-shot and after AdaBN, with a DR-grade breakdown (grade 0
  diabetics / grade 1 / referable). Every mBRSET patient is diabetic and the
  camera differs, so the plain mBRSET MAE mixes device shift with a biological
  shift; the within-mBRSET breakdown is the cleaner signal.
* **Age-gap bias correction.** A regression's gap is anti-correlated with age
  (regression to the mean: the young are over-, the old under-predicted).
  Following Beheshti et al. (2019), ``gap = a + b*age`` is fit on the healthy
  VAL predictions and subtracted everywhere (``gap_corrected``). Any disease
  association must use the corrected gap; the raw one is age-confounded.
* ``<ckpt-dir>/<run>_predictions.csv`` — one row per scored image (BRSET test,
  BRSET non-healthy, mBRSET) with age, prediction, raw and corrected gap and
  the clinical metadata that travels with the row. This is the input to the
  disease-association step.

Run::

    python train_retinal_age.py --root <BRSET> --external-test-root <mBRSET> --inspect
    python train_retinal_age.py --root <BRSET> --external-test-root <mBRSET> \\
        --seed 0 --bn-adapt --results-json exp_retinal_age/student_seed0.json

``run_retinal_age.sh`` runs the seeds; ``summarize_retinal_age.py`` aggregates.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=r".*epoch parameter in `scheduler.step\(\)`.*")
import torch.nn as nn                                    # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset, DeviceAug                       # noqa: E402
from model import MBRSETClassifier                                  # noqa: E402
from fundus_utils import seed_everything, seed_worker               # noqa: E402
from metrics import CSVLogger                                       # noqa: E402
from train_mbrset import ModelEMA, adapt_bn, pick_device, backbone_kwargs_for   # noqa: E402

TASK = "retinal_age"
HEALTHY_DEFS = ("nodm", "dr0", "gradable", "all")
# Age bins for the per-bin MAE table. BRSET is 40-70 heavy; <30 and 80+ are thin.
AGE_BIN_EDGES = (30, 40, 50, 60, 70, 80)
AGE_BIN_LABELS = ("<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+")
# BRSET's ophthalmic flags beyond DR (raw CSV column names).
PATHOLOGY_COLS = ("amd", "drusens", "increased_cup_disc", "hypertensive_retinopathy",
                  "vascular_occlusion", "hemorrhage", "myopic_fundus", "retinal_detachment",
                  "scar", "nevus")
# Metadata carried into the predictions table when present (either schema).
META_COLS = ("sex", "laterality", "dr_grade", "gradable", "diabetes", "dm_time", "final_edema",
             "insulin", "systemic_hypertension", "nephropathy", "neuropathy",
             "acute_myocardial_infarction", "vascular_disease", "diabetic_foot", "obesity",
             "smoking", "alcohol_consumption", "camera")

_POS_TOKENS = frozenset({"yes", "y", "true", "t", "1", "1.0", "sim", "present", "positive"})
_NEG_TOKENS = frozenset({"no", "n", "false", "f", "0", "0.0", "nao", "não", "absent", "negative"})


def _flag(value) -> float:
    """Strict yes/no -> 1.0/0.0; anything unrecognised -> NaN (never a silent 0).

    Same contract as ``_binary_flag`` on the systemic branch: an unaudited
    encoding (a 2 from a 1/2 release, "unknown") must NOT become a confident
    negative, because here a false "no diabetes" puts a diabetic into the
    healthy training cohort.
    """
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        t = value.strip().lower()
        if t in _POS_TOKENS:
            return 1.0
        if t in _NEG_TOKENS:
            return 0.0
        return np.nan
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(f):
        return np.nan
    return 1.0 if f == 1.0 else (0.0 if f == 0.0 else np.nan)


def age_bin(age) -> np.ndarray:
    """Age -> bin index into AGE_BIN_LABELS."""
    return np.digitize(np.asarray(age, dtype=np.float64), AGE_BIN_EDGES)


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #
def build_cohort(df: pd.DataFrame, healthy: str = "nodm", exclude_pathology: bool = False,
                 raw_csv: Union[str, pd.DataFrame, None] = None, image_ext: str = ".jpg",
                 min_age: Optional[float] = None, max_age: Optional[float] = None) -> pd.DataFrame:
    """Annotate an mBRSET-schema frame with the healthy-cohort decision.

    Returns a copy restricted to rows with a numeric age (and inside
    ``[min_age, max_age]`` if given) with these added columns:

    ``gradable`` (1/0/NaN), ``diabetes`` (1/0/NaN), ``dr_grade`` (float),
    ``pathology`` (1/0/NaN; only with ``exclude_pathology``),
    ``disease_free_image``, ``disease_free_patient`` (patient-level AND over its
    images), ``healthy`` (the training-eligibility flag) and ``exclusion``
    (the first reason a row is not healthy, "" if it is).

    Raises when the definition needs a column the release does not have, rather
    than silently training on everyone.
    """
    if healthy not in HEALTHY_DEFS:
        raise ValueError(f"healthy must be one of {HEALTHY_DEFS}, got {healthy!r}")
    for c in ("file", "patient", "age"):
        if c not in df.columns:
            raise KeyError(f"cohort needs column {c!r}; columns present: {sorted(df.columns)}")
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out = out[out["age"].notna()]
    if min_age is not None:
        out = out[out["age"] >= min_age]
    if max_age is not None:
        out = out[out["age"] <= max_age]
    out = out.reset_index(drop=True)

    if "final_quality" in out.columns:
        out["gradable"] = out["final_quality"].map(_flag).astype(float)
    else:
        warnings.warn("no final_quality column: every image is treated as gradable")
        out["gradable"] = 1.0
    out["diabetes"] = (out["diabetes"].map(_flag).astype(float) if "diabetes" in out.columns
                       else np.nan)
    out["dr_grade"] = (pd.to_numeric(out["final_icdr"], errors="coerce")
                       if "final_icdr" in out.columns else np.nan)

    if exclude_pathology:
        if raw_csv is None:
            raise ValueError("--exclude-pathology needs the raw BRSET CSV (its ophthalmic flag "
                             "columns are not part of the mBRSET schema)")
        raw = pd.read_csv(raw_csv) if isinstance(raw_csv, str) else raw_csv
        cols = [c for c in PATHOLOGY_COLS if c in raw.columns]
        if not cols or "image_id" not in raw.columns:
            raise KeyError(f"raw CSV has none of the pathology columns {PATHOLOGY_COLS} "
                           f"(or no image_id); columns: {sorted(raw.columns)}")
        key = raw["image_id"].astype(str).map(
            lambda f: f if os.path.splitext(f)[1] else f + image_ext)
        flags = raw[cols].apply(lambda s: s.map(_flag))
        # max over flags: 1 if any flag set, 0 if all known-negative, NaN if all unknown
        any_path = flags.max(axis=1, skipna=True)
        out["pathology"] = out["file"].map(pd.Series(any_path.values, index=key.values)).astype(float)
    else:
        out["pathology"] = np.nan

    if healthy == "nodm":
        if out["diabetes"].isna().all():
            raise ValueError("--healthy nodm needs a usable 'diabetes' column (BRSET's "
                             "'diabetes' yes/no). None found or none parseable; use "
                             "--healthy dr0 or fix the mapping in brset_dataset.py.")
        img_ok = (out["dr_grade"] == 0) & (out["diabetes"] == 0)   # NaN -> False
    elif healthy == "dr0":
        img_ok = out["dr_grade"] == 0
    else:
        img_ok = pd.Series(True, index=out.index)
    if exclude_pathology:
        img_ok = img_ok & (out["pathology"] == 0)                   # unknown -> excluded
    out["disease_free_image"] = img_ok.astype(bool)
    out["disease_free_patient"] = out.groupby("patient")["disease_free_image"].transform("all").astype(bool)

    if healthy == "all":
        out["healthy"] = True
    else:
        out["healthy"] = out["disease_free_patient"] & (out["gradable"] == 1.0)

    # First reason a row is not healthy, for the cohort report.
    reason = np.where(out["healthy"], "", "other")
    if healthy != "all":
        reason = np.where(~out["healthy"] & (out["gradable"] != 1.0), "ungradable", reason)
    if healthy == "nodm":
        reason = np.where(~out["healthy"] & (out["gradable"] == 1.0)
                          & out.groupby("patient")["diabetes"].transform(lambda s: (s != 0).any()),
                          "diabetes", reason)
    if healthy in ("nodm", "dr0"):
        has_dr = out.groupby("patient")["dr_grade"].transform(lambda s: (s != 0).any())
        reason = np.where((reason == "other") & has_dr, "dr", reason)
    if exclude_pathology:
        has_p = out.groupby("patient")["pathology"].transform(lambda s: (s != 0).any())
        reason = np.where((reason == "other") & has_p, "pathology", reason)
    out["exclusion"] = reason
    return out


def split_by_patient(df: pd.DataFrame, seed: int, val_frac: float = 0.10,
                     test_frac: float = 0.20) -> pd.Series:
    """Patient-grouped 70/10/20 split, stratified on (age bin, healthy).

    Mirrors :func:`dataset.stratified_split` (two StratifiedGroupKFold stages,
    same seeding) but stratifies on the age decade rather than a class label,
    and on the healthy flag so each split holds a similar healthy fraction.
    Returns a Series of 'train' | 'val' | 'test' aligned with ``df``.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    if "patient" not in df.columns:
        raise ValueError("split_by_patient needs a 'patient' column; a per-image split would "
                         "let both eyes of a patient straddle train and test.")
    y = age_bin(df["age"].to_numpy()) * 2 + df["healthy"].astype(int).to_numpy()
    groups = df["patient"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # "least populated class" from thin bins
        n_splits = max(2, int(round(1.0 / max(test_frac, 1e-6))))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tv_idx, test_idx = next(sgkf.split(df, y, groups))
        rel_val = val_frac / (1.0 - test_frac)
        n_v = max(2, int(round(1.0 / max(rel_val, 1e-6))))
        sgkf_v = StratifiedGroupKFold(n_splits=n_v, shuffle=True, random_state=seed)
        tr_rel, val_rel = next(sgkf_v.split(df.iloc[tv_idx], y[tv_idx], groups[tv_idx]))
    split = np.empty(len(df), dtype=object)
    split[test_idx] = "test"
    split[tv_idx[tr_rel]] = "train"
    split[tv_idx[val_rel]] = "val"
    return pd.Series(split, index=df.index, name="split")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(age, pred, patient=None) -> Dict[str, object]:
    """MAE / RMSE / Pearson r / R² / mean gap, MAE by age bin, patient-level MAE."""
    age = np.asarray(age, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = int(len(age))
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mae": nan, "rmse": nan, "r": nan, "r2": nan, "mean_gap": nan,
                "sd_gap": nan, "gap_age_slope": nan, "by_bin": {}, "n_patients": 0,
                "patient_mae": nan}
    gap = pred - age
    ss_tot = float(((age - age.mean()) ** 2).sum())
    r = float(np.corrcoef(age, pred)[0, 1]) if n > 1 and age.std() > 0 and pred.std() > 0 else nan
    slope = float(np.polyfit(age, gap, 1)[0]) if n > 2 and age.std() > 0 else nan
    bins = age_bin(age)
    by_bin = {}
    for b, lab in enumerate(AGE_BIN_LABELS):
        m = bins == b
        by_bin[lab] = {"n": int(m.sum()),
                       "mae": float(np.abs(gap[m]).mean()) if m.any() else nan,
                       "mean_gap": float(gap[m].mean()) if m.any() else nan}
    out = {"n": n, "mae": float(np.abs(gap).mean()), "rmse": float(np.sqrt((gap ** 2).mean())),
           "r": r, "r2": (float(1.0 - (gap ** 2).sum() / ss_tot) if ss_tot > 0 else nan),
           "mean_gap": float(gap.mean()), "sd_gap": float(gap.std()),
           "gap_age_slope": slope, "by_bin": by_bin, "n_patients": 0, "patient_mae": nan}
    if patient is not None:
        pp = pd.DataFrame({"p": np.asarray(patient), "age": age, "pred": pred}).groupby("p").mean()
        out["n_patients"] = int(len(pp))
        out["patient_mae"] = float(np.abs(pp["pred"] - pp["age"]).mean())
    return out


def bias_fit(age, gap) -> Dict[str, float]:
    """Fit ``gap = a + b*age`` (Beheshti et al. 2019). Returns {a, b, n}."""
    age = np.asarray(age, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    n = int(len(age))
    if n < 3 or age.std() == 0:
        return {"a": float(gap.mean()) if n else 0.0, "b": 0.0, "n": n}
    b, a = np.polyfit(age, gap, 1)
    return {"a": float(a), "b": float(b), "n": n}


def bias_apply(age, gap, fit: Dict[str, float]) -> np.ndarray:
    """Corrected gap = gap - (a + b*age)."""
    age = np.asarray(age, dtype=np.float64)
    return np.asarray(gap, dtype=np.float64) - (fit["a"] + fit["b"] * age)


def _with_corrected(m: Dict[str, object], age, pred, fit) -> Dict[str, object]:
    if m["n"]:
        gc = bias_apply(age, np.asarray(pred) - np.asarray(age), fit)
        m["mean_gap_corrected"] = float(gc.mean())
        m["sd_gap_corrected"] = float(gc.std())
    else:
        m["mean_gap_corrected"] = m["sd_gap_corrected"] = float("nan")
    return m


def fmt_bins(by_bin: Dict[str, Dict[str, float]]) -> str:
    cells = []
    for lab in AGE_BIN_LABELS:
        b = by_bin.get(lab)
        if not b or not b["n"]:
            cells.append(f"{lab}: -")
        else:
            cells.append(f"{lab}: {b['mae']:.1f} (n={b['n']}, gap {b['mean_gap']:+.1f})")
    return "  ".join(cells)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader, device, mean: float, std: float) -> np.ndarray:
    """Predicted age (years) in loader order. The head outputs standardised age."""
    model.eval()
    outs = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        z = model(x).float().cpu().numpy().reshape(-1)
        outs.append(z * std + mean)
    return np.concatenate(outs) if outs else np.zeros(0)


def score_frame(model, ds: MBRSETDataset, frame: pd.DataFrame, loader, device,
                mean: float, std: float) -> pd.DataFrame:
    """Score ``ds`` (built from ``frame``) and return ``frame`` rows that were
    actually scored (the dataset drops missing files) with a ``pred_age`` column."""
    preds = predict(model, loader, device, mean, std)
    got = pd.DataFrame({"file": np.asarray(ds.files, dtype=object), "pred_age": preds})
    merged = frame.merge(got, on="file", how="inner")
    merged["gap"] = merged["pred_age"] - merged["age"]
    return merged


def cohort_report(df: pd.DataFrame, split: Optional[pd.Series], healthy: str, title: str) -> str:
    L = [f"\n--- {title}: {len(df)} images, {df['patient'].nunique()} patients "
         f"(healthy={healthy}) ---"]
    if split is not None and "healthy" in df.columns:      # training set only
        h = df[df["healthy"]]
        L.append(f"  healthy (training-eligible): {len(h)} images, {h['patient'].nunique()} patients")
        ex = df.loc[~df["healthy"], "exclusion"].value_counts()
        if len(ex):
            L.append("  excluded: " + ", ".join(f"{k}={v}" for k, v in ex.items()))
    if split is not None:
        for sp in ("train", "val", "test"):
            m = split == sp
            hm = m & df["healthy"]
            hist = np.bincount(age_bin(df.loc[hm, "age"]), minlength=len(AGE_BIN_LABELS))
            L.append(f"  {sp:<5} all={int(m.sum()):>6} imgs/{df.loc[m, 'patient'].nunique():>5} pts   "
                     f"healthy={int(hm.sum()):>6} imgs/{df.loc[hm, 'patient'].nunique():>5} pts   "
                     f"healthy age bins " + " ".join(f"{l}:{c}" for l, c in zip(AGE_BIN_LABELS, hist)))
    else:
        hist = np.bincount(age_bin(df["age"]), minlength=len(AGE_BIN_LABELS))
        L.append("  age bins " + " ".join(f"{l}:{c}" for l, c in zip(AGE_BIN_LABELS, hist)))
        if "dr_grade" in df.columns:
            vc = df["dr_grade"].value_counts(dropna=False).sort_index()
            L.append("  DR grade " + " ".join(f"{k}:{v}" for k, v in vc.items()))
    L.append(f"  age: mean={df['age'].mean():.1f} sd={df['age'].std():.1f} "
             f"min={df['age'].min():.0f} max={df['age'].max():.0f}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retinal age regression: BRSET healthy cohort -> mBRSET.")
    p.add_argument("--root", required=True, help="Training dataset root (BRSET).")
    p.add_argument("--dataset", default="brset", choices=["mbrset", "brset"])
    p.add_argument("--external-test-root", default=None, help="mBRSET root, scored zero-shot.")
    p.add_argument("--external-test-dataset", default="mbrset", choices=["mbrset", "brset"])
    p.add_argument("--external-all", action="store_true",
                   help="Score every external image with an age; default keeps gradable only.")
    p.add_argument("--image-ext", default=".jpg")
    # cohort
    p.add_argument("--healthy", default="nodm", choices=list(HEALTHY_DEFS),
                   help="Training-cohort rule (see module docstring). Default nodm.")
    p.add_argument("--exclude-pathology", action="store_true",
                   help="Also drop patients with any BRSET ophthalmic flag (raw CSV).")
    p.add_argument("--min-age", type=float, default=None)
    p.add_argument("--max-age", type=float, default=None)
    p.add_argument("--inspect", action="store_true",
                   help="Print the cohort / split report and exit without touching images.")
    # model / recipe
    p.add_argument("--backbone", default="timm:mobilenetv4_conv_small.e2400_r224_in1k",
                   help="mobilenetv3_small or timm:<name>. GCG is not used for this task.")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--loss", default="l1", choices=["l1", "huber", "mse"],
                   help="On standardised age. l1 optimises MAE directly.")
    p.add_argument("--huber-delta", type=float, default=1.0, help="In SD units of age.")
    p.add_argument("--age-balance", action="store_true",
                   help="Sample train images with weight 1/sqrt(age-bin frequency) to lift the tails.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--gpu-aug", dest="gpu_aug", action="store_true", default=None)
    p.add_argument("--no-gpu-aug", dest="gpu_aug", action="store_false")
    p.add_argument("--bn-adapt", action="store_true",
                   help="AdaBN on the external IMAGES (no labels) -> 'external_bnadapt'.")
    p.add_argument("--bn-adapt-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-dir", default="ck_retinal_age")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-json", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.external_test_root and \
            os.path.realpath(args.root) == os.path.realpath(args.external_test_root):
        raise SystemExit("[fatal] --root and --external-test-root are the same directory")

    from brset_dataset import load_any                      # noqa: E402
    src = load_any(args.root, args.dataset, image_ext=args.image_ext)
    cohort = build_cohort(src["df"], healthy=args.healthy, exclude_pathology=args.exclude_pathology,
                          raw_csv=(src["csv"] if args.exclude_pathology else None),
                          image_ext=args.image_ext, min_age=args.min_age, max_age=args.max_age)
    if cohort["healthy"].sum() == 0:
        raise SystemExit(f"[fatal] no healthy images under --healthy {args.healthy}; "
                         f"run --inspect and check the exclusion reasons.")
    cohort["split"] = split_by_patient(cohort, seed=args.seed)
    print(cohort_report(cohort, cohort["split"], args.healthy,
                        f"{args.dataset} @ {args.root}"))

    ext_df = None
    if args.external_test_root:
        ext = load_any(args.external_test_root, args.external_test_dataset, image_ext=args.image_ext)
        ext_df = build_cohort(ext["df"], healthy="gradable")
        if not args.external_all:
            ext_df = ext_df[ext_df["gradable"] == 1.0].reset_index(drop=True)
        print(cohort_report(ext_df, None, "gradable" if not args.external_all else "all",
                            f"external {args.external_test_dataset} @ {args.external_test_root}"))
    if args.inspect:
        print("\n[inspect] no images touched; drop --inspect to train.")
        return 0

    seed_everything(args.seed, deterministic=not args.nondeterministic)
    device = pick_device()
    run_name = args.run_name or f"student_seed{args.seed}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, f"{run_name}.pt")
    img_dir = src["images_dir"]

    tr_df = cohort[(cohort["split"] == "train") & cohort["healthy"]].reset_index(drop=True)
    va_df = cohort[(cohort["split"] == "val") & cohort["healthy"]].reset_index(drop=True)
    # Everything scored but never trained on: the test partition (all rows) plus
    # the non-healthy rows of the train/val partitions.
    sc_df = cohort[(cohort["split"] == "test") | ~cohort["healthy"]].reset_index(drop=True)
    sc_df["cohort"] = np.where(sc_df["healthy"], "healthy", "nonhealthy")

    gpu_aug = args.gpu_aug if args.gpu_aug is not None else (device.type == "cuda")
    mk = lambda df, sp, d=img_dir: MBRSETDataset(csv=df, images_dir=d, task="age", split=sp,
                                                 image_size=args.image_size, drop_missing_files=True,
                                                 fov_crop=True, device_aug=(gpu_aug and sp == "train"))
    dev_aug = DeviceAug() if gpu_aug else None
    train_ds, val_ds, sc_ds = mk(tr_df, "train"), mk(va_df, "val"), mk(sc_df, "val")
    ext_ds = mk(ext_df, "val", ext["images_dir"]) if ext_df is not None else None
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"[fatal] empty train ({len(train_ds)}) or val ({len(val_ds)}) after "
                         f"dropping missing files; check --root / --image-ext.")

    # Standardise the target on the TRAIN ages: the head then starts near the
    # mean instead of having to climb from 0 to ~55 years.
    ages = train_ds.labels.numpy().astype(np.float64)
    mean, std = float(ages.mean()), float(max(ages.std(), 1e-6))

    print(f"[info] device : {device}")
    print(f"[info] train  : {args.dataset} @ {args.root}   healthy={args.healthy}"
          + ("  +exclude-pathology" if args.exclude_pathology else ""))
    print(f"[info] splits : train={len(train_ds)} val={len(val_ds)} (healthy)  "
          f"scored={len(sc_ds)} (test all + non-healthy)"
          + (f"  external={len(ext_ds)}" if ext_ds is not None else ""))
    print(f"[info] target : age mean={mean:.1f} sd={std:.1f} (standardised for the head)")

    g = torch.Generator(); g.manual_seed(args.seed)
    sampler = None
    if args.age_balance:
        b = age_bin(ages)
        freq = np.bincount(b, minlength=len(AGE_BIN_LABELS)).astype(np.float64)
        w = 1.0 / np.sqrt(np.maximum(freq[b], 1.0))
        sampler = WeightedRandomSampler(torch.as_tensor(w / w.sum(), dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True, generator=g)
    dl = lambda ds, **kw: DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, generator=g,
                                     pin_memory=(device.type == "cuda"),
                                     persistent_workers=(args.num_workers > 0), **kw)
    train_loader = dl(train_ds, sampler=sampler, shuffle=(sampler is None), drop_last=True)
    val_loader, sc_loader = dl(val_ds, shuffle=False), dl(sc_ds, shuffle=False)
    ext_loader = dl(ext_ds, shuffle=False) if ext_ds is not None else None

    model = MBRSETClassifier(num_classes=None, regression=True, pretrained=not args.no_pretrained,
                             use_gcg=False, dropout=args.dropout, backbone=args.backbone,
                             backbone_kwargs=backbone_kwargs_for(args.backbone, args.image_size)
                             ).to(device)
    use_cl = device.type == "cuda" and args.nondeterministic
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[info] model  : {args.backbone}, {n_params/1e6:.3f}M params, regression head")

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    use_amp = args.amp if args.amp is not None else (device.type == "cuda")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    print(f"[info] amp    : {('on(' + str(amp_dtype).split('.')[-1] + ')') if use_amp else 'off'}  "
          f"gpu_aug={'on' if gpu_aug else 'off'}  loss={args.loss}  "
          f"age_balance={'on' if args.age_balance else 'off'}")

    def crit(pred_z: torch.Tensor, y_age: torch.Tensor) -> torch.Tensor:
        z = (y_age.float() - mean) / std
        pz = pred_z.float()
        if args.loss == "l1":
            return F.l1_loss(pz, z)
        if args.loss == "huber":
            return F.smooth_l1_loss(pz, z, beta=args.huber_delta)
        return F.mse_loss(pz, z)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = max(0, min(args.warmup_epochs, args.epochs - 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warm))
    sched = (torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm), cos],
        milestones=[warm]) if warm > 0 else cos)
    csv_log = CSVLogger(os.path.join(args.ckpt_dir, f"{run_name}_metrics.csv"))

    def eval_split(m, loader, ds, frame):
        sf = score_frame(m, ds, frame, loader, device, mean, std)
        return regression_metrics(sf["age"], sf["pred_age"], sf["patient"]), sf

    best_mae, best_epoch, since = float("inf"), -1, 0
    print(f"\n=== training up to {args.epochs} epochs (healthy val MAE selects best) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb, t_ep = 0.0, 0, time.time()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if dev_aug is not None:
                x = dev_aug(x)
            if use_cl:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
            loss = crit(out, y)                     # fp32 outside autocast
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            run = run + loss.detach(); nb += 1
            if args.log_every and nb % args.log_every == 0:
                el = time.time() - t_ep
                print(f"    step {nb}/{len(train_loader)}  loss={float(run)/nb:.4f}  "
                      f"{nb*args.batch_size/el:.0f} img/s  ({el:.0f}s)", flush=True)
        run = float(run) / max(nb, 1)
        eval_model = ema.ema if ema is not None else model
        vm, _ = eval_split(eval_model, val_loader, val_ds, va_df)
        tag = ""
        if vm["mae"] < best_mae:
            best_mae, best_epoch, since = vm["mae"], epoch, 0
            torch.save({"model": eval_model.state_dict(), "epoch": epoch, "val": vm,
                        "use_gcg": False, "task": TASK, "regression": True, "num_classes": None,
                        "target_norm": {"mean": mean, "std": std},
                        "selection_metric": "mae", "args": vars(args)}, ckpt)
            tag = "  <- best"
        else:
            since += 1
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={run:.4f}  val_MAE={vm['mae']:.2f}y  "
              f"r={vm['r']:.3f}  gap={vm['mean_gap']:+.2f}  [{time.time()-t_ep:.0f}s]{tag}", flush=True)
        csv_log.log({"epoch": epoch, "train_loss": round(run, 5),
                     **{f"val_{k}": round(v, 5) for k, v in vm.items()
                        if isinstance(v, float) and k != "n"}})
        sched.step()
        if args.patience and since >= args.patience:
            print(f"  early stop: no val MAE gain for {args.patience} epochs")
            break
    csv_log.close()
    print(f"[info] selection: best val MAE={best_mae:.3f}y @ epoch {best_epoch}")

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])

    # --- bias correction from the healthy VAL predictions of the best model ---
    vm, vf = eval_split(model, val_loader, val_ds, va_df)
    fit = bias_fit(vf["age"], vf["gap"])
    fit["fit_on"] = "val_healthy"
    _with_corrected(vm, vf["age"], vf["pred_age"], fit)
    print(f"\n=== VAL (healthy, best checkpoint) ===  MAE={vm['mae']:.2f}y  r={vm['r']:.3f}  "
          f"gap-vs-age slope={vm['gap_age_slope']:+.3f}/y")
    print(f"  bias correction: gap = {fit['a']:+.2f} {fit['b']:+.4f}*age  (n={fit['n']}); "
          f"corrected gap is what disease analyses should use")

    # --- BRSET: test partition (healthy + non-healthy) and never-trained non-healthy ---
    sc = score_frame(model, sc_ds, sc_df, sc_loader, device, mean, std)
    sc["gap_corrected"] = bias_apply(sc["age"], sc["gap"], fit)
    sub = {
        "test_healthy": sc[(sc["split"] == "test") & (sc["cohort"] == "healthy")],
        "test_all": sc[sc["split"] == "test"],
        "test_nonhealthy": sc[(sc["split"] == "test") & (sc["cohort"] == "nonhealthy")],
        "unseen_nonhealthy": sc[(sc["split"] != "test") & (sc["cohort"] == "nonhealthy")],
    }
    res_sets = {}
    for k, f in sub.items():
        res_sets[k] = _with_corrected(regression_metrics(f["age"], f["pred_age"], f["patient"]),
                                      f["age"], f["pred_age"], fit)
    th = res_sets["test_healthy"]
    print(f"\n=== TEST healthy (in-domain, held-out patients, n={th['n']}, "
          f"{th['n_patients']} patients) ===")
    print(f"  MAE={th['mae']:.2f}y  RMSE={th['rmse']:.2f}  r={th['r']:.3f}  R2={th['r2']:.3f}  "
          f"patient-level MAE={th['patient_mae']:.2f}y  mean gap={th['mean_gap']:+.2f} "
          f"(corrected {th['mean_gap_corrected']:+.2f})")
    print(f"  MAE by age bin: {fmt_bins(th['by_bin'])}")
    for k in ("test_all", "test_nonhealthy", "unseen_nonhealthy"):
        m = res_sets[k]
        if m["n"]:
            print(f"  {k:<18} n={m['n']:<6} MAE={m['mae']:.2f}y  r={m['r']:.3f}  "
                  f"gap={m['mean_gap']:+.2f}  corrected gap={m['mean_gap_corrected']:+.2f}")
    if res_sets["test_nonhealthy"]["n"] and th["n"]:
        d = res_sets["test_nonhealthy"]["mean_gap_corrected"] - th["mean_gap_corrected"]
        print(f"  non-healthy minus healthy corrected gap (test): {d:+.2f}y  "
              f"(the retinal-age-gap signal; one seed, descriptive)")

    # --- external (mBRSET), zero-shot ---
    em, em_by_dr, ef = None, None, None
    if ext_loader is not None:
        ef = score_frame(model, ext_ds, ext_df, ext_loader, device, mean, std)
        ef["gap_corrected"] = bias_apply(ef["age"], ef["gap"], fit)
        em = _with_corrected(regression_metrics(ef["age"], ef["pred_age"], ef["patient"]),
                             ef["age"], ef["pred_age"], fit)
        em_by_dr = _by_dr(ef, fit)
        _print_external(f"EXTERNAL {args.external_test_dataset} (zero-shot)", em, em_by_dr, th)

    result = {"task": TASK, "seed": args.seed, "backbone": args.backbone,
              "train_dataset": args.dataset, "healthy": args.healthy,
              "exclude_pathology": args.exclude_pathology,
              "cohort": {"n_images": int(len(cohort)), "n_patients": int(cohort["patient"].nunique()),
                         "n_healthy_images": int(cohort["healthy"].sum()),
                         "n_healthy_patients": int(cohort.loc[cohort["healthy"], "patient"].nunique()),
                         "exclusions": {k: int(v) for k, v in
                                        cohort.loc[~cohort["healthy"], "exclusion"].value_counts().items()}},
              "n_train": len(train_ds), "n_val": len(val_ds), "n_scored": len(sc_ds),
              "target_norm": {"mean": mean, "std": std},
              "best_val_mae": best_mae, "best_epoch": best_epoch, "selection_metric": "mae",
              "val": vm, "bias_correction": fit, **res_sets,
              "external_dataset": args.external_test_dataset if em else None,
              "external": em, "external_by_dr": em_by_dr,
              "n_external": len(ext_ds) if ext_ds is not None else 0,
              "external_bnadapt": None, "external_bnadapt_by_dr": None,
              "domain_gap_mae": (em["mae"] - th["mae"]) if em else None,
              "amp": bool(use_amp), "params_m": round(n_params / 1e6, 4),
              "predictions_csv": os.path.abspath(os.path.join(args.ckpt_dir, f"{run_name}_predictions.csv")),
              "args": vars(args)}

    def write_results():
        if args.results_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[info] wrote {args.results_json}")

    def write_predictions(ext_frame):
        frames = [_pred_rows(sc, args.dataset)]
        if ext_frame is not None:
            e = ext_frame.copy(); e["split"] = "external"; e["cohort"] = "external"
            frames.append(_pred_rows(e, args.external_test_dataset))
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(result["predictions_csv"], index=False)
        print(f"[info] wrote {result['predictions_csv']} ({len(out)} rows)")

    # Written BEFORE the optional BN adaptation: a crash there must not lose the
    # zero-shot numbers and the predictions table.
    ck = torch.load(ckpt, map_location="cpu")
    ck["bias_correction"] = fit
    torch.save(ck, ckpt)
    write_results()
    write_predictions(ef)

    if ext_loader is not None and args.bn_adapt:
        g_bn = torch.Generator(); g_bn.manual_seed(args.seed)
        bn_loader = DataLoader(ext_ds, batch_size=args.batch_size, shuffle=True, generator=g_bn,
                               num_workers=args.num_workers, worker_init_fn=seed_worker,
                               pin_memory=(device.type == "cuda"))
        adapted, n_bn = adapt_bn(model, bn_loader, device, max_batches=args.bn_adapt_batches)
        result["bn_layers"] = n_bn
        if n_bn == 0:
            print(f"\n[warn] no BatchNorm layers in {args.backbone}; --bn-adapt is a no-op.")
        else:
            ea = score_frame(adapted, ext_ds, ext_df, ext_loader, device, mean, std)
            ea["gap_corrected"] = bias_apply(ea["age"], ea["gap"], fit)
            ema_m = _with_corrected(regression_metrics(ea["age"], ea["pred_age"], ea["patient"]),
                                    ea["age"], ea["pred_age"], fit)
            ema_by_dr = _by_dr(ea, fit)
            _print_external(f"EXTERNAL {args.external_test_dataset} after AdaBN "
                            f"({n_bn} BN layers, label-free, transductive)", ema_m, ema_by_dr, th)
            print(f"  BN-adapt effect on MAE: {em['mae']:.2f} -> {ema_m['mae']:.2f}y  "
                  f"delta={ema_m['mae'] - em['mae']:+.2f}  (paired within this run)")
            ef = ef.merge(ea[["file", "pred_age", "gap", "gap_corrected"]]
                          .rename(columns={"pred_age": "pred_age_bnadapt", "gap": "gap_bnadapt",
                                           "gap_corrected": "gap_bnadapt_corrected"}),
                          on="file", how="left")
            ck = torch.load(ckpt, map_location="cpu")
            ck["model_bnadapt"] = {k: v.detach().cpu() for k, v in adapted.state_dict().items()}
            ck["bn_adapt"] = {"dataset": args.external_test_dataset, "root": args.external_test_root,
                              "batches": args.bn_adapt_batches or "all", "bn_layers": n_bn,
                              "transductive": True, "external_bnadapt": ema_m}
            torch.save(ck, ckpt)
            result.update({"external_bnadapt": ema_m, "external_bnadapt_by_dr": ema_by_dr,
                           "bn_adapt_transductive": True})
            write_results()
            write_predictions(ef)

    done = os.path.join(args.ckpt_dir, f"{run_name}.done")
    with open(done, "w") as f:
        f.write((os.path.abspath(args.results_json) if args.results_json
                 else datetime.datetime.now(datetime.timezone.utc).isoformat()) + "\n")
    print(f"[info] wrote {done}")
    return 0


def _by_dr(frame: pd.DataFrame, fit) -> Optional[Dict[str, Dict[str, object]]]:
    """Metrics per DR stratum of an external frame (grade 0 / 1 / referable >=2)."""
    if "dr_grade" not in frame.columns or frame["dr_grade"].isna().all():
        return None
    groups = {"dr0": frame["dr_grade"] == 0, "dr1": frame["dr_grade"] == 1,
              "referable": frame["dr_grade"] >= 2}
    out = {}
    for k, m in groups.items():
        f = frame[m]
        out[k] = _with_corrected(regression_metrics(f["age"], f["pred_age"], f["patient"]),
                                 f["age"], f["pred_age"], fit)
    return out


def _print_external(title: str, em: Dict[str, object], by_dr, th: Dict[str, object]) -> None:
    print(f"\n=== {title}, n={em['n']}, {em['n_patients']} patients ===")
    print(f"  MAE={em['mae']:.2f}y  RMSE={em['rmse']:.2f}  r={em['r']:.3f}  "
          f"patient-level MAE={em['patient_mae']:.2f}y  mean gap={em['mean_gap']:+.2f} "
          f"(corrected {em['mean_gap_corrected']:+.2f})")
    print(f"  MAE by age bin: {fmt_bins(em['by_bin'])}")
    if th["n"]:
        print(f"  domain gap (external minus in-domain healthy MAE): {em['mae'] - th['mae']:+.2f}y")
    if by_dr:
        print("  by DR grade (all diabetic; corrected gap is the within-device signal):")
        for k, m in by_dr.items():
            if m["n"]:
                print(f"    {k:<10} n={m['n']:<5} MAE={m['mae']:.2f}y  gap={m['mean_gap']:+.2f}  "
                      f"corrected={m['mean_gap_corrected']:+.2f}")


def _pred_rows(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cols = ["file", "patient", "split", "cohort", "age", "pred_age", "gap", "gap_corrected"]
    opt = [c for c in META_COLS if c in frame.columns]
    opt += [c for c in ("pred_age_bnadapt", "gap_bnadapt", "gap_bnadapt_corrected")
            if c in frame.columns]
    out = frame[cols + opt].copy()
    out.insert(0, "dataset", dataset)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
