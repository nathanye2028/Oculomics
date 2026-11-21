#!/usr/bin/env python3
"""
metadata_model.py
=================
Predict each disease target from **clinical metadata alone** -- no pixels.

This is the comparator the capability gate is built on: a target only earns an
image head if the image model beats what the intake form already knows, on the
same patients. ``covariate_baseline.py`` answers the narrow version (age+sex,
one logistic regression, inside a training run); this script answers the broad
one, per target, before any GPU time is spent:

    python metadata_model.py --dataset mbrset --root <mBRSET root> --out exp_metadata/mbrset
    python metadata_model.py --dataset brset  --root <BRSET root>  --out exp_metadata/brset \\
        --external-root <mBRSET root> --external-dataset mbrset

Feature sets (nested, so the table reads left to right as "more chart"):

    age_sex    age, sex                                   -- what the photo itself encodes
    clinical   + diabetes duration / insulin / oral Rx    -- a screening intake form
               (+ BRSET's diabetes yes/no, nationality)
    full       + every other comorbidity / lifestyle field -- the whole chart
               (mBRSET only: hypertension, nephropathy, smoking, education, ...)

A feature set never contains the target's own column or its direct proxies
(:data:`PROXIES`), and never contains image-derived columns (ICDR grade, edema,
quality, artifacts, laterality, camera): those are what the image model is
supposed to supply, and camera is an acquisition variable, not a clinical one.

Two models per (target, feature set): a standardised logistic regression and a
shallow histogram gradient-boosted tree. The gate should be read against the
**stronger** of the two -- a baseline the image model can beat only because it
was linear is not a baseline.

What is reported per (target, feature set, model)
--------------------------------------------------
* ``test``  -- the same patient-grouped split the image trainer uses
  (``stratified_split``, val 0.10 / test 0.20, ``--split-seed`` 42), so a
  metadata AUROC and an image AUROC on that seed are **paired** on identical
  test patients. Fit on train+val (the metadata model needs no early stopping;
  using val too makes the baseline stronger, i.e. the gate stricter).
  AUROC + AUPRC with a **patient-cluster** bootstrap 95% CI, and sensitivity /
  specificity / PPV / flagged fraction at a threshold fixed on out-of-fold
  training predictions to hit ``--target-sens`` -- never AUROC alone.
* ``cv``    -- repeated patient-grouped 5-fold CV over the whole dataset
  (``--cv-repeats`` x 5 folds): mean and SD of AUROC. The SD is the noise
  floor a single-split delta has to clear.
* ``shuffle`` -- negative control: training labels permuted ``--n-shuffle`` times,
  scored on the real test labels. The mean must sit near 0.5 (flagged when it is
  more than 3 SE away -- something leaks).
* ``external`` -- (optional) the model fit on this dataset, scored on another
  (BRSET -> mBRSET) using only the features both datasets carry.

A target whose test split has fewer than ``--min-test-pos-patients`` positive
patients is flagged ``underpowered``: its AUROC is reported but should not be
headlined (nephropathy / neuropathy / MI are ~4-8% prevalent in mBRSET).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LABEL_REGISTRY, SYSTEMIC_TASKS, _binary_flag, _isnan_vector, stratified_split  # noqa: E402

# --------------------------------------------------------------------------- #
# Features and targets
# --------------------------------------------------------------------------- #
# Columns in the mBRSET schema (BRSET is renamed into it by brset_dataset.load_brset;
# ``diabetes`` and ``nationality`` are BRSET-only and carried through by load_table).
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "age_sex": ("age", "sex"),
    "clinical": ("age", "sex", "dm_time", "insulin", "insulin_time", "oraltreatment_dm",
                 "diabetes", "nationality"),
    "full": ("age", "sex", "dm_time", "insulin", "insulin_time", "oraltreatment_dm",
             "diabetes", "nationality",
             "educational_level", "insurance", "alcohol_consumption", "smoking", "obesity",
             "systemic_hypertension", "nephropathy", "neuropathy",
             "acute_myocardial_infarction", "vascular_disease", "diabetic_foot"),
}

# Target -> extra columns that restate it and must never be a feature for it.
# (The target's own source column is always excluded; these are the proxies.)
PROXIES: Dict[str, Tuple[str, ...]] = {
    "insulin": ("insulin_time",),
}

OCULAR_TARGETS = ("dr_referable", "dr_binary", "edema")
DEFAULT_TARGETS = OCULAR_TARGETS + tuple(SYSTEMIC_TASKS)

# BRSET columns load_brset drops (it keeps only the mBRSET-schema ones) but that
# are clinical metadata, not image labels.
BRSET_EXTRA_METADATA = ("diabetes", "nationality")


def load_table(root: str, dataset: str) -> Tuple[pd.DataFrame, str]:
    """Label CSV in the mBRSET schema, plus BRSET's extra clinical columns."""
    from brset_dataset import load_any
    src = load_any(root, dataset)
    df = src["df"]
    if dataset == "brset":
        raw = pd.read_csv(src["csv"])
        for c in BRSET_EXTRA_METADATA:
            if c in raw.columns and c not in df.columns and len(raw) == len(df):
                df[c] = raw[c].to_numpy()
    return df, src["csv"]


def target_vector(df: pd.DataFrame, task: str) -> np.ndarray:
    spec = LABEL_REGISTRY[task]
    if any(c not in df.columns for c in spec.source_cols):
        return np.full(len(df), np.nan)
    return np.array([spec.fn(dict(zip(spec.source_cols, row)))
                     for row in df[list(spec.source_cols)].to_numpy()], dtype=np.float64)


def features_for(task: str, feature_set: str, columns: Sequence[str]) -> List[str]:
    """Columns of ``feature_set`` present in ``columns``, minus the target and its proxies."""
    banned = set(LABEL_REGISTRY[task].source_cols) | set(PROXIES.get(task, ()))
    return [c for c in FEATURE_SETS[feature_set] if c in columns and c not in banned]


def _column_kind(s: pd.Series) -> str:
    """'binary' if every non-null cell is a recognised yes/no token, else
    'numeric' if it parses as a number, else 'categorical'."""
    vals = s.dropna()
    if len(vals) == 0:
        return "empty"
    if all(not np.isnan(_binary_flag(v)) for v in pd.unique(vals)):
        return "binary"
    if pd.to_numeric(vals, errors="coerce").notna().all():
        return "numeric"
    return "categorical"


def encode(df: pd.DataFrame, features: Sequence[str],
           kinds: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Numeric/binary columns -> float (NaN kept); categorical -> normalised str (NaN kept).

    ``kinds`` fixes the per-column decision from the training table so an
    external table is encoded the same way even if its values look different.
    """
    kinds = dict(kinds) if kinds else {c: _column_kind(df[c]) for c in features}
    out = {}
    for c in features:
        s = df[c]
        k = kinds[c]
        if k == "binary":
            out[c] = s.map(_binary_flag).astype(float)
        elif k in ("numeric", "empty"):
            out[c] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            out[c] = s.astype(str).str.strip().str.lower().where(s.notna())
    return pd.DataFrame(out, index=df.index), kinds


def build_model(kind: str, kinds: Dict[str, str], seed: int = 0):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    num = [c for c, k in kinds.items() if k in ("binary", "numeric")]
    cat = [c for c, k in kinds.items() if k == "categorical"]
    if kind == "logreg":
        ct = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True),
                                  StandardScaler()), num),
            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                  OneHotEncoder(handle_unknown="ignore")), cat),
        ])
        return make_pipeline(ct, LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=seed))
    if kind == "gbm":
        ct = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat),
        ])
        mask = [False] * len(num) + [True] * len(cat)
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
            l2_regularization=1.0, class_weight="balanced", early_stopping=False,
            categorical_features=mask if cat else None, random_state=seed)
        return make_pipeline(ct, clf)
    raise ValueError(f"unknown model {kind!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      n_boot: int = 1000, seed: int = 0) -> Dict[str, List[float]]:
    """95% CI for AUROC/AUPRC resampling *patients* (both eyes move together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    aucs, aps = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(_auc(y[idx], p[idx]))
        aps.append(_ap(y[idx], p[idx]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else [float("nan")] * 2
    return {"auroc_ci": ci(aucs), "auprc_ci": ci(aps)}


def threshold_at_sensitivity(y: np.ndarray, p: np.ndarray, target_sens: float) -> float:
    """Highest threshold whose sensitivity on (y, p) is >= target_sens."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return float("nan")
    k = int(np.floor((1.0 - target_sens) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    flag = p >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"threshold": thr, "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "flagged_fraction": div(tp + fp, len(y))}


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _xy(df: pd.DataFrame, task: str, features: Sequence[str],
        kinds: Optional[Dict[str, str]] = None):
    y = target_vector(df, task)
    keep = ~_isnan_vector(y)
    X, kinds = encode(df.loc[keep], features, kinds)
    groups = df.loc[keep, "patient"].to_numpy() if "patient" in df.columns else np.arange(keep.sum())
    return X, y[keep].astype(int), groups, kinds


def _oof_scores(model_kind: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                kinds: Dict[str, str], seed: int, n_splits: int = 5) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold
    oof = np.full(len(y), np.nan)
    n_splits = min(n_splits, int(min(np.bincount(y))))
    if n_splits < 2:
        return oof
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = build_model(model_kind, kinds, seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def repeated_cv(model_kind: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                kinds: Dict[str, str], repeats: int, seed: int) -> Dict[str, float]:
    from sklearn.model_selection import StratifiedGroupKFold
    aucs = []
    n_splits = min(5, int(min(np.bincount(y))))
    if n_splits < 2:
        return {"auroc_mean": float("nan"), "auroc_sd": float("nan"), "n_folds": 0}
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + 1000 + r)
        for tr, te in sgkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = build_model(model_kind, kinds, seed).fit(X.iloc[tr], y[tr])
            aucs.append(_auc(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return {"auroc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auroc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan"),
            "n_folds": len(aucs)}


def top_coefficients(model, k: int = 5) -> List[Tuple[str, float]]:
    """Largest-|coef| standardised logistic-regression terms, for reading what
    the metadata model leans on. Empty for the tree model."""
    clf = model[-1]
    if not hasattr(clf, "coef_"):
        return []
    names = model[0].get_feature_names_out()
    coef = clf.coef_.ravel()
    order = np.argsort(-np.abs(coef))[:k]
    return [(str(names[i]).split("__", 1)[-1], float(coef[i])) for i in order]


def evaluate_target(df: pd.DataFrame, task: str, feature_set: str, model_kind: str, *,
                    split_seed: int = 42, seed: int = 0, target_sens: float = 0.85,
                    n_boot: int = 1000, cv_repeats: int = 5, min_test_pos_patients: int = 20,
                    n_shuffle: int = 20,
                    external: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    feats = features_for(task, feature_set, df.columns)
    row: Dict[str, object] = {"task": task, "feature_set": feature_set, "model": model_kind,
                              "features": feats, "reason": None}
    if not feats:
        row["reason"] = "no features of this set present"
        return row
    y_all = target_vector(df, task)
    keep = ~_isnan_vector(y_all)
    if keep.sum() == 0 or len(np.unique(y_all[keep])) < 2:
        row["reason"] = "target missing or single-class"
        return row

    sp = stratified_split(df, task=task, val_frac=0.10, test_frac=0.20,
                          group_col="patient", seed=split_seed)
    fit_df = pd.concat([sp["train"], sp["val"]], ignore_index=True)
    X_tr, y_tr, g_tr, kinds = _xy(fit_df, task, feats)
    X_te, y_te, g_te, _ = _xy(sp["test"], task, feats, kinds)
    row.update(n_train=int(len(y_tr)), n_test=int(len(y_te)),
               prevalence=float(np.mean(np.concatenate([y_tr, y_te]))),
               test_pos=int(y_te.sum()), test_pos_patients=int(len(np.unique(g_te[y_te == 1]))),
               kinds=kinds)
    row["underpowered"] = row["test_pos_patients"] < min_test_pos_patients
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        row["reason"] = "single-class train or test split"
        return row

    model = build_model(model_kind, kinds, seed).fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    oof = _oof_scores(model_kind, X_tr, y_tr, g_tr, kinds, seed)
    ok = ~np.isnan(oof)
    thr = threshold_at_sensitivity(y_tr[ok], oof[ok], target_sens)
    row["test"] = {"auroc": _auc(y_te, p_te), "auprc": _ap(y_te, p_te),
                   **cluster_bootstrap(y_te, p_te, g_te, n_boot, seed),
                   "at_train_threshold": operating_point(y_te, p_te, thr),
                   "target_sensitivity": target_sens}
    row["top_coefficients"] = top_coefficients(model)

    # Negative control: same pipeline, training labels permuted. A single shuffle
    # is useless here -- a linear model on a strong feature (age) still ranks by it,
    # with a random sign, so each shuffle lands near AUROC or 1-AUROC. Report the
    # mean and its SE: expected 0.5, and a pipeline leak drags it toward the real
    # AUROC. (A permutation p-value is NOT reported: for the same sign reason about
    # half the shuffles tie the real AUROC, so p ~ 0.5 regardless of signal.)
    rng = np.random.default_rng(seed)
    shuf = np.array([_auc(y_te, build_model(model_kind, kinds, seed)
                          .fit(X_tr, rng.permutation(y_tr)).predict_proba(X_te)[:, 1])
                     for _ in range(n_shuffle)])
    row["shuffle_auroc"] = float(np.mean(shuf))
    row["shuffle_auroc_se"] = float(np.std(shuf, ddof=1) / np.sqrt(len(shuf))) if len(shuf) > 1 else float("nan")

    X_all, y_cv, g_all, _ = _xy(df, task, feats, kinds)
    row["cv"] = repeated_cv(model_kind, X_all, y_cv, g_all, kinds, cv_repeats, seed)

    if external is not None:
        row["external"] = score_external(df, external, task, feature_set, model_kind,
                                         seed=seed, n_boot=n_boot, thr_sens=target_sens)
    return row


def score_external(src: pd.DataFrame, ext: pd.DataFrame, task: str, feature_set: str,
                   model_kind: str, *, seed: int = 0, n_boot: int = 1000,
                   thr_sens: float = 0.85) -> Dict[str, object]:
    """Fit on all of ``src`` with the features both tables carry; score all of ``ext``.

    The threshold is fixed on ``src`` out-of-fold predictions and applied unchanged --
    the same threshold-transfer protocol as the image model.
    """
    feats = [c for c in features_for(task, feature_set, src.columns) if c in ext.columns]
    out: Dict[str, object] = {"features": feats, "reason": None}
    if not feats:
        out["reason"] = "no shared features"
        return out
    X_s, y_s, g_s, kinds = _xy(src, task, feats)
    X_e, y_e, g_e, _ = _xy(ext, task, feats, kinds)
    if len(np.unique(y_s)) < 2 or len(np.unique(y_e)) < 2:
        out["reason"] = "single-class source or external"
        return out
    m = build_model(model_kind, kinds, seed).fit(X_s, y_s)
    p = m.predict_proba(X_e)[:, 1]
    oof = _oof_scores(model_kind, X_s, y_s, g_s, kinds, seed)
    ok = ~np.isnan(oof)
    thr = threshold_at_sensitivity(y_s[ok], oof[ok], thr_sens)
    out.update(n=int(len(y_e)), pos=int(y_e.sum()), auroc=_auc(y_e, p), auprc=_ap(y_e, p),
               **cluster_bootstrap(y_e, p, g_e, n_boot, seed),
               at_source_threshold=operating_point(y_e, p, thr))
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def flat_rows(results: List[Dict[str, object]]) -> pd.DataFrame:
    recs = []
    for r in results:
        t, cv, ext = r.get("test") or {}, r.get("cv") or {}, r.get("external") or {}
        op = t.get("at_train_threshold") or {}
        recs.append({
            "task": r["task"], "feature_set": r["feature_set"], "model": r["model"],
            "n_features": len(r["features"]), "prevalence": r.get("prevalence"),
            "test_pos_patients": r.get("test_pos_patients"), "underpowered": r.get("underpowered"),
            "test_auroc": t.get("auroc"),
            "test_auroc_lo": (t.get("auroc_ci") or [None, None])[0],
            "test_auroc_hi": (t.get("auroc_ci") or [None, None])[1],
            "test_auprc": t.get("auprc"),
            "sens@thr": op.get("sensitivity"), "spec@thr": op.get("specificity"),
            "ppv@thr": op.get("ppv"), "flagged@thr": op.get("flagged_fraction"),
            "cv_auroc_mean": cv.get("auroc_mean"), "cv_auroc_sd": cv.get("auroc_sd"),
            "shuffle_auroc": r.get("shuffle_auroc"), "shuffle_se": r.get("shuffle_auroc_se"),
            "ext_auroc": ext.get("auroc"),
            "ext_sens@thr": (ext.get("at_source_threshold") or {}).get("sensitivity"),
            "reason": r.get("reason"),
        })
    return pd.DataFrame(recs)


def best_per_task(table: pd.DataFrame) -> pd.DataFrame:
    """The gate comparator: for each task, the strongest metadata model by CV AUROC
    (CV, not test, so choosing the comparator does not peek at the held-out split)."""
    t = table.dropna(subset=["cv_auroc_mean"])
    if t.empty:
        return t
    idx = t.groupby("task")["cv_auroc_mean"].idxmax()
    return t.loc[idx].sort_values("cv_auroc_mean", ascending=False).reset_index(drop=True)


def _fmt(table: pd.DataFrame, cols: Sequence[str]) -> str:
    return table[list(cols)].to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Metadata-only disease prediction (no images).")
    p.add_argument("--dataset", choices=["mbrset", "brset"], required=True)
    p.add_argument("--root", required=True, help="dataset root (dir holding the label CSV)")
    p.add_argument("--out", default="exp_metadata", help="output directory")
    p.add_argument("--tasks", nargs="+", default=None,
                   help=f"targets (default: every one the CSV carries of {list(DEFAULT_TARGETS)})")
    p.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    p.add_argument("--models", nargs="+", default=["logreg", "gbm"], choices=["logreg", "gbm"])
    p.add_argument("--split-seed", type=int, default=42,
                   help="patient split seed; keep equal to the image run's to pair the test sets")
    p.add_argument("--seed", type=int, default=0, help="model / bootstrap / shuffle seed")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--cv-repeats", type=int, default=5)
    p.add_argument("--min-test-pos-patients", type=int, default=20)
    p.add_argument("--n-shuffle", type=int, default=20, help="label permutations for the negative control")
    p.add_argument("--external-root", default=None)
    p.add_argument("--external-dataset", choices=["mbrset", "brset"], default=None)
    a = p.parse_args(argv)

    df, csv = load_table(a.root, a.dataset)
    ext = None
    if a.external_root:
        if not a.external_dataset:
            p.error("--external-root needs --external-dataset")
        ext, _ = load_table(a.external_root, a.external_dataset)
    tasks = a.tasks or [t for t in DEFAULT_TARGETS
                        if all(c in df.columns for c in LABEL_REGISTRY[t].source_cols)]
    print(f"{a.dataset}: {csv}  ({len(df)} rows, {df['patient'].nunique()} patients)")
    print(f"tasks: {tasks}")
    for fs in a.feature_sets:
        present = [c for c in FEATURE_SETS[fs] if c in df.columns]
        print(f"  {fs:<9} features present: {present}")

    results = []
    for task in tasks:
        for fs in a.feature_sets:
            for mk in a.models:
                r = evaluate_target(df, task, fs, mk, split_seed=a.split_seed, seed=a.seed,
                                    target_sens=a.target_sens, n_boot=a.n_boot,
                                    cv_repeats=a.cv_repeats,
                                    min_test_pos_patients=a.min_test_pos_patients,
                                    n_shuffle=a.n_shuffle,
                                    external=ext if task in OCULAR_TARGETS else None)
                results.append(r)
                t = r.get("test") or {}
                print(f"  {task:<22} {fs:<9} {mk:<6} test AUROC "
                      f"{t.get('auroc', float('nan')):.3f}  cv "
                      f"{(r.get('cv') or {}).get('auroc_mean', float('nan')):.3f}"
                      f"{'  [' + r['reason'] + ']' if r.get('reason') else ''}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    table = flat_rows(results)
    best = best_per_task(table)
    table.to_csv(os.path.join(a.out, "metadata_results.csv"), index=False)
    best.to_csv(os.path.join(a.out, "metadata_best_per_task.csv"), index=False)
    with open(os.path.join(a.out, "metadata_results.json"), "w") as f:
        json.dump({"dataset": a.dataset, "csv": csv, "args": vars(a), "results": results},
                  f, indent=2, default=float)

    print(f"\n=== STRONGEST METADATA MODEL PER TARGET ({a.dataset}; chosen by CV AUROC) ===")
    print(_fmt(best, ["task", "feature_set", "model", "prevalence", "test_pos_patients",
                      "test_auroc", "test_auroc_lo", "test_auroc_hi", "cv_auroc_mean",
                      "cv_auroc_sd", "sens@thr", "spec@thr", "ppv@thr", "shuffle_auroc",
                      "underpowered"]))
    if ext is not None:
        e = table.dropna(subset=["ext_auroc"])
        if not e.empty:
            print(f"\n=== EXTERNAL ({a.dataset} -> {a.external_dataset}, shared features only) ===")
            print(_fmt(e, ["task", "feature_set", "model", "test_auroc", "ext_auroc", "ext_sens@thr"]))
    bad = table[(table["shuffle_auroc"] - 0.5).abs() > 3 * table["shuffle_se"]].dropna(subset=["shuffle_auroc"])
    if not bad.empty:
        print(f"\nWARNING: shuffled-label control far from 0.5 for "
              f"{sorted(set(bad['task']))} -- check for leakage before trusting these rows.")
    print(f"\nwrote {a.out}/metadata_results.{{csv,json}} and metadata_best_per_task.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
