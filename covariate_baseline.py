"""
covariate_baseline.py
=====================
The age+sex logistic-regression baseline every systemic (oculomics) result must
be read against.

Why it exists
-------------
Every systemic target in mBRSET (hypertension, nephropathy, myocardial
infarction, ...) is strongly associated with age and diabetes duration in a
diabetic cohort, and a fundus photograph encodes age well (Poplin et al. 2018
predicted age to ~3.3 years MAE). An image model can therefore reach a
respectable AUROC on "nephropathy" by learning age and nothing about the kidney.
The honest comparison is: does the image model beat a logistic regression on
the tabular covariates it could have inferred? ``train_mbrset.py
--covariate-baseline`` fits this on the SAME patient-grouped train split and
scores it on the SAME test split, so the per-seed difference is paired.

    from covariate_baseline import covariate_baseline
    cb = covariate_baseline(splits["train"], splits["test"], task="hypertension")
    cb["auroc"]      # NaN if the task is not binary, a covariate is missing,
                     # or the test split is single-class -- with cb["reason"] set

Only ``age`` and ``sex`` by default. ``dm_time`` (diabetes duration) is a far
stronger predictor of the microvascular complications but is NOT inferable
from a photograph in the same way, so it is offered as an opt-in second,
stricter baseline (``features=("age", "sex", "dm_time")``) rather than the
default: beating age+sex says "there is retinal signal beyond ageing"; beating
age+sex+duration says "beyond what a clinician already knows from the chart".
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LABEL_REGISTRY, _isnan_vector          # noqa: E402

DEFAULT_FEATURES = ("age", "sex")


def _labels(df: pd.DataFrame, task: str) -> np.ndarray:
    spec = LABEL_REGISTRY[task]
    if any(c not in df.columns for c in spec.source_cols):
        return np.full(len(df), np.nan)
    return np.array([spec.fn(dict(zip(spec.source_cols, row)))
                     for row in df[list(spec.source_cols)].to_numpy()], dtype=np.float64)


def _feature_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Numeric matrix; object columns (e.g. sex as 'M'/'F') are factorised.

    The factorisation is per-call, so it is applied to the concatenation of
    train and test by the caller -- see :func:`covariate_baseline`.
    """
    cols = []
    for f in features:
        s = df[f]
        if s.dtype == object:
            codes, _ = pd.factorize(s.astype(str).str.strip().str.lower().where(s.notna()))
            col = codes.astype(np.float64)
            col[codes < 0] = np.nan
        else:
            col = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        cols.append(col)
    return np.stack(cols, axis=1)


def covariate_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, task: str,
                       features: Sequence[str] = DEFAULT_FEATURES,
                       seed: int = 0) -> Dict[str, object]:
    """Fit logistic regression on ``features`` (train split) and score AUROC on
    the test split. Returns a JSON-serialisable dict; ``auroc`` is NaN with a
    ``reason`` whenever the number would be meaningless."""
    out: Dict[str, object] = {"features": list(features), "auroc": float("nan"),
                              "n_train": 0, "n_test": 0, "test_pos": None, "reason": None}
    spec = LABEL_REGISTRY.get(task)
    if spec is None or spec.num_classes != 2:
        out["reason"] = f"task {task!r} is not binary"
        return out
    missing = [f for f in features if f not in train_df.columns or f not in test_df.columns]
    if missing:
        out["reason"] = f"covariate column(s) missing: {missing}"
        return out

    # Factorise object columns over train+test together so codes agree.
    both = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    X_all = _feature_matrix(both, features)
    y_all = _labels(both, task)
    n_tr = len(train_df)
    X_tr, X_te = X_all[:n_tr], X_all[n_tr:]
    y_tr, y_te = y_all[:n_tr], y_all[n_tr:]
    keep_tr = ~(_isnan_vector(y_tr) | np.isnan(X_tr).any(1))
    keep_te = ~(_isnan_vector(y_te) | np.isnan(X_te).any(1))
    X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr].astype(int)
    X_te, y_te = X_te[keep_te], y_te[keep_te].astype(int)
    out["n_train"], out["n_test"] = int(len(y_tr)), int(len(y_te))
    out["test_pos"] = int((y_te == 1).sum()) if len(y_te) else None
    if len(np.unique(y_tr)) < 2:
        out["reason"] = "single-class train split"
        return out
    if len(np.unique(y_te)) < 2:
        out["reason"] = "single-class test split"
        return out

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced",
                                           random_state=seed))
    clf.fit(X_tr, y_tr)
    out["auroc"] = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def incremental_value(val_df: pd.DataFrame, val_prob: np.ndarray,
                      test_df: pd.DataFrame, test_prob: np.ndarray, task: str,
                      features: Sequence[str] = DEFAULT_FEATURES,
                      subsets: Optional[Dict[str, np.ndarray]] = None,
                      n_boot: int = 2000, seed: int = 0) -> Dict[str, object]:
    """Does the photo add to the covariates? The standard oculomics test.

    Two logistic regressions are fit on the VALIDATION rows -- covariates alone,
    and covariates + the image model's logit -- and both are scored on the TEST
    rows. ``delta`` = AUROC(covariates + image) - AUROC(covariates), with a
    patient-cluster bootstrap 95% CI and the share of resamples where the image
    adds nothing (``p_no_gain``, one-sided). Fitting both on the same rows with
    the same method makes the comparison like-for-like; neither model sees a
    test label. (The image model was checkpoint-selected on these val rows, so
    its val scores are mildly optimistic: that can only overweight the image
    term, which the test score would then penalise, not reward.)

    ``*_prob`` are the image model's P(class 1) in the same row order as the
    frames. ``subsets`` maps a name to a boolean mask over ``test_df`` rows
    (e.g. non-diabetic DR-grade-0 patients); the same two fitted models are
    re-scored inside each.
    """
    out: Dict[str, object] = {"features": list(features), "reason": None}
    spec = LABEL_REGISTRY.get(task)
    if spec is None or spec.num_classes != 2:
        out["reason"] = f"task {task!r} is not binary"
        return out
    missing = [f for f in features if f not in val_df.columns or f not in test_df.columns]
    if missing:
        out["reason"] = f"covariate column(s) missing: {missing}"
        return out
    if "patient" not in test_df.columns:
        out["reason"] = "no patient column for the cluster bootstrap"
        return out

    both = pd.concat([val_df, test_df], axis=0, ignore_index=True)
    X_all = _feature_matrix(both, features)
    y_all = _labels(both, task)
    z_all = _logit(np.concatenate([val_prob, test_prob]))
    n_v = len(val_df)
    ok = ~(_isnan_vector(y_all) | np.isnan(X_all).any(1) | np.isnan(z_all))
    v, t = ok[:n_v], ok[n_v:]
    Xv, yv, zv = X_all[:n_v][v], y_all[:n_v][v].astype(int), z_all[:n_v][v]
    Xt, yt, zt = X_all[n_v:][t], y_all[n_v:][t].astype(int), z_all[n_v:][t]
    groups = test_df["patient"].astype(str).to_numpy()[t]
    out.update(n_val=int(len(yv)), n_test=int(len(yt)), test_pos=int(yt.sum()),
               test_pos_patients=int(len(np.unique(groups[yt == 1]))))
    if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
        out["reason"] = "single-class val or test rows"
        return out

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(X, y):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)

    cov = fit(Xv, yv)
    comb = fit(np.column_stack([Xv, zv]), yv)
    p_cov = cov.predict_proba(Xt)[:, 1]
    p_comb = comb.predict_proba(np.column_stack([Xt, zt]))[:, 1]
    out["image_coef_std"] = float(comb[-1].coef_.ravel()[-1])    # per SD of the image logit

    def block(mask: np.ndarray) -> Dict[str, object]:
        y, g = yt[mask], groups[mask]
        res: Dict[str, object] = {"n": int(mask.sum()), "pos": int(y.sum()),
                                  "pos_patients": int(len(np.unique(g[y == 1])))}
        if len(np.unique(y)) < 2:
            res["reason"] = "single-class"
            return res
        a_cov = float(roc_auc_score(y, p_cov[mask]))
        a_comb = float(roc_auc_score(y, p_comb[mask]))
        res.update(auroc_covariates=a_cov, auroc_combined=a_comb,
                   auroc_image=float(roc_auc_score(y, zt[mask])), delta=a_comb - a_cov)
        # Patient-cluster bootstrap of the paired delta (both eyes move together).
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        pc, pb = p_cov[mask], p_comb[mask]
        deltas = []
        for _ in range(n_boot):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pc[idx]))
        if deltas:
            res["delta_ci"] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
            res["p_no_gain"] = float(np.mean(np.asarray(deltas) <= 0))
        return res

    out["test"] = block(np.ones(len(yt), dtype=bool))
    out["subsets"] = {}
    for name, m in (subsets or {}).items():
        out["subsets"][name] = block(np.asarray(m, dtype=bool)[t])
    return out


def image_minus_covariate(image_auroc: Optional[float], cb: Dict[str, object]) -> Optional[float]:
    """Paired within-run delta (image model minus covariate baseline); None if
    either side is missing/NaN so summarisers skip it instead of averaging NaN."""
    a = cb.get("auroc") if cb else None
    if image_auroc is None or a is None or a != a or image_auroc != image_auroc:
        return None
    return float(image_auroc - a)
