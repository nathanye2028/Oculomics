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
    out, fit = _fit_incremental(val_df, val_prob, test_df, test_prob, task, features)
    if fit is None:
        return out
    n = len(fit["y"])
    out["test"] = incremental_block(fit["y"], fit["groups"], fit["p_cov"], fit["p_comb"], fit["z"],
                                    np.ones(n, dtype=bool), n_boot=n_boot, seed=seed)
    out["subsets"] = {}
    for name, m in (subsets or {}).items():
        out["subsets"][name] = incremental_block(fit["y"], fit["groups"], fit["p_cov"], fit["p_comb"],
                                                 fit["z"], np.asarray(m, dtype=bool)[fit["keep"]],
                                                 n_boot=n_boot, seed=seed)
    return out


def _fit_incremental(val_df: pd.DataFrame, val_prob: np.ndarray, test_df: pd.DataFrame,
                     test_prob: np.ndarray, task: str, features: Sequence[str]):
    """The two logistic models of :func:`incremental_value`, fit on the val rows
    and scored on the test rows. Returns ``(out, fit)``: ``out`` holds the counts
    (or a ``reason``); ``fit`` the kept test rows' arrays, or None when the
    comparison cannot be made."""
    out: Dict[str, object] = {"features": list(features), "reason": None}
    spec = LABEL_REGISTRY.get(task)
    if spec is None or spec.num_classes != 2:
        out["reason"] = f"task {task!r} is not binary"
        return out, None
    missing = [f for f in features if f not in val_df.columns or f not in test_df.columns]
    if missing:
        out["reason"] = f"covariate column(s) missing: {missing}"
        return out, None
    if "patient" not in test_df.columns:
        out["reason"] = "no patient column for the cluster bootstrap"
        return out, None

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
        return out, None

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(X, y):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)

    cov = fit(Xv, yv)
    comb = fit(np.column_stack([Xv, zv]), yv)
    out["image_coef_std"] = float(comb[-1].coef_.ravel()[-1])    # per SD of the image logit
    return out, {"keep": t, "y": yt, "z": zt, "groups": groups,
                 "p_cov": cov.predict_proba(Xt)[:, 1],
                 "p_comb": comb.predict_proba(np.column_stack([Xt, zt]))[:, 1]}


def incremental_block(y: np.ndarray, groups: np.ndarray, p_cov: np.ndarray, p_comb: np.ndarray,
                      z: np.ndarray, mask: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> Dict[str, object]:
    """AUROC of covariates alone, covariates + image and the image score alone on
    the ``mask`` rows; ``delta`` (combined minus covariates) with a patient-cluster
    bootstrap 95 % CI (both eyes move together) and ``p_no_gain``. Shared by one
    run's test split and by folds pooled out of fold (summarize_systemic.py)."""
    from sklearn.metrics import roc_auc_score
    mask = np.asarray(mask, dtype=bool)
    yy, g = np.asarray(y)[mask].astype(int), np.asarray(groups).astype(str)[mask]
    res: Dict[str, object] = {"n": int(mask.sum()), "pos": int(yy.sum()),
                              "pos_patients": int(len(np.unique(g[yy == 1])))}
    if len(np.unique(yy)) < 2:
        res["reason"] = "single-class"
        return res
    pc, pb = np.asarray(p_cov)[mask], np.asarray(p_comb)[mask]
    a_cov = float(roc_auc_score(yy, pc))
    a_comb = float(roc_auc_score(yy, pb))
    res.update(auroc_covariates=a_cov, auroc_combined=a_comb,
               auroc_image=float(roc_auc_score(yy, np.asarray(z)[mask])), delta=a_comb - a_cov)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(yy[idx])) < 2:
            continue
        deltas.append(roc_auc_score(yy[idx], pb[idx]) - roc_auc_score(yy[idx], pc[idx]))
    if deltas:
        res["delta_ci"] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        res["p_no_gain"] = float(np.mean(np.asarray(deltas) <= 0))
    return res


def incremental_rows(val_df: pd.DataFrame, val_prob: np.ndarray, test_df: pd.DataFrame,
                     test_prob: np.ndarray, task: str,
                     features: Sequence[str] = DEFAULT_FEATURES) -> pd.DataFrame:
    """Per test row, the two fitted models' probabilities (``p_covariates``,
    ``p_combined``) in ``test_df`` order, NaN where a row has no label or a
    covariate is missing, or where :func:`incremental_value` would refuse. Written
    per fold, so folds can be pooled out of fold."""
    out = pd.DataFrame({"p_covariates": np.nan, "p_combined": np.nan}, index=range(len(test_df)))
    _, fit = _fit_incremental(val_df, val_prob, test_df, test_prob, task, features)
    if fit is not None:
        keep = np.flatnonzero(fit["keep"])
        out.loc[keep, "p_covariates"] = fit["p_cov"]
        out.loc[keep, "p_combined"] = fit["p_comb"]
    return out


def matched_pair_concordance(y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                             age: np.ndarray, exact: Optional[Dict[str, np.ndarray]] = None,
                             caliper: float = 2.0, n_boot: int = 2000,
                             seed: int = 0) -> Dict[str, object]:
    """How often ``score`` ranks a positive patient above a negative one who is
    within ``caliper`` years of age and identical on every ``exact`` column (sex,
    diabetes status, ...). Every such case-control pair counts once (ties count
    1/2), so what the pairs share cannot carry the result: 0.5 = nothing beyond
    them. The board version of "beyond age and sex": given two such people, one
    with the condition, how often did the photo pick the right one?

    Per patient: image scores averaged, label / age / exact columns taken from the
    patient's rows (constant for patient-level labels). 95 % CI from resampling
    patients, each pair weighted by how often both of its patients were drawn."""
    cols = {"g": np.asarray(groups).astype(str), "y": np.asarray(y, dtype=float),
            "p": np.asarray(score, dtype=float), "age": pd.to_numeric(pd.Series(age), errors="coerce").to_numpy()}
    names = list(exact or {})
    for k in names:
        cols[f"x_{k}"] = pd.Series(np.asarray(exact[k], dtype=object)).astype(str).str.strip().str.lower() \
            .where(pd.Series(exact[k]).notna(), None).to_numpy()
    d = pd.DataFrame(cols).dropna()
    out: Dict[str, object] = {"caliper_years": float(caliper), "matched_on": ["age"] + names}
    if d.empty:
        out["reason"] = "no patient with a score, a label, an age and every matching column"
        return out
    agg = {"y": ("y", "max"), "p": ("p", "mean"), "age": ("age", "first")}
    agg.update({f"x_{k}": (f"x_{k}", "first") for k in names})
    per = d.groupby("g").agg(**agg).reset_index(drop=True)
    key = per[[f"x_{k}" for k in names]].astype(str).agg("|".join, axis=1) if names else \
        pd.Series("", index=per.index)
    ii, jj, val = [], [], []
    for k in key.unique():
        a = np.flatnonzero(((per["y"] == 1) & (key == k)).to_numpy())
        b = np.flatnonzero(((per["y"] == 0) & (key == k)).to_numpy())
        if not len(a) or not len(b):
            continue
        close = np.abs(per["age"].to_numpy()[a][:, None] - per["age"].to_numpy()[b][None, :]) <= caliper
        ia, ib = np.nonzero(close)
        pa, pb = per["p"].to_numpy()[a[ia]], per["p"].to_numpy()[b[ib]]
        ii.append(a[ia]); jj.append(b[ib])
        val.append(np.where(pa > pb, 1.0, np.where(pa == pb, 0.5, 0.0)))
    out.update(cases=int((per["y"] == 1).sum()), controls=int((per["y"] == 0).sum()))
    if not ii:
        out["reason"] = "no case-control pair matches"
        return out
    ii, jj, val = np.concatenate(ii), np.concatenate(jj), np.concatenate(val)
    out.update(pairs=int(len(val)), cases_matched=int(len(np.unique(ii))),
               controls_matched=int(len(np.unique(jj))), concordance=float(val.mean()))
    rng = np.random.default_rng(seed)
    n, boots = len(per), []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        ww = w[ii] * w[jj]
        if ww.sum() > 0:
            boots.append(float((ww * val).sum() / ww.sum()))
    if boots:
        out["ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["p_no_signal"] = float(np.mean(np.asarray(boots) <= 0.5))
    return out


def image_minus_covariate(image_auroc: Optional[float], cb: Dict[str, object]) -> Optional[float]:
    """Paired within-run delta (image model minus covariate baseline); None if
    either side is missing/NaN so summarisers skip it instead of averaging NaN."""
    a = cb.get("auroc") if cb else None
    if image_auroc is None or a is None or a != a or image_auroc != image_auroc:
        return None
    return float(image_auroc - a)
