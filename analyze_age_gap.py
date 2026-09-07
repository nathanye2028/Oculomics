#!/usr/bin/env python3
"""
analyze_age_gap.py
==================
Step 2 of the retinal-age track: does the **retinal age gap** (predicted minus
chronological age, bias-corrected) track disease?

Reads the per-image predictions table written by ``train_retinal_age.py`` (or
pooled over seeds by ``summarize_retinal_age.py``) and, per dataset, tests every
available exposure — diabetes, insulin use, any / referable DR, macular edema,
BRSET's ophthalmic flags (joined from the raw BRSET CSV), mBRSET's systemic
labels — against the gap at the **patient** level, adjusted for age, age², sex
and (BRSET) camera. Per exposure it reports:

* mean corrected gap in exposed vs reference patients; the unadjusted (Welch)
  and covariate-adjusted (OLS) difference with 95 % CI and p; Benjamini-Hochberg
  q across the table; Cohen's d;
* **disease prevalence by quintile of the gap** and the odds ratio per +5 years
  of gap from a logistic regression with the same covariates — the "prevalence
  vs retinal age" view;
* the trend across DR grades 0-4 and against diabetes duration.

Which gap column
----------------
In-domain rows (``cohort`` healthy / nonhealthy): ``gap_corrected``, the gap
bias-corrected with the in-domain healthy-val fit. External rows (``cohort``
external): ``gap_bnadapt_recal_corrected`` — the AdaBN prediction, device-
calibrated and corrected WITHIN the external set — when present, otherwise
``gap_recal_corrected``, otherwise ``gap_bnadapt`` with a warning (the in-domain
correction does not transfer to phone images; age stays a covariate, which
absorbs the linear part). ``--gap-col`` overrides both.

Unit of analysis is the patient (both eyes averaged) and only gradable images
count, because an ungradable image can read as older for optical reasons. That
artefact is quantified separately (ungradable vs gradable among disease-free
patients, all images). Nothing in the table was trained on: healthy rows are the
held-out test split, non-healthy rows were excluded from training.

    python analyze_age_gap.py --predictions exp_retinal_age/predictions_pooled.csv \\
        --brset-csv <BRSET>/labels_brset.csv --out exp_retinal_age/associations

This is descriptive epidemiology on one cohort per dataset: report effect sizes
with their intervals; the q-values guard the table, not the thesis.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_retinal_age import PATHOLOGY_COLS, _flag          # noqa: E402

SYSTEMIC_COLS = ("systemic_hypertension", "nephropathy", "neuropathy", "acute_myocardial_infarction",
                 "vascular_disease", "diabetic_foot", "obesity", "smoking", "alcohol_consumption")
FLAG_COLS = ("diabetes", "insulin", "final_edema", "gradable") + PATHOLOGY_COLS + SYSTEMIC_COLS
EXTERNAL_GAP_PREFERENCE = ("gap_bnadapt_recal_corrected", "gap_recal_corrected", "gap_bnadapt")
MIN_CASES = 10           # fewer exposed (or reference) patients than this -> row reported as n/a


# --------------------------------------------------------------------------- #
# Statistics (numpy + scipy only; no statsmodels on the boxes)
# --------------------------------------------------------------------------- #
def ols(y: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
    """OLS with an intercept. Returns coef / se / p / ci for every column of X."""
    y = np.asarray(y, float)
    Xc = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = Xc.shape
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    tc = stats.t.ppf(0.975, dof)
    return {"coef": beta[1:], "se": se[1:], "p": p[1:], "lo": (beta - tc * se)[1:],
            "hi": (beta + tc * se)[1:], "n": n, "dof": dof}


def logit(y: np.ndarray, X: np.ndarray, max_iter: int = 60, ridge: float = 1e-6
          ) -> Optional[Dict[str, np.ndarray]]:
    """Logistic regression by Newton-Raphson with an intercept; Wald SEs.
    Returns None if it does not converge (separation, tiny groups)."""
    y = np.asarray(y, float)
    Xc = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = Xc.shape
    beta = np.zeros(k)
    H = None
    for _ in range(max_iter):
        eta = np.clip(Xc @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        H = Xc.T @ (Xc * w[:, None]) + ridge * np.eye(k)
        g = Xc.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        return None
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 25:
        return None
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.norm.sf(np.abs(z))
    return {"coef": beta[1:], "se": se[1:], "p": p[1:], "lo": (beta - 1.96 * se)[1:],
            "hi": (beta + 1.96 * se)[1:], "n": n}


def bh_fdr(p: List[float]) -> List[float]:
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return list(q)
    idx = np.where(ok)[0]
    order = np.argsort(p[idx])
    ranked = p[idx][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[idx[order]] = np.clip(ranked, 0, 1)
    return list(q)


def welch(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
    """mean(a)-mean(b), 95% CI, p (Welch t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tc = stats.t.ppf(0.975, dof)
    p = 2 * stats.t.sf(abs(d / se), dof) if se > 0 else float("nan")
    return float(d), float(d - tc * se), float(d + tc * se), float(p)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_predictions(path: str, condition: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(path)
    for c in ("dataset", "file", "patient", "age", "cohort"):
        if c not in df.columns:
            raise KeyError(f"{path} lacks column {c!r}; expected a train_retinal_age.py predictions table")
    used = None
    if "condition" in df.columns:
        conds = list(pd.unique(df["condition"]))
        used = condition or ("student" if "student" in conds else conds[0])
        if used not in conds:
            raise ValueError(f"condition {used!r} not in {conds}")
        df = df[df["condition"] == used].copy()
    return df, used


def attach_brset_flags(df: pd.DataFrame, brset_csv: str, image_ext: str = ".jpg") -> Tuple[pd.DataFrame, List[str]]:
    """Join BRSET's ophthalmic flags (raw CSV, by image id) onto the brset rows."""
    raw = pd.read_csv(brset_csv)
    cols = [c for c in PATHOLOGY_COLS if c in raw.columns]
    if not cols or "image_id" not in raw.columns:
        warnings.warn(f"{brset_csv}: no ophthalmic flag columns / image_id; nothing joined")
        return df, []
    key = raw["image_id"].astype(str).map(lambda f: f if os.path.splitext(f)[1] else f + image_ext)
    flags = raw[cols].apply(lambda s: s.map(_flag)).astype(float)
    flags.index = key.values
    flags = flags[~flags.index.duplicated()]
    out = df.copy()
    is_b = out["dataset"].astype(str).str.lower() == "brset"
    for c in cols:
        out.loc[is_b, c] = out.loc[is_b, "file"].map(flags[c])
    return out, cols


def pick_gap_col(df: pd.DataFrame, override: Optional[str] = None) -> Tuple[str, str]:
    """(column, note) for one dataset's rows."""
    if override:
        if override not in df.columns:
            raise KeyError(f"--gap-col {override!r} not in the table")
        return override, "user override"
    external = (df["cohort"].astype(str) == "external").all()
    if not external:
        return "gap_corrected", "bias-corrected with the in-domain healthy-val fit"
    for c in EXTERNAL_GAP_PREFERENCE:
        if c in df.columns and df[c].notna().any():
            note = {"gap_bnadapt_recal_corrected": "AdaBN prediction, device-calibrated and corrected within this set",
                    "gap_recal_corrected": "device-calibrated and corrected within this set (no AdaBN column)",
                    "gap_bnadapt": "AdaBN prediction, NOT device-calibrated (rerun the trainer for "
                                   "external_recal); age is a covariate below"}[c]
            if c == "gap_bnadapt":
                warnings.warn("external rows have no device-calibrated gap column; using gap_bnadapt")
            return c, note
    return "gap", "raw gap (no adapted / calibrated columns present)"


def _sex_numeric(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0 and s.dtype == object:
        v = s.astype(str).str.strip().str.lower().map(
            {"m": 0.0, "male": 0.0, "masculino": 0.0, "f": 1.0, "female": 1.0, "feminino": 1.0})
    return v.astype(float)


def to_patients(df: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    """Patient-level table: gap and age averaged over the patient's images, flags = any eye,
    DR grade = worst eye, gradable = every image gradable."""
    d = df.copy()
    for c in FLAG_COLS:
        if c in d.columns:
            d[c] = d[c].map(_flag).astype(float)
    if "sex" in d.columns:
        d["sex"] = _sex_numeric(d["sex"])
    d["dr_grade"] = pd.to_numeric(d["dr_grade"], errors="coerce") if "dr_grade" in d.columns else np.nan
    d["dm_time"] = pd.to_numeric(d["dm_time"], errors="coerce") if "dm_time" in d.columns else np.nan
    agg = {"gap": (gap_col, "mean"), "age": ("age", "mean"), "n_images": ("file", "size"),
           "split": ("split", "first"), "dr_grade": ("dr_grade", "max"), "dm_time": ("dm_time", "mean"),
           "cohort": ("cohort", lambda s: "healthy" if (s.astype(str) == "healthy").all() else str(s.iloc[0]))}
    if "sex" in d.columns:
        agg["sex"] = ("sex", "first")
    if "camera" in d.columns:
        agg["camera"] = ("camera", "first")
    for c in FLAG_COLS:
        if c in d.columns:
            agg[c] = (c, "min" if c == "gradable" else "max")
    pat = d.groupby("patient", sort=False).agg(**agg).reset_index()
    pat["dr_any"] = np.where(pat["dr_grade"].isna(), np.nan, (pat["dr_grade"] >= 1).astype(float))
    pat["dr_referable"] = np.where(pat["dr_grade"].isna(), np.nan, (pat["dr_grade"] >= 2).astype(float))
    if "gradable" in pat.columns:
        pat["ungradable"] = 1.0 - pat["gradable"]
    return pat


def covariates(pat: pd.DataFrame, min_camera: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """age (centred, /10), age², sex, camera dummies (levels with >= min_camera patients)."""
    X = pd.DataFrame(index=pat.index)
    a = (pat["age"] - pat["age"].mean()) / 10.0
    X["age10"] = a
    X["age10_sq"] = a ** 2
    used = ["age", "age²"]
    if "sex" in pat.columns and pat["sex"].notna().sum() > 0 and pat["sex"].nunique(dropna=True) == 2:
        X["sex"] = pat["sex"]
        used.append("sex")
    if "camera" in pat.columns:
        vc = pat["camera"].astype(str).value_counts()
        levels = [l for l, n in vc.items() if n >= min_camera and l not in ("nan", "")]
        if len(levels) >= 2:
            for l in levels[1:]:
                X[f"camera={l}"] = (pat["camera"].astype(str) == l).astype(float)
            used.append(f"camera({len(levels)} levels)")
    return X, used


# --------------------------------------------------------------------------- #
# Associations
# --------------------------------------------------------------------------- #
def assoc_binary(pat: pd.DataFrame, X: pd.DataFrame, col: str, label: str,
                 within: Optional[str] = None) -> Dict[str, object]:
    row = {"exposure": label, "column": col, "within": within or "all", "n_exposed": 0, "n_ref": 0,
           "gap_exposed": np.nan, "gap_ref": np.nan, "delta": np.nan, "delta_lo": np.nan, "delta_hi": np.nan,
           "p_unadj": np.nan, "adj_delta": np.nan, "adj_lo": np.nan, "adj_hi": np.nan, "p_adj": np.nan,
           "cohen_d": np.nan, "or_per5y": np.nan, "or_lo": np.nan, "or_hi": np.nan, "p_or": np.nan,
           "quintiles": None, "note": ""}
    if col not in pat.columns:
        row["note"] = "column absent"
        return row
    d = pat
    if within:
        if within in pat.columns and pat[within].notna().any():
            d = pat[pat[within] == 1.0]
        else:
            row["within"] = "all (no such column: every patient qualifies)"
    keep = d[col].notna() & d["gap"].notna() & X.loc[d.index].notna().all(axis=1)
    d, Xd = d[keep], X.loc[d.index[keep]]
    x = d[col].astype(float).to_numpy()
    g = d["gap"].to_numpy(float)
    n1, n0 = int((x == 1).sum()), int((x == 0).sum())
    row.update({"n_exposed": n1, "n_ref": n0})
    if n1 < MIN_CASES or n0 < MIN_CASES:
        row["note"] = f"too few patients (need >= {MIN_CASES} per group)"
        return row
    row["gap_exposed"], row["gap_ref"] = float(g[x == 1].mean()), float(g[x == 0].mean())
    row["delta"], row["delta_lo"], row["delta_hi"], row["p_unadj"] = welch(g[x == 1], g[x == 0])
    sd0 = float(g[x == 0].std(ddof=1))
    row["cohen_d"] = row["delta"] / sd0 if sd0 > 0 else np.nan
    o = ols(g, np.column_stack([x, Xd.to_numpy(float)]))
    row.update({"adj_delta": float(o["coef"][0]), "adj_lo": float(o["lo"][0]),
                "adj_hi": float(o["hi"][0]), "p_adj": float(o["p"][0])})
    lg = logit(x, np.column_stack([g / 5.0, Xd.to_numpy(float)]))
    # A Wald SE above 5 on the log-odds means the fit is numerically unstable (near-constant
    # gap, quasi-separation): an OR of 20,000 with a CI spanning ten orders of magnitude is
    # not a result, so it is reported as unstable instead.
    if lg is not None and float(lg["se"][0]) > 5.0:
        lg = None
    if lg is not None:
        row.update({"or_per5y": float(np.exp(lg["coef"][0])), "or_lo": float(np.exp(lg["lo"][0])),
                    "or_hi": float(np.exp(lg["hi"][0])), "p_or": float(lg["p"][0])})
    else:
        row["note"] = "logistic unstable / did not converge"
    try:
        qbin = pd.qcut(g, 5, labels=False, duplicates="drop")
        row["quintiles"] = [(int((qbin == q).sum()), float(x[qbin == q].mean()))
                            for q in sorted(pd.unique(qbin))]
    except ValueError:
        row["quintiles"] = None
    return row


def assoc_grade(pat: pd.DataFrame, X: pd.DataFrame, within: Optional[str] = "diabetes") -> Optional[Dict[str, object]]:
    d = pat
    if within and within in pat.columns and pat[within].notna().any():
        d = pat[pat[within] == 1.0]
    keep = d["dr_grade"].notna() & d["gap"].notna() & X.loc[d.index].notna().all(axis=1)
    d, Xd = d[keep], X.loc[d.index[keep]]
    if len(d) < 2 * MIN_CASES or d["dr_grade"].nunique() < 2:
        return None
    by = []
    for gr, f in d.groupby("dr_grade"):
        g = f["gap"].to_numpy(float)
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        by.append({"grade": int(gr), "n": int(len(g)), "mean": float(g.mean()),
                   "lo": float(g.mean() - 1.96 * se), "hi": float(g.mean() + 1.96 * se)})
    o = ols(d["gap"].to_numpy(float), np.column_stack([d["dr_grade"].to_numpy(float), Xd.to_numpy(float)]))
    rho, p_rho = stats.spearmanr(d["dr_grade"], d["gap"])
    return {"by_grade": by, "slope": float(o["coef"][0]), "lo": float(o["lo"][0]), "hi": float(o["hi"][0]),
            "p": float(o["p"][0]), "rho": float(rho), "p_rho": float(p_rho), "n": int(len(d))}


def assoc_duration(pat: pd.DataFrame, X: pd.DataFrame, within: Optional[str] = "diabetes") -> Optional[Dict[str, object]]:
    d = pat
    if within and within in pat.columns and pat[within].notna().any():
        d = pat[pat[within] == 1.0]
    keep = d["dm_time"].notna() & d["gap"].notna() & X.loc[d.index].notna().all(axis=1)
    d, Xd = d[keep], X.loc[d.index[keep]]
    if len(d) < 2 * MIN_CASES or d["dm_time"].nunique() < 3:
        return None
    rho, p_rho = stats.spearmanr(d["dm_time"], d["gap"])
    o = ols(d["gap"].to_numpy(float), np.column_stack([d["dm_time"].to_numpy(float) / 10.0, Xd.to_numpy(float)]))
    return {"n": int(len(d)), "rho": float(rho), "p_rho": float(p_rho), "slope_per10y": float(o["coef"][0]),
            "lo": float(o["lo"][0]), "hi": float(o["hi"][0]), "p": float(o["p"][0])}


def exposures_for(pat: pd.DataFrame, is_external: bool, joined: List[str]) -> List[Tuple[str, str, Optional[str]]]:
    """(column, label, within) in report order, only for columns present."""
    ex: List[Tuple[str, str, Optional[str]]] = []
    if not is_external:
        ex.append(("diabetes", "diabetes", None))
    ex += [("insulin", "insulin use", "diabetes"), ("dr_any", "any DR (grade >= 1)", "diabetes"),
           ("dr_referable", "referable DR (grade >= 2)", "diabetes"), ("final_edema", "macular edema", "diabetes")]
    ex += [(c, c.replace("_", " "), None) for c in joined]
    ex += [(c, c.replace("_", " "), None) for c in SYSTEMIC_COLS]
    return [(c, l, w) for c, l, w in ex if c in pat.columns and pat[c].notna().any()]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _f(v, prec=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"


def analyse_dataset(df: pd.DataFrame, name: str, gap_override: Optional[str], joined: List[str],
                    include_ungradable: bool) -> Tuple[List[str], List[Dict[str, object]]]:
    L: List[str] = []
    gap_col, note = pick_gap_col(df, gap_override)
    is_ext = (df["cohort"].astype(str) == "external").all()
    pat_all = to_patients(df, gap_col)
    if "gradable" in pat_all.columns and not include_ungradable:
        img_ok = df[df["gradable"].map(_flag) == 1.0] if "gradable" in df.columns else df
        pat = to_patients(img_ok, gap_col)
        quality_note = f"gradable images only ({len(pat)} of {len(pat_all)} patients keep >= 1 gradable image)"
    else:
        pat = pat_all
        quality_note = "all images"
    X, used = covariates(pat)
    L.append(f"\n## {name}  ({'external' if is_ext else 'in-domain'} rows)")
    L.append(f"gap column: `{gap_col}` — {note}. Unit: patient ({len(pat)} patients, "
             f"{int(pat['n_images'].sum())} images; {quality_note}). Covariates: {', '.join(used)}.")
    if not is_ext:
        h = pat[pat["cohort"] == "healthy"]
        if len(h):
            L.append(f"reference sanity check — healthy held-out patients: mean gap {_f(h['gap'].mean(), sign=True)} y "
                     f"(should sit near 0), SD {_f(h['gap'].std())} y, n={len(h)}")

    rows = [assoc_binary(pat, X, c, l, w) for c, l, w in exposures_for(pat, is_ext, joined)]
    for r in rows:
        r["dataset"] = name
    qs = bh_fdr([r["p_adj"] for r in rows])
    for r, q in zip(rows, qs):
        r["q_adj"] = q
    if rows:
        L.append(f"\n| exposure | within | n exposed / ref | gap exposed / ref (y) | unadjusted Δ [95% CI] | "
                 f"adjusted Δ [95% CI] | p | q | Cohen d | OR per +5 y gap [95% CI] | note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['exposure']} | {r['within']} | {r['n_exposed']} / {r['n_ref']} | "
                     f"{_f(r['gap_exposed'], sign=True)} / {_f(r['gap_ref'], sign=True)} | "
                     f"{_f(r['delta'], sign=True)} [{_f(r['delta_lo'], sign=True)}, {_f(r['delta_hi'], sign=True)}] | "
                     f"{_f(r['adj_delta'], sign=True)} [{_f(r['adj_lo'], sign=True)}, {_f(r['adj_hi'], sign=True)}] | "
                     f"{_f(r['p_adj'], 4)} | {_f(r['q_adj'], 3)} | {_f(r['cohen_d'])} | "
                     f"{_f(r['or_per5y'])} [{_f(r['or_lo'])}, {_f(r['or_hi'])}] | {r['note']} |")
        L.append("\nΔ = exposed minus reference, in years of corrected gap. OR per +5 y: odds of the exposure "
                 "for a 5-year-older retina, adjusted for the covariates. q = Benjamini-Hochberg over this table.")
        quint = [r for r in rows if r["quintiles"]]
        if quint:
            k = max(len(r["quintiles"]) for r in quint)
            L.append(f"\n### {name}: prevalence by quintile of the corrected gap (Q1 = youngest-looking retinas)")
            L.append("| exposure | " + " | ".join(f"Q{i+1}" for i in range(k)) + " |")
            L.append("|---|" + "---|" * k)
            for r in quint:
                cells = [f"{100*p:.1f}% (n={n})" for n, p in r["quintiles"]]
                L.append(f"| {r['exposure']} ({r['within']}) | " + " | ".join(cells + ["-"] * (k - len(cells))) + " |")

    gr = assoc_grade(pat, X)
    if gr:
        L.append(f"\n### {name}: corrected gap by DR grade (worst eye; {'all patients' if is_ext else 'diabetics'}, n={gr['n']})")
        L.append("| grade | n | mean gap [95% CI] |")
        L.append("|---|---|---|")
        for b in gr["by_grade"]:
            L.append(f"| {b['grade']} | {b['n']} | {_f(b['mean'], sign=True)} [{_f(b['lo'], sign=True)}, {_f(b['hi'], sign=True)}] |")
        L.append(f"adjusted slope {_f(gr['slope'], sign=True)} y per grade [{_f(gr['lo'], sign=True)}, "
                 f"{_f(gr['hi'], sign=True)}], p={_f(gr['p'], 4)}; Spearman ρ={_f(gr['rho'], 3)} (p={_f(gr['p_rho'], 4)})")
    du = assoc_duration(pat, X)
    if du:
        L.append(f"\n### {name}: corrected gap vs diabetes duration (n={du['n']})")
        L.append(f"Spearman ρ={_f(du['rho'], 3)} (p={_f(du['p_rho'], 4)}); adjusted slope "
                 f"{_f(du['slope_per10y'], sign=True)} y per 10 years of diabetes "
                 f"[{_f(du['lo'], sign=True)}, {_f(du['hi'], sign=True)}], p={_f(du['p'], 4)}")

    # Quality artefact: ungradable vs gradable among disease-free patients, all images.
    if "ungradable" in pat_all.columns and not is_ext:
        free = pat_all
        for c in ("diabetes", "dr_any"):
            if c in free.columns and free[c].notna().any():
                free = free[free[c] != 1.0]
        Xf, _ = covariates(free)
        qr = assoc_binary(free, Xf, "ungradable", "ungradable image (quality artefact check)")
        L.append(f"\n### {name}: image-quality artefact check (disease-free patients, all images)")
        if qr["note"] and qr["n_exposed"] < MIN_CASES:
            L.append(f"n/a — {qr['note']} (ungradable patients: {qr['n_exposed']})")
        else:
            L.append(f"patients with an ungradable image read {_f(qr['adj_delta'], sign=True)} y "
                     f"[{_f(qr['adj_lo'], sign=True)}, {_f(qr['adj_hi'], sign=True)}] older than fully "
                     f"gradable ones (n={qr['n_exposed']} vs {qr['n_ref']}, p={_f(qr['p_adj'], 4)}). "
                     f"If this is comparable to the disease effects above, quality — not biology — is a "
                     f"candidate explanation for any exposure that also lowers image quality.")
    return L, rows


def maybe_plots(all_rows: List[Dict[str, object]], out_dir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = [r for r in all_rows if r.get("quintiles")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in rows:
        ys = [100 * p for _, p in r["quintiles"]]
        ax.plot(range(1, len(ys) + 1), ys, marker="o", label=f"{r['dataset']}: {r['exposure']}")
    ax.set_xlabel("quintile of corrected retinal age gap (1 = youngest-looking)")
    ax.set_ylabel("prevalence (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "prevalence_by_gap_quintile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Retinal age gap vs disease, per dataset, patient-level.")
    p.add_argument("--predictions", required=True,
                   help="predictions_pooled.csv (summarize_retinal_age.py) or a run's *_predictions.csv")
    p.add_argument("--condition", default=None, help="Pooled file only: which condition (default student).")
    p.add_argument("--brset-csv", default=None, help="Raw BRSET label CSV: joins the ophthalmic flags.")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--gap-col", default=None, help="Force one gap column for every dataset.")
    p.add_argument("--include-ungradable", action="store_true")
    p.add_argument("--datasets", nargs="+", default=None, help="Subset of dataset names (default all).")
    p.add_argument("--out", default=None, help="Output dir (default: next to the predictions file).")
    p.add_argument("--plots", action="store_true", help="Prevalence-by-quintile PNG (needs matplotlib).")
    args = p.parse_args()

    df, cond = load_predictions(args.predictions, args.condition)
    joined: List[str] = []
    if args.brset_csv:
        df, joined = attach_brset_flags(df, args.brset_csv, args.image_ext)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), "associations")
    os.makedirs(out_dir, exist_ok=True)

    L = [f"# Retinal age gap vs disease", f"predictions: `{args.predictions}`"
         + (f" (condition `{cond}`)" if cond else "")
         + (f"; BRSET flags joined from `{args.brset_csv}` ({len(joined)} columns)" if joined else "")]
    all_rows: List[Dict[str, object]] = []
    names = args.datasets or list(pd.unique(df["dataset"]))
    for name in names:
        sub = df[df["dataset"] == name]
        if sub.empty:
            L.append(f"\n## {name}: no rows")
            continue
        Ld, rows = analyse_dataset(sub, str(name), args.gap_col, joined, args.include_ungradable)
        L += Ld
        all_rows += rows
    L.append("\nReading guide: the exposed-minus-reference Δ in years is the effect size; the adjusted Δ "
             "removes what age, age², sex and camera explain. A within-'diabetes' row compares diabetics with "
             "vs without the exposure. Prevalence by quintile is the 'prevalence vs retinal age' view; the OR "
             "per +5 y is its one-number summary. One cohort per dataset, no causal claim.")
    text = "\n".join(L)
    print(text)
    md = os.path.join(out_dir, "age_gap_associations.md")
    with open(md, "w") as f:
        f.write(text + "\n")
    csv = os.path.join(out_dir, "age_gap_associations.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "quintiles"} for r in all_rows]).to_csv(csv, index=False)
    print(f"\n[info] wrote {md}\n[info] wrote {csv}")
    if args.plots:
        png = maybe_plots(all_rows, out_dir)
        print(f"[info] wrote {png}" if png else "[warn] --plots: matplotlib not installed or nothing to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
