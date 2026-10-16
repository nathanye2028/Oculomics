#!/usr/bin/env python3
"""
brset_comorbidities.py
======================
BRSET's free-text ``comorbidities`` column -> **one 0/1/NaN flag per systemic
condition, per patient**, plus the audit that decides whether those flags can be
trained on.

Why this exists
---------------
mBRSET ships its systemic labels as clean yes/no columns (hypertension,
nephropathy, MI, ...), but every mBRSET patient is diabetic and every photo is
handheld. BRSET -- tabletop, diabetic AND non-diabetic -- has no such columns;
its systemic history lives in one self-reported free-text field. Scanned on the
lab box (2026-10-15): of 8,524 patients, 4,330 are ``na`` (never recorded),
1,437 are ``0`` (recorded, none), and the rest list conditions such as
"hypertension , hypothyroidism" -- hypertension alone for ~1,745 patients, the
largest systemic label source available to this project.

    python brset_comorbidities.py --root <BRSET root> --out exp_comorbid

Three rules that matter more than the vocabulary
------------------------------------------------
``na`` is missing, never negative.
    A patient whose history was not taken is not a patient without
    hypertension. Every flag is NaN for an unrecorded patient, so a trainer's
    ``drop_missing_labels`` removes them instead of learning "no history taken"
    as "healthy".

A recorded list is read as complete.
    "hypertension" with no mention of heart disease becomes heart_disease = 0.
    That is the only way to get negatives out of a free-text list, and it is
    also the main source of label noise -- the audit's diabetes cross-check
    (free text vs BRSET's own ``diabetes`` column) estimates how incomplete the
    lists are.

Missingness and camera must be checked before any image model.
    BRSET mixes two cameras (Canon CR, Nikon NF5050). If histories were taken
    mostly on one camera's patients, or a condition is recorded more often on
    one camera after adjusting for age, an image model can score well by
    recognising the camera. The audit fits small logistic regressions
    (camera + age + sex + diabetes) for "history recorded" and for every target,
    and flags any target whose camera odds ratio excludes 1.

Outputs (``--out``)
-------------------
``brset_comorbidity_flags.csv``  one row per patient: covariates, the raw text,
                                 ``atom_<condition>`` and ``cm_<target>`` flags
``comorbidity_audit.md``         the human-readable audit (also printed)
``comorbidity_audit.json``       the same numbers for summarisers

``unmapped`` items are listed in the audit: anything there is a condition the
vocabulary below does not know yet. Add a pattern to :data:`ATOMS` rather than
editing the CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
# (atom, include pattern, exclude pattern). Matched per list item, after the
# string is lower-cased and split on , ; / + and the words "and" / "e". Written
# against the items seen on the real CSV (lab box, 2026-10-15 and the first
# audit's unmapped list), misspellings and Portuguese spellings included
# ("hydroxichloroquine", "chron disease", "parkison", "hipoitireoidismo",
# "leucemia", "dpoc").
ATOMS: List[Tuple[str, str, Optional[str]]] = [
    # -- cardiovascular / metabolic
    ("hypertension", r"hypertens|\bsah\b|high blood pressure|\bhtn\b",
     r"ocular|intraocular|pulmonary|intracranial|portal"),     # not systemic BP
    ("ocular_hypertension", r"(?:ocular|intraocular) hypertens", None),
    ("diabetes", r"diabet|\bdm\b|\bdm ?[12]\b", r"insipidus"),
    ("diabetes_type1", r"diabetes ?(?:type ?)?1\b|\bdm ?1\b|type ?1 diab", None),
    ("dyslipidemia", r"cholesterol|colesterol|lipid|triglycerid", None),
    ("myocardial_infarction", r"myocard|infarct|heart attack", r"cerebr|encephal|brain|retin"),
    ("heart_failure", r"cardiac insufficien|heart failure|cardiac failure|heart insufficien", None),
    ("arrhythmia", r"arr?h?ythm|atrial fibrillation|pacemaker", None),
    ("valve_disease", r"valv", None),
    # "catheterism" = cateterismo, a cardiac catheterisation in Brazilian charts
    ("cardiopathy", r"cardiopath|cardiomyopath|heart disease|coronar|angina|ischemic heart"
                    r"|catheteri|cateteris", None),
    ("stroke", r"stroke|vascular accident|\bcva\b|\bavc\b|cerebrovascular"
               r"|cerebral (?:infarct|isch|hemorr)", None),
    ("thrombosis", r"thromb|trombos|embolism", r"thrombocyt"),
    ("aneurysm", r"aneurys|aneuris", None),
    ("obesity", r"obes", None),
    ("smoking", r"tabag|smok", None),
    # -- kidney / liver
    ("kidney_disease", r"kidney|renal|nephr|dialysis", r"transplant"),
    ("kidney_transplant", r"(?:kidney|renal).*transplant|transplant.*(?:kidney|renal)", None),
    ("liver_disease", r"hepatic|liver|cirrho", None),
    # -- brain / nervous system
    ("alzheimer", r"alzheimer|dementia", None),
    ("parkinson", r"park[a-z]*son", None),
    ("multiple_sclerosis", r"multiple sclerosis|sclerosis multiple", None),
    ("neuromyelitis_optica", r"devic|neuromyelit", None),
    ("epilepsy", r"epilep|seizure|convuls", None),
    ("encephalitis", r"encephalit|meningit", None),
    ("hydrocephalus", r"hydroceph|hidroce", None),
    ("intracranial_hypertension", r"intracranial hypertens|pseudotumor", None),
    ("brain_tumor", r"meningioma|glioma|astrocytoma|brain tumo", None),
    ("pituitary_adenoma", r"hypophys|pituitar|prolactinoma", None),
    ("cerebral_palsy", r"cerebral palsy", None),
    ("migraine", r"migraine", None),
    # -- immune / drugs with retinal effects
    ("lupus", r"lupus|\bsle\b", None),
    ("rheumatoid_arthritis", r"rheumatoid", None),
    ("sjogren", r"sj.?gren", None),
    ("psoriasis", r"psoria", None),
    ("vasculitis", r"vasculit|arteritis|beh.?et", None),
    ("crohn", r"cro?hn|chron disease|colitis", None),
    ("autoimmune_other", r"sarcoid|spondyl|vitiligo|myasthen|scleroderm|dermatomyosit", None),
    ("antimalarial_use", r"chloroqui|hydroxi?chloroq|plaquenil", None),
    # -- other
    ("hypothyroidism", r"hypothyr|hipoi?tireo|hashimoto", None),
    ("hyperthyroidism", r"hyperthyr|graves", None),
    ("thyroid_other", r"thyroidit|goiter|thyr\w* nodul", None),
    ("sickle_cell", r"sickle|falciform", None),
    ("anemia", r"anemi|anaemi", r"sickle"),
    ("blood_disorder", r"thalass|talass|polic[iy]tem|polycythem|hemophil|thrombocyt", None),
    ("hiv", r"immunodeficiency virus|\bhiv\b|\baids\b", None),
    ("chronic_infection", r"chagas|syphil|hepatitis|tubercul|toxoplasm|hansen|lepros", None),
    ("cancer", r"cancer|carcinom|lymphom|leukemi|leukaemi|leucemi|melanom|neoplas|tumou?r|metasta",
     r"hypophys|pituitar|prolactinoma|meningioma|glioma|astrocytoma|brain tumo"),
    ("asthma_copd", r"asthma|copd|\bdpoc\b|obstructive pulmonary|bronch|emphysem", None),
    ("osteoarticular", r"arthros|osteopor|osteoarth|gout|fibromi?y?alg|(?<!rheumatoid )arthritis",
     None),
    ("prostate", r"prosta", None),
    ("psychiatric", r"depress|anxiet|bipolar|schizo|psych", None),
    ("rhinitis_allergy", r"rhinit|allerg", None),
    ("genetic_syndrome", r"down syndrome|trisom|mccune|muscular dystroph|marfan|turner|neurofibromat", None),
    # eye conditions listed as "comorbidities": kept out of the unmapped list,
    # never used as a systemic target
    ("ocular_condition", r"glaucom|catarac|uveit|keratocon|myop|retin|macul|amblyop|strabism|blind"
                         r"|cone dystroph|rod dystroph|albinis|stargardt|nystagm", None),
]
_COMPILED = [(name, re.compile(inc), re.compile(exc) if exc else None) for name, inc, exc in ATOMS]
ATOM_NAMES = [a[0] for a in ATOMS]

# Targets = OR over atoms. These are what a trainer or the age-gap analysis
# reads (``cm_<name>``); the atoms stay in the CSV so a grouping can be audited.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "hypertension": ("hypertension",),
    "diabetes": ("diabetes", "diabetes_type1"),
    "dyslipidemia": ("dyslipidemia",),
    "heart_disease": ("myocardial_infarction", "heart_failure", "arrhythmia",
                      "valve_disease", "cardiopathy"),
    "myocardial_infarction": ("myocardial_infarction",),
    "stroke": ("stroke",),
    "cardiovascular_event": ("myocardial_infarction", "heart_failure", "arrhythmia",
                             "valve_disease", "cardiopathy", "stroke"),
    "kidney": ("kidney_disease", "kidney_transplant"),
    # neurodegenerative + demyelinating (MS, neuromyelitis optica)
    "neurodegenerative": ("alzheimer", "parkinson", "multiple_sclerosis", "neuromyelitis_optica"),
    "alzheimer": ("alzheimer",),
    # any structural / vascular / degenerative brain condition; migraine excluded
    "brain_any": ("alzheimer", "parkinson", "multiple_sclerosis", "neuromyelitis_optica", "stroke",
                  "epilepsy", "encephalitis", "hydrocephalus", "intracranial_hypertension",
                  "brain_tumor", "pituitary_adenoma", "cerebral_palsy"),
    "autoimmune": ("lupus", "rheumatoid_arthritis", "sjogren", "psoriasis", "vasculitis", "crohn",
                   "autoimmune_other"),
    "antimalarial_use": ("antimalarial_use",),
    "thyroid": ("hypothyroidism", "hyperthyroidism", "thyroid_other"),
}

MISSING_TOKENS = {"", "na", "n/a", "nan", "none recorded", "not informed", "-"}
NONE_TOKENS = {"0", "none", "no", "nothing", "denies", "without"}
_SPLIT = re.compile(r"[,;/+]|\band\b|\be\b")

# A 20% patient-level test split needs this many positive patients before an
# AUROC on it should be headlined -- the same bar metadata_model.py uses.
MIN_TEST_POS = 20
TEST_FRAC = 0.20


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def normalise(text) -> str:
    """Lower-case, collapse whitespace; anything non-string (NaN) -> ''."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_comorbidities(text) -> Tuple[Optional[Set[str]], List[str]]:
    """One free-text cell -> (atoms present, unmapped items).

    Returns ``(None, [])`` for a missing cell -- "not recorded", which every
    caller must keep distinct from ``(set(), [])`` = "recorded, none listed".
    """
    t = normalise(text)
    if t in MISSING_TOKENS:
        return None, []
    if t in NONE_TOKENS:
        return set(), []
    atoms: Set[str] = set()
    unmapped: List[str] = []
    for item in _SPLIT.split(t):
        item = item.strip(" .-:")
        if not item or item in NONE_TOKENS:
            continue
        hit = {name for name, inc, exc in _COMPILED
               if inc.search(item) and not (exc and exc.search(item))}
        if hit:
            atoms |= hit
        else:
            unmapped.append(item)
    return atoms, unmapped


def _yes_no(v) -> float:
    t = normalise(v)
    if t in ("yes", "y", "1", "true", "sim"):
        return 1.0
    if t in ("no", "n", "0", "false", "nao", "não"):
        return 0.0
    return np.nan


def _sex01(sex: Optional[pd.Series]) -> float:
    """BRSET sex 1 = male / 2 = female -> 1 / 0, the rule brset_dataset uses
    (1 = male in every dataset). Applied only to 1/2, so 0/1 passes through."""
    if sex is None or not sex.notna().any():
        return np.nan
    v = float(sex.dropna().iloc[0])
    return {1.0: 1.0, 2.0: 0.0}.get(v, v)


def read_brset_csv(path: str) -> pd.DataFrame:
    """Read every column as text so "NA" / "None" stay literal strings for the
    parser to classify (pandas would otherwise turn some, not all, into NaN),
    and strip the UTF-8 BOM the PhysioNet CSV carries on its first header."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    return df


def patient_flags(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Counter, Dict[str, int]]:
    """Image-level BRSET rows -> one row per patient.

    A patient is ``recorded`` if ANY of their images carries a non-missing
    history; atoms are OR-ed over images. Patients whose images disagree are
    counted (``inconsistent``) rather than silently resolved.
    """
    need = ["patient_id", "comorbidities"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"BRSET CSV lacks {missing}; columns present: {sorted(raw.columns)}")

    unmapped: Counter = Counter()
    stats = {"inconsistent_history": 0, "mixed_camera": 0}
    rows = []
    for pid, g in raw.groupby("patient_id", sort=False):
        parsed = [parse_comorbidities(v) for v in g["comorbidities"]]
        recorded = [a for a, _ in parsed if a is not None]
        for _, u in parsed:
            unmapped.update(u)
        texts = sorted({normalise(v) for v in g["comorbidities"]} - MISSING_TOKENS)
        if len(texts) > 1:
            stats["inconsistent_history"] += 1

        cams = sorted({c.strip() for c in g.get("camera", pd.Series(dtype=str)) if c.strip()})
        if len(cams) > 1:
            stats["mixed_camera"] += 1
        age = pd.to_numeric(g.get("patient_age"), errors="coerce") if "patient_age" in g else None
        sex = pd.to_numeric(g.get("patient_sex"), errors="coerce") if "patient_sex" in g else None
        dr = pd.to_numeric(g.get("DR_ICDR"), errors="coerce") if "DR_ICDR" in g else None
        dm = [_yes_no(v) for v in g["diabetes"]] if "diabetes" in g else [np.nan]
        ins = [_yes_no(v) for v in g["insuline"]] if "insuline" in g else [np.nan]

        row = {
            "patient_id": pid,
            "n_images": len(g),
            "camera": cams[0] if len(cams) == 1 else ("mixed" if cams else ""),
            "age": float(np.nanmedian(age)) if age is not None and age.notna().any() else np.nan,
            "sex": _sex01(sex),
            "diabetes_col": float(np.nanmax(dm)) if not np.all(np.isnan(dm)) else np.nan,
            "insulin": float(np.nanmax(ins)) if not np.all(np.isnan(ins)) else np.nan,
            "dr_worst": float(dr.max()) if dr is not None and dr.notna().any() else np.nan,
            "comorbidities_raw": " || ".join(texts),
            "recorded": float(bool(recorded)),
        }
        present = set().union(*recorded) if recorded else set()
        for a in ATOM_NAMES:
            row[f"atom_{a}"] = (float(a in present) if recorded else np.nan)
        for gname, members in GROUPS.items():
            row[f"cm_{gname}"] = (float(bool(present & set(members))) if recorded else np.nan)
        rows.append(row)
    return pd.DataFrame(rows), unmapped, stats


# mBRSET's structured yes/no columns that mean the same as a BRSET target, so a
# model fit on BRSET can be scored on mBRSET's clean labels. Hypertension only:
# the first metadata run (lab box, 2026-10-16) scored BRSET-fit kidney models at
# 0.38-0.43 on mBRSET nephropathy (BRSET's kidney patients are young transplant /
# CKD cases, not diabetic nephropathy) and MI at 0.44-0.56 from 24 BRSET cases.
MBRSET_EQUIVALENTS: Dict[str, str] = {
    "cm_hypertension": "systemic_hypertension",
}


def attach_flags(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Add the patient-level ``cm_<target>`` columns to an image-level BRSET table
    in the mBRSET schema (``brset_dataset.load_brset``: ``patient`` column).

    Joined on the patient id as text (load_brset reads it as int, the parser as
    str). A join that matches fewer than 99% of rows raises instead of filling
    the rest with NaN, which would look exactly like "history not recorded".
    """
    raw = read_brset_csv(csv_path)
    if "comorbidities" not in raw.columns:
        print(f"[warn] {csv_path} has no comorbidities column; no cm_* targets", file=sys.stderr)
        return df
    pf, _, _ = patient_flags(raw)
    cols = [f"cm_{g}" for g in GROUPS]
    lut = pf.set_index(pf["patient_id"].astype(str))[cols]
    key = df["patient"].astype(str)
    matched = float(key.isin(lut.index).mean()) if len(key) else 1.0
    if matched < 0.99:
        raise ValueError(f"comorbidity join matched only {matched:.1%} of BRSET rows on patient id; "
                         f"e.g. table {list(key.head(3))} vs parser {list(lut.index[:3])}")
    out = df.copy()
    joined = lut.reindex(key.to_numpy())
    for c in cols:
        out[c] = joined[c].to_numpy()
    return out


def mbrset_equivalents(df: pd.DataFrame) -> pd.DataFrame:
    """Copy mBRSET's structured columns under the BRSET target names."""
    out = df.copy()
    for cm, col in MBRSET_EQUIVALENTS.items():
        if col in out.columns:
            out[cm] = out[col]
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def logit_or(y: np.ndarray, X: np.ndarray, names: Sequence[str],
             max_iter: int = 50) -> Dict[str, Dict[str, float]]:
    """Unpenalised logistic regression by Newton-Raphson, Wald 95% CIs.

    numpy only (no statsmodels on either machine). An intercept is added. A
    tiny ridge keeps the Hessian invertible; ``stable`` is False when a
    coefficient runs away (quasi-separation), and such ORs should not be read.
    """
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X1.shape[1])
    converged = False
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
        W = p * (1 - p)
        H = X1.T @ (X1 * W[:, None]) + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(H, X1.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
    cov = np.linalg.inv(X1.T @ (X1 * (p * (1 - p))[:, None]) + 1e-8 * np.eye(len(b)))
    se = np.sqrt(np.diag(cov))
    stable = converged and bool(np.all(np.abs(b) < 15))
    out = {}
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):   # unstable fits: exp(+-huge)
        for i, n in enumerate(names, start=1):
            out[n] = {"or": float(np.exp(b[i])), "lo": float(np.exp(b[i] - 1.96 * se[i])),
                      "hi": float(np.exp(b[i] + 1.96 * se[i])), "stable": stable}
    return out


def _design(pf: pd.DataFrame, include_diabetes: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Nikon indicator + age per decade + sex (+ diabetes) for single-camera
    patients with complete covariates. The reference camera is the more common one."""
    cams = pf["camera"][~pf["camera"].isin(["", "mixed"])]
    if cams.nunique() < 2:
        return pf.iloc[0:0], []
    ref, other = cams.value_counts().index[:2]
    d = pf[pf["camera"].isin([ref, other])].copy()
    d["cam_other"] = (d["camera"] == other).astype(float)
    d["age_per10"] = d["age"] / 10.0
    cols = ["cam_other", "age_per10", "sex"] + (["diabetes_col"] if include_diabetes else [])
    d = d.dropna(subset=cols)
    d.attrs["camera_contrast"] = f"{other} vs {ref}"
    return d, cols


def _age_band(a: float) -> str:
    if np.isnan(a):
        return "unknown"
    return "<40" if a < 40 else "40-59" if a < 60 else "60-69" if a < 70 else "70+"


AGE_BANDS = ["<40", "40-59", "60-69", "70+", "unknown"]


def _rate_table(pf: pd.DataFrame, by: pd.Series, value: str) -> List[Dict[str, object]]:
    out = []
    groups = dict(list(pf.groupby(by, sort=True)))
    order = [b for b in AGE_BANDS if b in groups] if set(groups) <= set(AGE_BANDS) else list(groups)
    for level in order:
        g = groups[level]
        v = g[value].dropna()
        out.append({"level": str(level), "n": int(len(v)),
                    "rate": float(v.mean()) if len(v) else float("nan")})
    return out


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def audit(pf: pd.DataFrame, unmapped: Counter, stats: Dict[str, int],
          top_unmapped: int = 100) -> Dict[str, object]:
    """Every number the report prints, as a JSON-able dict."""
    rec = pf[pf["recorded"] == 1]
    none_listed = int((rec[[f"atom_{a}" for a in ATOM_NAMES]].sum(axis=1) == 0).sum())
    res: Dict[str, object] = {
        "patients": int(len(pf)),
        "recorded": int(len(rec)),
        "not_recorded": int((pf["recorded"] == 0).sum()),
        "recorded_none_listed": none_listed,
        "inconsistent_history": stats["inconsistent_history"],
        "mixed_camera": stats["mixed_camera"],
        "cameras": {str(k): int(v) for k, v in pf["camera"].value_counts().items()},
        "unmapped_items": int(sum(unmapped.values())),
        "unmapped_top": unmapped.most_common(top_unmapped),
    }

    # -- 1. who has a recorded history at all
    dr_band = pf["dr_worst"].map(lambda g: "unknown" if np.isnan(g) else "0" if g == 0
                                 else "1" if g == 1 else "2+")
    res["missingness"] = {
        "camera": _rate_table(pf, pf["camera"].replace("", "unknown"), "recorded"),
        "age_band": _rate_table(pf, pf["age"].map(_age_band), "recorded"),
        "sex": _rate_table(pf, pf["sex"].map(lambda s: "unknown" if np.isnan(s) else f"{int(s)}"),
                           "recorded"),
        "diabetes_col": _rate_table(pf, pf["diabetes_col"].map(
            lambda s: "unknown" if np.isnan(s) else ("yes" if s else "no")), "recorded"),
        "dr_worst": _rate_table(pf, dr_band, "recorded"),
    }
    d, cols = _design(pf)
    if len(d) and d["recorded"].nunique() == 2:
        res["missingness_adjusted"] = {
            "n": int(len(d)), "contrast": d.attrs["camera_contrast"],
            "or": logit_or(d["recorded"].to_numpy(), d[cols].to_numpy(), cols)}

    # -- 2. does the free text agree with BRSET's own diabetes column?
    both = rec.dropna(subset=["diabetes_col"])
    def _n(text: float, col: float) -> int:
        return int(((both["cm_diabetes"] == text) & (both["diabetes_col"] == col)).sum())
    res["diabetes_crosscheck"] = {
        "n": int(len(both)),
        "text_yes_col_yes": _n(1, 1), "text_yes_col_no": _n(1, 0),
        "text_no_col_yes": _n(0, 1), "text_no_col_no": _n(0, 0),
    }

    # -- 3. per target: size, age gap, camera association
    targets = {}
    for gname in GROUPS:
        col = f"cm_{gname}"
        v = rec[col]
        pos, neg = int((v == 1).sum()), int((v == 0).sum())
        t: Dict[str, object] = {
            "pos": pos, "neg": neg, "prev": pos / max(pos + neg, 1),
            "exp_test_pos": pos * TEST_FRAC,
            "underpowered": pos * TEST_FRAC < MIN_TEST_POS,
            "age_pos": float(rec.loc[v == 1, "age"].mean()) if pos else float("nan"),
            "age_neg": float(rec.loc[v == 0, "age"].mean()) if neg else float("nan"),
            "prev_by_camera": {str(c): float(g[col].mean()) for c, g in rec.groupby("camera")
                               if c not in ("", "mixed") and len(g)},
        }
        dt, tcols = _design(rec, include_diabetes=(gname != "diabetes"))
        if pos >= 10 and len(dt) and dt[col].nunique() == 2:
            fit = logit_or(dt[col].to_numpy(), dt[tcols].to_numpy(), tcols)
            cam = fit["cam_other"]
            t["camera_or"] = cam
            t["camera_flag"] = bool(cam["stable"] and (cam["lo"] > 1 or cam["hi"] < 1))
        targets[gname] = t
    res["targets"] = targets
    return res


def _fmt_rates(rows: List[Dict[str, object]]) -> str:
    return ", ".join(f"{r['level']} {100 * r['rate']:.0f}% (n={r['n']})" for r in rows)


def render(res: Dict[str, object], csv_path: str) -> str:
    """Markdown report. Verdict lines are written to be read by a person, not parsed."""
    L = [f"# BRSET comorbidity audit", "", f"Source: `{csv_path}`", ""]
    P = res["patients"]
    L += ["## Coverage", "",
          f"- patients: {P}",
          f"- history recorded: {res['recorded']} ({100 * res['recorded'] / P:.1f}%); "
          f"not recorded (`na`/blank): {res['not_recorded']}",
          f"- recorded but no condition listed: {res['recorded_none_listed']}",
          f"- patients whose images carry different histories: {res['inconsistent_history']}",
          f"- patients photographed on more than one camera: {res['mixed_camera']}",
          f"- cameras (patients): {res['cameras']}",
          f"- unmapped list items: {res['unmapped_items']}"]
    if res["unmapped_top"]:
        L.append("  - top unmapped (add to ATOMS if they are conditions): "
                 + ", ".join(f"{t} ({n})" for t, n in res["unmapped_top"]))
    L += ["", "## Who has a recorded history", ""]
    for k, rows in res["missingness"].items():
        L.append(f"- by {k}: {_fmt_rates(rows)}")
    L.append("- recorded patients per camera (what every camera contrast below rests on): "
             + ", ".join(f"{r['level']} {round(r['n'] * r['rate'])}"
                         for r in res["missingness"]["camera"]))
    adj = res.get("missingness_adjusted")
    if adj:
        o = adj["or"]["cam_other"]
        L += ["", f"Adjusted odds of a recorded history, {adj['contrast']} "
                  f"(age, sex, diabetes held fixed; n={adj['n']}): "
                  f"OR {o['or']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]"]
        if o["lo"] > 1.5 or o["hi"] < 1 / 1.5:
            L.append("**Missingness depends strongly on camera.** Train and test only on "
                     "recorded patients, and report results within each camera.")
    dc = res["diabetes_crosscheck"]
    if dc["n"]:
        L += ["", "## Free text vs BRSET's `diabetes` column (recorded patients)", "",
              "| | column yes | column no |", "|---|---|---|",
              f"| text lists diabetes | {dc['text_yes_col_yes']} | {dc['text_yes_col_no']} |",
              f"| text does not | {dc['text_no_col_yes']} | {dc['text_no_col_no']} |", ""]
        col_yes = dc["text_yes_col_yes"] + dc["text_no_col_yes"]
        if col_yes and dc["text_yes_col_no"] == 0 and dc["text_no_col_yes"] == 0:
            L.append("Perfect agreement. The column was most likely derived from this same "
                     "field, so this is NOT an independent check of how complete the lists are.")
        elif col_yes:
            L.append(f"Of patients the column calls diabetic, the free text lists diabetes for "
                     f"{100 * dc['text_yes_col_yes'] / col_yes:.0f}%. The rest are the free "
                     f"text's false negatives -- a rough estimate of how incomplete a list is.")
    L += ["", "## Targets (recorded patients only)", "",
          "| target | pos | neg | prev | test pos @20% | mean age pos / neg | prevalence by camera | camera OR (adj.) | |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, t in res["targets"].items():
        cam = t.get("camera_or")
        cam_s = f"{cam['or']:.2f} [{cam['lo']:.2f}, {cam['hi']:.2f}]" if cam else "--"
        if cam and not cam["stable"]:
            cam_s = "n/a (no cases on one camera)"
        notes = []
        if t["underpowered"]:
            notes.append("underpowered: CV / association only")
        if t.get("camera_flag"):
            notes.append("**camera-associated**")
        by_cam = ", ".join(f"{c} {100 * p:.1f}%" for c, p in t["prev_by_camera"].items())
        L.append(f"| {name} | {t['pos']} | {t['neg']} | {100 * t['prev']:.1f}% | "
                 f"{t['exp_test_pos']:.0f} | {t['age_pos']:.1f} / {t['age_neg']:.1f} | "
                 f"{by_cam} | {cam_s} | {'; '.join(notes)} |")
    L += ["", "Camera OR = odds of the condition on the second camera vs the first, "
              "adjusting for age, sex and diabetes. A flagged target can be learned from the "
              "camera alone: report its image AUROC within each camera.",
          "", "A recorded list is read as complete, so negatives include patients whose "
              "condition was simply not written down; see the diabetes cross-check above."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def find_csv(root: str) -> str:
    if os.path.isfile(root):
        return root
    from brset_dataset import LABEL_CSV_NAMES, resolve_root
    d = resolve_root(root, "brset")
    return next(os.path.join(d, n) for n in LABEL_CSV_NAMES["brset"]
                if os.path.isfile(os.path.join(d, n)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Parse BRSET comorbidities and audit them.")
    p.add_argument("--root", required=True, help="BRSET root (or the label CSV itself)")
    p.add_argument("--out", default="exp_comorbid", help="output directory")
    p.add_argument("--top-unmapped", type=int, default=100)
    args = p.parse_args(argv)

    csv_path = find_csv(args.root)
    raw = read_brset_csv(csv_path)
    pf, unmapped, stats = patient_flags(raw)
    res = audit(pf, unmapped, stats, top_unmapped=args.top_unmapped)
    report = render(res, csv_path)

    os.makedirs(args.out, exist_ok=True)
    pf.to_csv(os.path.join(args.out, "brset_comorbidity_flags.csv"), index=False)
    with open(os.path.join(args.out, "comorbidity_audit.md"), "w") as fh:
        fh.write(report)
    with open(os.path.join(args.out, "comorbidity_audit.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(report)
    print(f"[info] wrote {args.out}/brset_comorbidity_flags.csv ({len(pf)} patients), "
          f"comorbidity_audit.md, comorbidity_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
