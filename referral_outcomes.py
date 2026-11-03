#!/usr/bin/env python3
"""
referral_outcomes.py
====================
What a screening programme would see per 1,000 people screened, from a finished
RetinaReach run's saved predictions -- no GPU, no retraining, seconds per seed::

    python referral_outcomes.py --dir exp_rr_canon            # a sweep (seed*/) or one run dir

Per seed x device x protocol x rule, decided PER PATIENT:

* **refer** if any gradable eye scores at or above the shipped threshold (fixed on
  the home camera's validation split: what the device ships with);
* **retake photo** if no eye of the patient is gradable -- the dataset's own
  quality grade (BRSET quality, mBRSET final_quality, ODIR's keywords, Messidor-2
  adjudicated_gradable) stands in for the device's quality check;
* **truth** = referable in any eye.

Rules: ``dr_referable`` (ICDR >= 2) and ``dr_or_edema`` (referable DR or macular
edema in either eye; refer if either head fires) -- the closer match to IDx-DR's
"more than mild DR" endpoint, whose pivotal trial pre-set sensitivity > 85 % and
specificity > 82.5 % (``--target-sens`` / ``--target-spec``).

Per 1,000 screened, at the device's own prevalence: correct referrals, missed
cases, unnecessary referrals, retakes (and how many of the retaken patients were
referable). Sensitivity and specificity per patient with a patient-bootstrap 95 %
CI. The same misses and unnecessary referrals re-expressed at fixed prevalences
(``--prevalence``) put every camera at one disease rate:
missed = 1000 * prev * (1 - sens), unnecessary = 1000 * (1 - prev) * (1 - spec).

Outputs: ``<dir>/referral_outcomes.csv`` (one row per seed x device x protocol x
rule) and a printed mean-over-seeds table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_model import load_table, target_vector  # noqa: E402

RULES = ("dr_referable", "dr_or_edema")


def _ci(fn, patients: pd.DataFrame, n_boot: int, seed: int) -> List[float]:
    """95 % CI of ``fn(patients resample)`` (one row per patient, drawn with replacement)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(patients.iloc[rng.integers(0, len(patients), len(patients))])
        if v == v:
            vals.append(v)
    return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals
            else [float("nan")] * 2)


def patient_outcomes(eyes: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                     prevalences: Sequence[float] = ()) -> Dict[str, object]:
    """``eyes``: one row per image with ``patient``, ``gradable`` (bool), ``flag``
    (bool: at or above threshold) and ``truth`` (1 / 0 / NaN). Returns the
    per-patient counts, sensitivity / specificity with CIs, and the per-1,000
    figures (at the observed prevalence and at each of ``prevalences``)."""
    e = eyes.assign(flag=eyes["flag"].astype(bool) & eyes["gradable"].astype(bool))
    per = e.groupby("patient").agg(gradable=("gradable", "any"), refer=("flag", "any"),
                                   truth=("truth", "max"))
    # A patient with no gradable photo is a retake whether or not a grade exists
    # (Messidor-2's ungradable photos carry none); a patient WITH a result but no
    # grade cannot be scored and leaves the denominator.
    keep = ~per["gradable"] | per["truth"].notna()
    excluded = int((~keep).sum())
    per = per[keep]
    n = len(per)
    out: Dict[str, object] = {"patients": n, "unknown_truth": excluded}
    if n == 0:
        return out
    retake = ~per["gradable"]
    res = per[~retake]
    y, r = res["truth"].astype(int), res["refer"]
    tp, fn = int((r & (y == 1)).sum()), int((~r & (y == 1)).sum())
    fp, tn = int((r & (y == 0)).sum()), int((~r & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    k = 1000.0 / n
    out.update(prevalence=float(per["truth"].mean()), retakes=int(retake.sum()),
               referable_in_retakes=int((per.loc[retake, "truth"] == 1).sum()), tp=tp, fn=fn, fp=fp, tn=tn,
               sensitivity=sens, specificity=spec,
               per1000_correct_referrals=tp * k, per1000_missed=fn * k,
               per1000_unnecessary=fp * k, per1000_correct_clear=tn * k,
               per1000_retakes=float(retake.sum()) * k)

    def _sens(p):
        p = p[p["gradable"]]
        pos = p["truth"] == 1
        return float(p.loc[pos, "refer"].mean()) if pos.any() else float("nan")

    def _spec(p):
        p = p[p["gradable"]]
        neg = p["truth"] == 0
        return float((~p.loc[neg, "refer"]).mean()) if neg.any() else float("nan")
    out["sens_ci"] = _ci(_sens, per, n_boot, seed)
    out["spec_ci"] = _ci(_spec, per, n_boot, seed + 1)
    for prev in prevalences:
        tag = f"{prev:g}"
        out[f"per1000_missed_at_{tag}"] = 1000.0 * prev * (1 - sens) if sens == sens else float("nan")
        out[f"per1000_unnecessary_at_{tag}"] = 1000.0 * (1 - prev) * (1 - spec) if spec == spec else float("nan")
    return out


def eye_table(pred: pd.DataFrame, table: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
    """Per image: gradable, flag (at or above the shipped threshold) and truth for
    ``rule``; None when a needed score or threshold is missing."""
    need = {"dr_referable": ["dr_referable"], "dr_or_edema": ["dr_referable", "edema"]}[rule]
    if any(f"p_{t}" not in pred or f"threshold_{t}" not in pred for t in need):
        return None
    if any(pred[f"threshold_{t}"].isna().all() for t in need):
        return None
    df = pred.merge(table.drop(columns=[c for c in ("patient",) if c in table.columns]),
                    on="file", how="left", validate="many_to_one")
    q = target_vector(df, "quality") if "final_quality" in df.columns else np.full(len(df), np.nan)
    flag = np.zeros(len(df), dtype=bool)
    truth = np.full(len(df), np.nan)
    for t in need:
        flag |= (df[f"p_{t}"] >= df[f"threshold_{t}"]).to_numpy()
        yt = target_vector(df, t)
        truth = np.where(yt == 1, 1.0, np.where(np.isnan(truth), yt, truth))
    return pd.DataFrame({"patient": df["patient"].astype(str), "gradable": q != 0,
                         "flag": flag, "truth": truth})


def devices_of(results: dict) -> List[Tuple[str, str, str]]:
    """(device, root, dataset) for every device a run scored: the two validated
    cameras, each --unfamiliar camera and the held-out source camera."""
    a = results["args"]
    out = [(a["source_name"], a["source_root"], a["source_dataset"]),
           (a["target_name"], a["target_root"], a["target_dataset"])]
    for spec in a.get("unfamiliar") or []:
        out.append((spec[2] if len(spec) == 3 else spec[1], spec[0], spec[1]))
    h = results.get("holdout_camera")
    if h:
        out.append((h["name"], a["source_root"], a["source_dataset"]))
    return out


def run_dirs(d: str) -> List[Tuple[int, str]]:
    """(seed, run dir): ``d`` itself when it holds results.json, else its seed<n>/."""
    if os.path.isfile(os.path.join(d, "results.json")):
        with open(os.path.join(d, "results.json")) as f:
            return [(int(json.load(f)["args"]["seed"]), d)]
    out = []
    for f in sorted(glob.glob(os.path.join(d, "seed*", "results.json"))):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(os.path.dirname(f)))
        if m:
            out.append((int(m.group(1)), os.path.dirname(f)))
    return out


def run(d: str, n_boot: int = 1000, prevalences: Sequence[float] = (0.055, 0.156),
        target_sens: float = 0.85, target_spec: float = 0.825) -> pd.DataFrame:
    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    rows = []
    for seed, base in run_dirs(d):
        with open(os.path.join(base, "results.json")) as f:
            results = json.load(f)
        for device, root, ds in devices_of(results):
            for proto in ("trained", "calibrated"):
                p = os.path.join(base, f"predictions_{device}_{proto}.csv")
                if not os.path.exists(p):
                    continue
                if (root, ds) not in tables:
                    tables[(root, ds)] = load_table(root, ds)[0]
                pred = pd.read_csv(p)
                for rule in RULES:
                    eyes = eye_table(pred, tables[(root, ds)], rule)
                    if eyes is None:
                        continue
                    o = patient_outcomes(eyes, n_boot=n_boot, seed=seed, prevalences=prevalences)
                    o.update(seed=seed, device=device, protocol=proto, rule=rule,
                             meets_targets=bool(o.get("sensitivity", 0) >= target_sens
                                                and o.get("specificity", 0) >= target_spec))
                    rows.append(o)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-1,000 referral outcomes of a RetinaReach run.")
    p.add_argument("--dir", default="exp_retinareach", help="a sweep (seed*/) or a single run dir")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--prevalence", type=float, nargs="*", default=[0.055, 0.156],
                   help="fixed prevalences to re-express misses and unnecessary referrals at "
                        "(default: BRSET tabletop 5.5 %%, mBRSET handheld 15.6 %%)")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--target-spec", type=float, default=0.825)
    a = p.parse_args(argv)
    df = run(a.dir, a.n_boot, a.prevalence, a.target_sens, a.target_spec)
    if df.empty:
        raise SystemExit(f"[fatal] no predictions_<device>_<protocol>.csv under {a.dir} (or its seed*/)")
    df.to_csv(os.path.join(a.dir, "referral_outcomes.csv"), index=False)
    cols = ["prevalence", "sensitivity", "specificity", "per1000_correct_referrals", "per1000_missed",
            "per1000_unnecessary", "per1000_retakes"]
    cols += [c for c in df.columns if c.startswith(("per1000_missed_at_", "per1000_unnecessary_at_"))]
    g = df.groupby(["rule", "device", "protocol"], sort=False)
    tab = g[cols].mean()
    tab.insert(0, "seeds", g["seed"].nunique())
    tab.insert(1, "patients", g["patients"].max())
    tab["seeds_meeting_targets"] = g["meets_targets"].sum().astype(int)
    print(f"=== referral outcomes per patient, shipped threshold (mean over seeds; targets "
          f"sens >= {a.target_sens}, spec >= {a.target_spec}) ===")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) <= 1 else f"{v:.1f}"))
    print("\nper1000_*: out of every 1,000 patients screened on that camera. *_at_<prev>: the same "
          "sensitivity and specificity at a fixed prevalence, so cameras compare at one disease rate.")
    print(f"\nwrote {a.dir}/referral_outcomes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
