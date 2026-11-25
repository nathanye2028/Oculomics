#!/usr/bin/env python3
"""
dr_proxy_check.py
=================
Is a systemic probe reading the systemic condition, or diabetic retinopathy as a
stand-in for it? Runs on a finished RetinaReach sweep's saved predictions -- no
GPU, no retraining, seconds per seed::

    python dr_proxy_check.py --dir exp_retinareach

Why ask
-------
The systemic targets are linear probes on a trunk trained for DR and edema, and
in a diabetic cohort insulin use, hypertension and nephropathy all travel with
DR severity. A probe can therefore score above chance without seeing anything
but DR. The fine-tuned per-target insulin model, which never trained on DR,
scored LOWER than the probe on the DR-trained trunk -- consistent with that.

What it reports, per probe target on the handheld test split (calibrated protocol)
----------------------------------------------------------------------------------
auroc_probe         the probe's AUROC for the systemic target (as the gate sees it)
auroc_dr_score      the model's own referable-DR output used as the predictor of
                    the systemic target: what "it is just DR" would score
auroc_dr_grade      the grader's ICDR grade (0-4) used the same way
auroc_no_dr         the probe's AUROC among eyes graded ICDR 0 only
auroc_within_grade  the probe's AUROC counting only pairs of eyes with the SAME
                    ICDR grade (covariate-adjusted AUROC, Janes & Pepe 2008):
                    0.5 = nothing left once DR grade is held fixed
                    (+ patient-cluster bootstrap 95 % CI)

Reading it: ``auroc_dr_score`` close to ``auroc_probe`` and ``auroc_within_grade``
near 0.5 = the probe is a DR proxy. ``auroc_within_grade`` clearly above 0.5 =
image signal beyond the graded retinopathy. ICDR 0 is "no DR at grading
resolution", not "no diabetic change": sub-threshold lesions can still carry it.

Outputs: ``<dir>/dr_proxy.csv`` (one row per seed x target) and a printed
mean-over-seeds table.
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
from metadata_model import _auc, load_table, target_vector  # noqa: E402

GRADE_COL = "final_icdr"


def within_stratum_auroc(y: np.ndarray, p: np.ndarray, strata: np.ndarray) -> float:
    """AUROC over positive/negative pairs that share a stratum: the per-stratum
    AUROCs averaged with weights n_pos * n_neg. NaN if no stratum has both classes."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1 = float((y[m] == 1).sum())
        n0 = float((y[m] == 0).sum())
        if n1 and n0:
            num += _auc(y[m], p[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def _cluster_ci(fn, groups: np.ndarray, n_boot: int, seed: int) -> List[float]:
    """95 % CI of ``fn(index array)`` resampling patients (both eyes move together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        v = fn(np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]))
        if v == v:
            vals.append(v)
    return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals
            else [float("nan")] * 2)


def proxy_rows(pred: pd.DataFrame, table: pd.DataFrame, targets: Sequence[str],
               dr_target: str = "dr_referable", n_boot: int = 1000, seed: int = 0) -> List[dict]:
    """One row per target in ``targets`` with labels of both classes among the
    predicted files. ``pred``: file, patient, p_<target> columns (a
    ``predictions_<device>_<protocol>.csv``); ``table``: the device's label table."""
    df = pred.merge(table.drop(columns=[c for c in ("patient",) if c in table.columns]),
                    on="file", how="left", validate="many_to_one")
    grade = pd.to_numeric(df[GRADE_COL], errors="coerce").to_numpy() if GRADE_COL in df else \
        np.full(len(df), np.nan)
    d = df[f"p_{dr_target}"].to_numpy() if f"p_{dr_target}" in df else np.full(len(df), np.nan)
    g_all = df["patient"].to_numpy()
    rows = []
    for t in targets:
        if f"p_{t}" not in df:
            continue
        y_all, p_all = target_vector(df, t), df[f"p_{t}"].to_numpy()
        ok = ~np.isnan(y_all) & ~np.isnan(p_all)
        y, p, g = y_all[ok].astype(int), p_all[ok], g_all[ok]
        if len(np.unique(y)) < 2:
            continue
        gr, ds = grade[ok], d[ok]
        has_g, has_d = ~np.isnan(gr), ~np.isnan(ds)
        no_dr = has_g & (gr == 0)
        yg, pg, sg, gg = y[has_g], p[has_g], gr[has_g], g[has_g]
        lo, hi = (_cluster_ci(lambda i: within_stratum_auroc(yg[i], pg[i], sg[i]), gg, n_boot, seed)
                  if has_g.any() else [float("nan")] * 2)
        rows.append({
            "target": t, "n": int(ok.sum()), "pos": int(y.sum()),
            "auroc_probe": _auc(y, p),
            "auroc_dr_score": _auc(y[has_d], ds[has_d]) if has_d.any() else float("nan"),
            "auroc_dr_grade": _auc(y[has_g], gr[has_g]) if has_g.any() else float("nan"),
            "n_no_dr": int(no_dr.sum()), "pos_no_dr": int(y[no_dr].sum()),
            "auroc_no_dr": _auc(y[no_dr], p[no_dr]) if no_dr.any() else float("nan"),
            "auroc_within_grade": within_stratum_auroc(yg, pg, sg) if has_g.any() else float("nan"),
            "within_lo": lo, "within_hi": hi})
    return rows


def run(d: str, device: Optional[str], protocol: str, n_boot: int) -> pd.DataFrame:
    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    out = []
    for f in sorted(glob.glob(os.path.join(d, "seed*", "results.json"))):
        base = os.path.dirname(f)
        m = re.fullmatch(r"seed(\d+)", os.path.basename(base))
        if m is None:
            continue
        with open(f) as fh:
            args = json.load(fh)["args"]
        name = device or args["target_name"]
        if name != args["target_name"]:
            raise SystemExit(f"[fatal] {base}: probes are labelled on {args['target_name']!r} only")
        pred_p = os.path.join(base, f"predictions_{name}_{protocol}.csv")
        if not os.path.exists(pred_p):
            print(f"[skip] {base}: no {os.path.basename(pred_p)}")
            continue
        key = (args["target_root"], args["target_dataset"])
        if key not in tables:
            tables[key] = load_table(*key)[0]
        rows = proxy_rows(pd.read_csv(pred_p), tables[key], args["probe_targets"],
                          args["source_targets"][0], n_boot, int(m.group(1)))
        out += [dict(r, seed=int(m.group(1)), device=name) for r in rows]
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Are the systemic probes reading DR as a stand-in?")
    p.add_argument("--dir", default="exp_retinareach")
    p.add_argument("--device", default=None, help="default: each run's handheld device")
    p.add_argument("--protocol", default="calibrated", choices=["calibrated"])
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    df = run(a.dir, a.device, a.protocol, a.n_boot)
    if df.empty:
        raise SystemExit(f"[fatal] no seed*/predictions_*_{a.protocol}.csv with probe labels under {a.dir}")
    df.to_csv(os.path.join(a.dir, "dr_proxy.csv"), index=False)
    cols = ["auroc_probe", "auroc_dr_score", "auroc_dr_grade", "auroc_no_dr", "auroc_within_grade",
            "within_lo", "within_hi"]
    g = df.groupby("target", sort=False)
    tab = g[cols].mean()
    tab.insert(0, "seeds", g["seed"].nunique())
    tab.insert(1, "pos", g["pos"].max())
    tab["seeds_within_above_half"] = g["within_lo"].apply(lambda v: int((v > 0.5).sum()))
    print(f"=== systemic probes vs DR as a stand-in ({df['device'].iloc[0]} test split, {a.protocol}; "
          f"mean over seeds) ===")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nauroc_dr_score ~ auroc_probe and auroc_within_grade ~ 0.5: the probe is a DR proxy.\n"
          "seeds_within_above_half: runs whose within-grade CI lies above 0.5 (signal beyond the "
          "DR grade).")
    print(f"\nwrote {a.dir}/dr_proxy.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
