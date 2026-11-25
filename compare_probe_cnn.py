"""
compare_probe_cnn.py
====================
The Recommendation-1 decision, on the same patients: does the frozen-RETFound
probe pass where the trained CNN does not?

Reads the pooled out-of-fold test rows of two conditions (``<dir>/<task>/<cond>_fold<k>_seed<s>_test.csv``
from probe_systemic.py and train_mbrset.py --fold), joins them on the image file,
and per arm reports the two pre-specified readings:

  * incremental delta: AUROC(covariates + photo) - AUROC(covariates), with a
    patient-cluster bootstrap CI (``covariate_baseline.incremental_block``);
  * matched-pair concordance: positive vs negative patients with the same
    covariates except age, matched within ``--caliper`` years.

An arm PASSES when the delta's CI lower bound is above 0 AND the concordance's CI
lower bound is above 0.5. Nothing is judged unless both arms are complete (every
fold's JSON and rows), share n_folds, fold seed, covariates and camera, put every
image in the same fold, and cover the same rows. Then the paired difference, probe minus CNN, of the photo
score alone and of the combined model, on the identical rows (patients resampled
together). If the probe passes and the CNN does not, distil fold K's probe into
fold K's student:

    train_mbrset.py ... --fold K --n-folds 5 --fold-seed 42 \\
        --teacher exp_probe/<set>/<task>/probe_foldK_seed0_probe.npz

    python compare_probe_cnn.py --probe-dir exp_probe/mbrset --cnn-dir exp_systemic_folds \\
        --tasks hypertension nephropathy myocardial_infarction neuropathy
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RUN = re.compile(r"^(?P<cond>.+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json$")


def pooled_rows(task_dir: str, cond: str, seed: int):
    """``(rows, info)`` for one condition and seed, or ``(None, reason)``. Complete
    means a fold JSON (written last, the completion marker) and its test rows for
    every fold 0..n_folds-1, with n_folds and the fold seed read from the JSONs,
    and no patient tested twice."""
    import json
    recs = {}
    for p in glob.glob(os.path.join(task_dir, f"{cond}_fold*_seed{seed}.json")):
        m = _RUN.match(os.path.basename(p))
        if m and m["cond"] == cond and int(m["seed"]) == seed:
            with open(p) as f:
                rec = json.load(f)
            rec["_csv"] = os.path.splitext(p)[0] + "_test.csv"
            recs[int(m["fold"])] = rec
    if not recs:
        return None, "no finished fold"
    n = {r.get("n_folds") for r in recs.values()}
    fs = {r.get("fold_seed") for r in recs.values()}
    if len(n) != 1 or len(fs) != 1 or None in n:
        return None, f"folds disagree on n_folds {n} / fold seed {fs}"
    n = n.pop()
    if sorted(recs) != list(range(n)) or not all(os.path.isfile(r["_csv"]) for r in recs.values()):
        return None, f"incomplete: folds {sorted(recs)} of {n} finished"
    rows = pd.concat([pd.read_csv(recs[k]["_csv"]) for k in range(n)], ignore_index=True)
    if rows.groupby(rows["patient"].astype(str))["fold"].nunique().max() > 1:
        raise SystemExit(f"[fatal] {task_dir} {cond}: a patient is tested in two folds")
    r0 = recs[0]
    inc = r0.get("incremental") or r0.get("incremental_regression") or {}
    return rows, {"n_folds": n, "fold_seed": fs.pop(), "features": inc.get("features"),
                  "camera": r0.get("camera") or (r0.get("args") or {}).get("camera")}


def arm_reading(rows: pd.DataFrame, feats: List[str], caliper: float, n_boot: int, seed: int) -> dict:
    from covariate_baseline import incremental_block, matched_pair_concordance
    ok = rows[["label", "p_image", "p_covariates", "p_combined"]].notna().all(1).to_numpy()
    y, g = rows["label"].to_numpy(), rows["patient"].astype(str).to_numpy()
    inc = incremental_block(np.nan_to_num(y), g, rows["p_covariates"].to_numpy(),
                            rows["p_combined"].to_numpy(), rows["p_image"].to_numpy(), ok,
                            n_boot=n_boot, seed=seed)
    exact = [c for c in feats if c != "age" and c in rows.columns]
    mp = matched_pair_concordance(y[ok], rows["p_image"].to_numpy()[ok], g[ok], rows["age"].to_numpy()[ok],
                                  exact={k: rows[k].to_numpy()[ok] for k in exact}, caliper=caliper,
                                  n_boot=n_boot, seed=seed) if "age" in rows.columns else {}
    passes = bool(inc.get("delta_ci", [-1])[0] > 0 and mp.get("ci", [0])[0] > 0.5)
    return {"incremental": inc, "matched_pairs": mp, "passes": passes}


def compare(probe_dir: str, cnn_dir: str, tasks: List[str], probe_cond: str = "probe",
            cnn_cond: str = "ctrl", seed: int = 0, feats: Optional[List[str]] = None,
            caliper: float = 2.0, n_boot: int = 2000) -> Dict[str, dict]:
    from capability_gate import paired_delta_bootstrap
    out, L = {}, [f"PROBE ({probe_dir}, {probe_cond}) vs CNN ({cnn_dir}, {cnn_cond}), seed {seed}"]
    for t in tasks:
        pr, pinfo = pooled_rows(os.path.join(probe_dir, t), probe_cond, seed)
        cn, cinfo = pooled_rows(os.path.join(cnn_dir, t), cnn_cond, seed)
        if pr is None or cn is None:
            L.append(f"\n[{t}] not compared: probe {'ok' if pr is not None else pinfo}; "
                     f"CNN {'ok' if cn is not None else cinfo}")
            continue
        diff = {k: (pinfo[k], cinfo[k]) for k in ("n_folds", "fold_seed", "features", "camera")
                if pinfo[k] != cinfo[k]}
        if diff:
            L.append(f"\n[{t}] not compared: the arms differ in {diff} (probe vs CNN); rerun one of them "
                     f"with the other's FOLDS / FOLD_SEED / COVARIATES / CAMERA")
            continue
        need = {"p_covariates", "p_combined"}
        if not (need <= set(pr.columns) and need <= set(cn.columns)):
            L.append(f"\n[{t}] not compared: an arm has no covariate scores in its fold rows "
                     f"(its incremental model was not fit; check the covariate columns)")
            continue
        f = feats or pinfo["features"] or [c for c in ("age", "sex", "diabetes")
                                           if c in pr.columns and c in cn.columns]
        j = pr.merge(cn[["file", "fold", "p_image", "p_combined"]], on="file", suffixes=("_probe", "_cnn"))
        if not (j["fold_probe"] == j["fold_cnn"]).all():
            L.append(f"\n[{t}] not compared: {int((j['fold_probe'] != j['fold_cnn']).sum())} images sit in "
                     f"different folds in the two arms (not one partition)")
            continue
        j = j.dropna(subset=["label", "p_image_probe", "p_image_cnn", "p_combined_probe", "p_combined_cnn"])
        same = len(j) / max(len(pr), len(cn))
        if same < 0.99:
            L.append(f"\n[{t}] not compared: only {same:.0%} of rows match on file, so the arms do not cover "
                     f"the same patients (check --camera, roots, missing images)")
            continue
        res = {"probe": arm_reading(pr, f, caliper, n_boot, seed),
               "cnn": arm_reading(cn, f, caliper, n_boot, seed)}
        y, g = j["label"].to_numpy().astype(int), j["patient"].astype(str).to_numpy()
        res["paired"] = {
            "n_rows": int(len(j)), "row_overlap": float(same),
            "image": paired_delta_bootstrap(y, j["p_image_probe"].to_numpy(), j["p_image_cnn"].to_numpy(),
                                            g, n_boot, seed),
            "combined": paired_delta_bootstrap(y, j["p_combined_probe"].to_numpy(),
                                               j["p_combined_cnn"].to_numpy(), g, n_boot, seed)}
        out[t] = res
        L.append(f"\n--- {t} --- ({len(j)} rows, {len(np.unique(g))} patients on both arms; "
                 f"covariates {'+'.join(f)})")
        for arm in ("probe", "cnn"):
            inc, mp = res[arm]["incremental"], res[arm]["matched_pairs"]
            lo, hi = inc.get("delta_ci", [float("nan")] * 2)
            ml, mh = mp.get("ci", [float("nan")] * 2)
            L.append(f"  {arm:<5}  delta {inc.get('delta', float('nan')):+.4f} [{lo:+.4f}, {hi:+.4f}]   "
                     f"matched pairs {mp.get('concordance', float('nan')):.1%} [{ml:.1%}, {mh:.1%}] "
                     f"of {mp.get('pairs', 0):,}   -> {'PASSES' if res[arm]['passes'] else 'does not pass'}")
        for k in ("image", "combined"):
            d = res["paired"][k]
            L.append(f"  probe minus CNN, {k:<8} AUROC {d['delta']:+.4f} [{d['delta_lo']:+.4f}, "
                     f"{d['delta_hi']:+.4f}]")
        if res["probe"]["passes"] and not res["cnn"]["passes"]:
            L.append(f"  => the probe passes and the CNN does not: distil it (train_mbrset.py --fold K "
                     f"--teacher {probe_dir}/{t}/{probe_cond}_foldK_seed{seed}_probe.npz)")
    print("\n".join(L))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--cnn-dir", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--probe-cond", default="probe")
    p.add_argument("--cnn-cond", default="ctrl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--covariates", nargs="+", default=None,
                   help="exact-match columns of the pairs besides age (default: age, sex, diabetes if present)")
    p.add_argument("--caliper", type=float, default=2.0)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args(argv)
    compare(a.probe_dir, a.cnn_dir, a.tasks, a.probe_cond, a.cnn_cond, a.seed, a.covariates,
            a.caliper, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
