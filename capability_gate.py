#!/usr/bin/env python3
"""
capability_gate.py
==================
The RetinaReach **capability gate**: for every (target, device), does the image
carry evidence beyond what the intake form already knows, and does the shipped
threshold still hold? Statistics only -- no torch, no images -- so the metadata
half runs anywhere the label CSV is, and the image half is joined in from the
trainer's per-image predictions.

    # the bar each target must clear on a device (metadata only, runs today):
    python capability_gate.py --dataset mbrset --root <mBRSET> --out exp_retinareach/gate_mbrset
    # re-gate a finished run's predictions (e.g. with another margin):
    python capability_gate.py --dataset mbrset --root <mBRSET> --out <dir> \\
        --predictions <run>/predictions_handheld_calibrated.csv --protocol calibrated

The split
---------
One trunk sees every training image, so all heads must share ONE patient
partition per dataset: a patient in the test split of one target must not be a
training patient of another. :func:`multitask_split` is ``stratified_split``'s
procedure (patient-grouped StratifiedGroupKFold, val 0.10 / test 0.20) over ALL
rows -- a row missing one target's label stays in the split and is simply
masked for that target -- stratified on one target with "missing" as its own
stratum. ``--split-seed`` (42) is fixed and decoupled from the training seed
(plan section 5.1). The metadata comparator is refit on this same partition, so
image and metadata scores are paired on identical test rows.

The comparator
--------------
``metadata_model.py``'s logistic regression and shallow GBM, over the
``--gate-feature-sets`` (default ``age_sex clinical``: what a screening intake
form records). The stronger of those, chosen by repeated patient-grouped CV on
train+val only (the held-out rows never pick their own comparator), is fit on
train+val and scored on the test rows. The ``full`` chart (every other
comorbidity) is reported alongside as ``full_chart_auroc`` but does not gate:
at a screening site it would contain the diagnoses being screened for.

A label column whose decoded classes are not exactly {0, 1} (e.g. a 1/2-coded
release) is refused for that dataset by :func:`encoding_audit`, never trained.

The rule (per target, device, protocol)
---------------------------------------
    NOT_VALIDATED      no labels for the target in this device's test split
    UNDERPOWERED       fewer than --min-test-pos-patients positive patients
    NO_IMAGE_EVIDENCE  delta = AUROC(image) - AUROC(comparator) on the same rows
                       fails  delta >= margin  AND  lower 95% CI(delta) > 0,
                       margin = max(--gate-margin, comparator CV SD);
                       CI from a paired patient-cluster bootstrap
    THRESHOLD_DRIFT    sensitivity at the SHIPPED threshold (fixed on the home
                       validation split, never on this device) is below
                       --target-sens minus --sens-tolerance
    SUPPORTED          otherwise
The default --gate-margin 0.02 is the project's measured noise floor: fixed-seed
run-to-run SD of mBRSET AUROC ~0.02, seed-to-seed SD 0.012 (REPORT.md 5.3).
A multi-seed summary replaces it with the image model's own seed SD.
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
from dataset import LABEL_REGISTRY, _isnan_vector  # noqa: E402
from metadata_model import (FEATURE_SETS, _auc, _xy, build_model, cluster_bootstrap,  # noqa: E402
                            features_for, operating_point, repeated_cv, target_vector,
                            threshold_at_sensitivity)

GATE_FEATURE_SETS = ("age_sex", "clinical")
DEFAULT_MODELS = ("logreg", "gbm")


# --------------------------------------------------------------------------- #
# Split and labels
# --------------------------------------------------------------------------- #
def multitask_split(df: pd.DataFrame, strat_task: str, seed: int = 42, val_frac: float = 0.10,
                    test_frac: float = 0.20, group_col: str = "patient") -> Dict[str, pd.DataFrame]:
    """One patient-grouped train/val/test partition over every row of ``df``."""
    from sklearn.model_selection import StratifiedGroupKFold
    if group_col not in df.columns:
        raise ValueError(f"multitask_split: no {group_col!r} column; a patient-grouped split "
                         f"cannot be downgraded to a per-image one silently")
    df = df.reset_index(drop=True)
    y = target_vector(df, strat_task)
    strata = np.where(np.isnan(y), -1, y).astype(int)
    groups = df[group_col].to_numpy()
    k = max(2, int(round(1.0 / test_frac)))
    tv, te = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(df, strata, groups))
    k_v = max(2, int(round(1.0 / (val_frac / (1.0 - test_frac)))))
    tr, va = next(StratifiedGroupKFold(k_v, shuffle=True, random_state=seed)
                  .split(df.iloc[tv], strata[tv], groups[tv]))
    out = {"train": df.iloc[tv].iloc[tr], "val": df.iloc[tv].iloc[va], "test": df.iloc[te]}
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def label_matrix(df: pd.DataFrame, targets: Sequence[str]) -> np.ndarray:
    """[N, K] float labels, NaN where a target is missing or not in this CSV."""
    return np.stack([target_vector(df, t) for t in targets], axis=1) if len(targets) else \
        np.zeros((len(df), 0))


def encoding_audit(df: pd.DataFrame, target: str) -> Dict[str, object]:
    """Did the label function understand this CSV's encoding? A binary column
    coded 1/2 comes out as one class (1 -> positive, 2 -> NaN): count the raw
    non-null cells the label function turned into NaN, and refuse the target on
    this dataset if more than 1 % of them were lost."""
    spec = LABEL_REGISTRY[target]
    if any(c not in df.columns for c in spec.source_cols):
        return {"ok": False, "reason": f"column(s) {list(spec.source_cols)} absent"}
    raw_ok = df[list(spec.source_cols)].notna().all(axis=1).to_numpy()
    y = target_vector(df, target)
    lost = int((raw_ok & np.isnan(y)).sum())
    n = int(raw_ok.sum())
    classes = sorted(set(np.unique(y[~np.isnan(y)]).astype(int).tolist()))
    if n == 0:
        return {"ok": False, "reason": "no non-null cells", "n": 0}
    if lost > 0.01 * n:
        vals = pd.unique(df.loc[raw_ok & np.isnan(y), spec.source_cols[0]])[:5]
        return {"ok": False, "n": n, "lost": lost,
                "reason": f"{lost}/{n} non-null cells not understood (e.g. {list(vals)}): "
                          f"unaudited encoding"}
    if not set(classes) <= {0, 1}:
        # e.g. a 1/2-coded column through _yes_no, which passes numbers through
        return {"ok": False, "n": n, "reason": f"decoded classes {classes} are not 0/1: "
                                               f"unaudited encoding"}
    if len(classes) < 2:
        return {"ok": False, "n": n, "reason": f"single class {classes}"}
    return {"ok": True, "n": n, "lost": lost, "reason": None}


# --------------------------------------------------------------------------- #
# Metadata comparator
# --------------------------------------------------------------------------- #
def metadata_comparator(fit_df: pd.DataFrame, test_df: pd.DataFrame, task: str,
                        feature_sets: Sequence[str] = GATE_FEATURE_SETS,
                        models: Sequence[str] = DEFAULT_MODELS, seed: int = 0,
                        cv_repeats: int = 5) -> Dict[str, object]:
    """Strongest metadata-only model for ``task`` (chosen by grouped CV on
    ``fit_df``), fit on ``fit_df``, scored on ``test_df``.

    ``scores`` is aligned to ``test_df``'s rows (NaN where the label is
    missing), so it pairs row-for-row with the image model's scores.
    """
    out: Dict[str, object] = {"task": task, "scores": np.full(len(test_df), np.nan),
                              "feature_set": None, "model": None, "cv_mean": float("nan"),
                              "cv_sd": float("nan"), "features": [], "reason": None}
    y_fit = target_vector(fit_df, task)
    if np.all(np.isnan(y_fit)) or len(np.unique(y_fit[~np.isnan(y_fit)])) < 2:
        out["reason"] = "target missing or single-class in train+val"
        return out
    best = None
    for fs in feature_sets:
        feats = features_for(task, fs, fit_df.columns)
        if not feats:
            continue
        X, y, g, kinds = _xy(fit_df, task, feats)
        for mk in models:
            cv = repeated_cv(mk, X, y, g, kinds, cv_repeats, seed)
            if cv["auroc_mean"] == cv["auroc_mean"] and (best is None or cv["auroc_mean"] > best[0]):
                best = (cv["auroc_mean"], cv["auroc_sd"], fs, mk, feats, kinds, X, y)
    if best is None:
        out["reason"] = "no metadata features of the gate sets in this CSV"
        return out
    cv_mean, cv_sd, fs, mk, feats, kinds, X, y = best
    model = build_model(mk, kinds, seed).fit(X, y)
    y_te = target_vector(test_df, task)
    keep = ~np.isnan(y_te)
    if keep.any():
        X_te, _, _, _ = _xy(test_df, task, feats, kinds)
        out["scores"][keep] = model.predict_proba(X_te)[:, 1]
    out.update(feature_set=fs, model=mk, cv_mean=float(cv_mean), cv_sd=float(cv_sd),
               features=list(feats))
    return out


def full_chart_auroc(fit_df: pd.DataFrame, test_df: pd.DataFrame, task: str,
                     seed: int = 0) -> float:
    """Reference only: the whole-chart logistic regression on the same rows."""
    feats = features_for(task, "full", fit_df.columns)
    y_fit = target_vector(fit_df, task)
    y_te = target_vector(test_df, task)
    if not feats or len(np.unique(y_fit[~np.isnan(y_fit)])) < 2 or \
            len(np.unique(y_te[~np.isnan(y_te)])) < 2:
        return float("nan")
    X, y, _, kinds = _xy(fit_df, task, feats)
    X_te, yt, _, _ = _xy(test_df, task, feats, kinds)
    return _auc(yt, build_model("logreg", kinds, seed).fit(X, y).predict_proba(X_te)[:, 1])


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_delta_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray,
                           n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """AUROC(a) - AUROC(b) on the same rows, with a 95 % CI from resampling
    PATIENTS (both eyes and repeat images move together, both models see the
    same resample)."""
    d = _auc(y, a) - _auc(y, b)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(_auc(y[idx], a[idx]) - _auc(y[idx], b[idx]))
    lo, hi = (np.percentile(ds, [2.5, 97.5]) if ds else (float("nan"), float("nan")))
    return {"delta": float(d), "delta_lo": float(lo), "delta_hi": float(hi), "n_boot": len(ds)}


def patient_auroc(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """AUROC after averaging each patient's image scores (patient-level label =
    max over their images; systemic labels are constant within a patient)."""
    d = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").agg(y=("y", "max"), p=("p", "mean"))
    return _auc(d["y"].to_numpy().astype(int), d["p"].to_numpy())


def gate_status(has_labels: bool, pos_patients: int, min_pos_patients: int,
                delta: float, delta_lo: float, margin: float,
                sens: float, target_sens: float, sens_tolerance: float,
                label_reason: Optional[str] = None) -> Tuple[str, str]:
    """The decision table in the module docstring; returns (status, reason)."""
    if not has_labels:
        return "NOT_VALIDATED", label_reason or "no labels on this device"
    if pos_patients < min_pos_patients:
        return "UNDERPOWERED", f"{pos_patients} positive patients < {min_pos_patients}"
    if not (delta == delta and delta_lo == delta_lo):
        return "NO_IMAGE_EVIDENCE", "image-minus-metadata delta could not be computed"
    if delta < margin or delta_lo <= 0:
        return "NO_IMAGE_EVIDENCE", (f"image-minus-metadata {delta:+.3f} "
                                     f"[{delta_lo:+.3f}, ...] vs margin {margin:.3f}")
    if not (sens == sens):
        return "THRESHOLD_DRIFT", ("no shipped threshold (the home validation split had "
                                   "no positives to fix one on)")
    if sens < target_sens - sens_tolerance:
        return "THRESHOLD_DRIFT", (f"sensitivity {sens:.2f} at the shipped threshold "
                                   f"< {target_sens - sens_tolerance:.2f}")
    return "SUPPORTED", (f"image-minus-metadata {delta:+.3f} [{delta_lo:+.3f}, ...], "
                         f"sensitivity {sens:.2f} at the shipped threshold")


def gate_cell(target: str, device: str, protocol: str, test_df: pd.DataFrame,
              scores: np.ndarray, threshold: float, comparator: Dict[str, object], *,
              target_sens: float = 0.85, sens_tolerance: float = 0.10, gate_margin: float = 0.02,
              min_pos_patients: int = 20, n_boot: int = 1000, seed: int = 0,
              audit: Optional[Dict[str, object]] = None,
              full_chart: float = float("nan")) -> Dict[str, object]:
    """One row of the capability table: discrimination, operating point at the
    shipped threshold, comparator, paired delta and status."""
    row: Dict[str, object] = {"target": target, "device": device, "protocol": protocol,
                              "threshold": float(threshold)}
    y_all = target_vector(test_df, target)
    ok = ~np.isnan(y_all) & ~np.isnan(scores)
    audit_ok = audit is None or audit.get("ok", False)
    has = bool(audit_ok and ok.sum() > 0 and len(np.unique(y_all[ok])) == 2)
    y, p = y_all[ok].astype(int), scores[ok]
    g = test_df["patient"].to_numpy()[ok]
    pos_pat = int(len(np.unique(g[y == 1]))) if has else 0
    row.update(n=int(ok.sum()), pos=int(y.sum()) if has else 0, pos_patients=pos_pat,
               prevalence=float(y.mean()) if has else float("nan"))
    if has:
        ci = cluster_bootstrap(y, p, g, n_boot, seed)
        op = operating_point(y, p, threshold)
        if not (threshold == threshold):        # no threshold was fixed: nothing to report at it
            op = {k: float("nan") for k in op}
        thr_oracle = threshold_at_sensitivity(y, p, target_sens)
        row.update(auroc=_auc(y, p), auroc_lo=ci["auroc_ci"][0], auroc_hi=ci["auroc_ci"][1],
                   patient_auroc=patient_auroc(y, p, g),
                   sensitivity=op["sensitivity"], specificity=op["specificity"], ppv=op["ppv"],
                   flagged_fraction=op["flagged_fraction"],
                   oracle_threshold=thr_oracle)
        m = np.asarray(comparator.get("scores", np.full(len(test_df), np.nan)))[ok]
        if comparator.get("reason") is None and not np.isnan(m).any():
            meta_auc = _auc(y, m)
            pd_ = paired_delta_bootstrap(y, p, m, g, n_boot, seed)
            margin = max(gate_margin, comparator.get("cv_sd", 0.0) or 0.0)
            row.update(meta_auroc=meta_auc, meta_feature_set=comparator["feature_set"],
                       meta_model=comparator["model"], meta_cv_sd=comparator["cv_sd"], **pd_)
        else:
            # No comparator available (no metadata features on this device):
            # the image must then beat chance by the same margin.
            chance = np.full(len(y), 0.5)
            pd_ = paired_delta_bootstrap(y, p, chance, g, n_boot, seed)
            margin = gate_margin
            row.update(meta_auroc=0.5, meta_feature_set="none (chance)", meta_model=None,
                       meta_cv_sd=float("nan"), **pd_)
        row["margin"] = float(margin)
        row["full_chart_auroc"] = full_chart
        status, reason = gate_status(True, pos_pat, min_pos_patients, pd_["delta"],
                                     pd_["delta_lo"], margin, op["sensitivity"],
                                     target_sens, sens_tolerance)
    else:
        why = (audit or {}).get("reason") if not audit_ok else None
        status, reason = gate_status(False, 0, min_pos_patients, float("nan"), float("nan"),
                                     gate_margin, float("nan"), target_sens, sens_tolerance,
                                     label_reason=why or "no labels on this device")
    row.update(status=status, reason=reason, shown=status == "SUPPORTED")
    return row


TABLE_COLS = ["target", "device", "protocol", "status", "n", "pos_patients", "prevalence",
              "auroc", "auroc_lo", "auroc_hi", "meta_auroc", "meta_feature_set", "delta",
              "delta_lo", "delta_hi", "margin", "full_chart_auroc", "threshold", "sensitivity",
              "specificity", "ppv", "flagged_fraction", "reason"]


def format_table(rows: List[Dict[str, object]], cols: Sequence[str] = TABLE_COLS) -> str:
    t = pd.DataFrame(rows)
    return t[[c for c in cols if c in t.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    from metadata_model import load_table
    from retinareach import PROBE_TARGETS, SOURCE_TARGETS
    p = argparse.ArgumentParser(description="RetinaReach capability gate (metadata side, or re-gate "
                                            "a run's predictions).")
    p.add_argument("--dataset", choices=["mbrset", "brset"], required=True)
    p.add_argument("--root", required=True, help="dataset root (directory holding the label CSV)")
    p.add_argument("--out", required=True)
    p.add_argument("--targets", nargs="+", default=list(SOURCE_TARGETS) + list(PROBE_TARGETS))
    p.add_argument("--strat-target", default="dr_referable",
                   help="target the patient split is stratified on (must match the trainer)")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-feature-sets", nargs="+", default=list(GATE_FEATURE_SETS),
                   choices=list(FEATURE_SETS))
    p.add_argument("--cv-repeats", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--min-test-pos-patients", type=int, default=20)
    p.add_argument("--predictions", default=None,
                   help="per-image predictions CSV written by train_retinareach.py "
                        "(columns file, p_<target>, threshold_<target>)")
    p.add_argument("--device-name", default=None, help="label for the device column")
    p.add_argument("--protocol", default="calibrated")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--sens-tolerance", type=float, default=0.10)
    p.add_argument("--gate-margin", type=float, default=0.02)
    a = p.parse_args(argv)

    df, csv = load_table(a.root, a.dataset)
    split = multitask_split(df, a.strat_target, seed=a.split_seed)
    fit = pd.concat([split["train"], split["val"]], ignore_index=True)
    test = split["test"]
    device = a.device_name or a.dataset
    print(f"{a.dataset}: {csv}  rows={len(df)} patients={df['patient'].nunique()}  "
          f"split(seed {a.split_seed}): train={len(split['train'])} val={len(split['val'])} "
          f"test={len(test)} rows")
    preds = None
    if a.predictions:
        preds = pd.read_csv(a.predictions)
        n_test = len(test)
        test = test.merge(preds, on="file", how="inner", suffixes=("", "_pred"))
        print(f"joined {len(test)} test rows with predictions from {a.predictions}")
        if len(test) != len(preds):
            raise SystemExit(f"[fatal] {len(preds) - len(test)} predicted rows are not in this "
                             f"split's test set ({n_test} rows): --split-seed / --strat-target "
                             f"must match the run that wrote {a.predictions}")

    rows, bars = [], []
    for t in a.targets:
        audit = encoding_audit(df, t)
        if not audit["ok"]:
            bars.append({"target": t, "status": "NOT_VALIDATED", "reason": audit["reason"]})
            if preds is not None:
                rows.append(gate_cell(t, device, a.protocol, test, np.full(len(test), np.nan),
                                      float("nan"), {"reason": "n/a"}, audit=audit))
            continue
        comp = metadata_comparator(fit, test, t, a.gate_feature_sets, seed=a.seed,
                                   cv_repeats=a.cv_repeats)
        y = target_vector(test, t)
        keep = ~np.isnan(y)
        g = test["patient"].to_numpy()
        pos_pat = int(len(np.unique(g[keep][y[keep] == 1])))
        bar = {"target": t, "prevalence": float(np.nanmean(target_vector(df, t))),
               "test_rows": int(keep.sum()), "test_pos_patients": pos_pat,
               "underpowered": pos_pat < a.min_test_pos_patients,
               "comparator": f"{comp['feature_set']}/{comp['model']}",
               "cv_auroc": comp["cv_mean"], "cv_sd": comp["cv_sd"],
               "full_chart_auroc": full_chart_auroc(fit, test, t, a.seed), "reason": comp["reason"]}
        if comp["reason"] is None and keep.any() and len(np.unique(y[keep])) == 2:
            s = np.asarray(comp["scores"])[keep]
            bar["test_auroc"] = _auc(y[keep].astype(int), s)
            ci = cluster_bootstrap(y[keep].astype(int), s, g[keep], a.n_boot, a.seed)
            bar["test_auroc_lo"], bar["test_auroc_hi"] = ci["auroc_ci"]
            bar["image_must_reach"] = bar["test_auroc"] + max(a.gate_margin, comp["cv_sd"])
        bars.append(bar)
        print(f"  {t:<22} comparator {bar['comparator']:<16} test AUROC "
              f"{bar.get('test_auroc', float('nan')):.3f}  -> image must reach "
              f"{bar.get('image_must_reach', float('nan')):.3f}  "
              f"(pos patients {pos_pat}{', UNDERPOWERED' if bar['underpowered'] else ''})", flush=True)
        if preds is not None and f"p_{t}" in test.columns:
            thr = float(test[f"threshold_{t}"].iloc[0]) if f"threshold_{t}" in test.columns \
                else float("nan")
            sc = test[f"p_{t}"].to_numpy(float)
            if np.isnan(sc).all():
                audit = {"ok": False, "reason": f"no predictions for {t} under {a.protocol}"}
            rows.append(gate_cell(t, device, a.protocol, test, sc,
                                  thr, comp, target_sens=a.target_sens,
                                  sens_tolerance=a.sens_tolerance, gate_margin=a.gate_margin,
                                  min_pos_patients=a.min_test_pos_patients, n_boot=a.n_boot,
                                  seed=a.seed, audit=audit, full_chart=bar["full_chart_auroc"]))

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(bars).to_csv(os.path.join(a.out, "metadata_bar.csv"), index=False)
    with open(os.path.join(a.out, "metadata_bar.json"), "w") as f:
        json.dump({"dataset": a.dataset, "csv": csv, "args": vars(a), "bars": bars}, f,
                  indent=2, default=float)
    print(f"\n=== METADATA BAR ({a.dataset}, split seed {a.split_seed}; gate sets "
          f"{'+'.join(a.gate_feature_sets)}) ===")
    print(pd.DataFrame(bars)[["target", "prevalence", "test_pos_patients", "underpowered",
                              "comparator", "cv_auroc", "cv_sd", "test_auroc",
                              "image_must_reach", "full_chart_auroc", "reason"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "capability.csv"), index=False)
        print(f"\n=== CAPABILITY ({device}, {a.protocol}) ===")
        print(format_table(rows))
    print(f"\nwrote {a.out}/metadata_bar.{{csv,json}}" + (" and capability.csv" if rows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
