#!/usr/bin/env python3
"""
inspect_mbrset.py
=================
Pre-flight for the systemic (oculomics) targets: read the mBRSET label CSV and
say, per task, whether it is trainable and what an age+sex baseline already
scores -- BEFORE a GPU sweep is spent on it.

    python inspect_mbrset.py --csv <mBRSET>/labels_mbrset.csv
    python inspect_mbrset.py --csv ... --tasks hypertension nephropathy --strict

For every requested task it prints the raw distinct values of the source column
(so a 1/2 or "Sim"/"Nao" release is caught here, not as an empty dataset deep in
the trainer), the derived 0/1 counts after ``dataset._binary_flag``, the
prevalence, the number of patients, and the AUROC of a logistic regression on
age+sex fit on a seed-42 patient-grouped split. ``--strict`` exits non-zero if
any requested task has fewer than two classes -- run_systemic.sh uses it as a gate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LABEL_REGISTRY, SYSTEMIC_TASKS, _isnan_vector, stratified_split  # noqa: E402
from covariate_baseline import covariate_baseline, DEFAULT_FEATURES                   # noqa: E402


def inspect(df: pd.DataFrame, tasks, features=DEFAULT_FEATURES, seed: int = 42):
    """Per-task report as a list of dicts (also what the tests check)."""
    rows = []
    for task in tasks:
        spec = LABEL_REGISTRY[task]
        col = spec.source_cols[0]
        row = {"task": task, "column": col, "present": col in df.columns,
               "raw_values": None, "n": 0, "pos": 0, "prevalence": None,
               "patients": None, "covariate_auroc": float("nan"), "ok": False}
        if not row["present"]:
            rows.append(row)
            continue
        raw = pd.unique(df[col].dropna())
        row["raw_values"] = sorted(map(str, raw))[:12]
        y = np.array([spec.fn({col: v}) for v in df[col].to_numpy()], dtype=np.float64)
        keep = ~_isnan_vector(y)
        row["n"] = int(keep.sum())
        row["pos"] = int((y[keep] == 1).sum())
        row["prevalence"] = (row["pos"] / row["n"]) if row["n"] else None
        if "patient" in df.columns:
            row["patients"] = int(df.loc[keep, "patient"].nunique())
        row["ok"] = row["n"] > 0 and 0 < row["pos"] < row["n"]
        if row["ok"] and "patient" in df.columns:
            try:
                sp = stratified_split(df, task=task, val_frac=0.10, test_frac=0.20,
                                      group_col="patient", seed=seed)
                cb = covariate_baseline(sp["train"], sp["test"], task, features=features)
                row["covariate_auroc"] = cb["auroc"]
                row["covariate_reason"] = cb.get("reason")
            except ValueError as e:              # a class too rare to stratify
                row["covariate_reason"] = str(e)
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-flight the systemic targets on an mBRSET CSV.")
    p.add_argument("--csv", required=True, help="labels_mbrset.csv")
    p.add_argument("--tasks", nargs="+", default=list(SYSTEMIC_TASKS),
                   help="Tasks to check (default: every systemic task).")
    p.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES),
                   help="Covariates for the logistic baseline (default: age sex).")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any requested task is missing or single-class.")
    args = p.parse_args()

    bad = [t for t in args.tasks if t not in LABEL_REGISTRY]
    if bad:
        print(f"[fatal] unknown task(s) {bad}; known: {sorted(LABEL_REGISTRY)}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.csv)
    print(f"\n{'='*78}\nmBRSET CSV: {args.csv}\n  {len(df)} rows, {len(df.columns)} columns"
          + (f", {df['patient'].nunique()} patients" if "patient" in df.columns else "")
          + f"\n{'='*78}")
    missing_cov = [f for f in args.features if f not in df.columns]
    if missing_cov:
        print(f"[warn] covariate column(s) {missing_cov} absent; no baseline AUROC will be printed")

    rows = inspect(df, args.tasks, features=tuple(args.features))
    print(f"\n{'task':<22}{'column':<28}{'n':>6}{'pos':>6}{'prev':>8}{'pts':>6}"
          f"{'age+sex AUROC':>15}  status")
    failed = []
    for r in rows:
        if not r["present"]:
            print(f"{r['task']:<22}{r['column']:<28}{'':>6}{'':>6}{'':>8}{'':>6}{'':>15}  MISSING column")
            failed.append(r["task"]); continue
        prev = f"{r['prevalence']:.1%}" if r["prevalence"] is not None else "-"
        cb = f"{r['covariate_auroc']:.3f}" if r["covariate_auroc"] == r["covariate_auroc"] else "-"
        status = "ok" if r["ok"] else "SINGLE-CLASS / EMPTY after _binary_flag"
        print(f"{r['task']:<22}{r['column']:<28}{r['n']:>6}{r['pos']:>6}{prev:>8}"
              f"{(r['patients'] or 0):>6}{cb:>15}  {status}")
        print(f"{'':<22}raw values: {r['raw_values']}")
        if not r["ok"]:
            failed.append(r["task"])

    print("\nRead the age+sex AUROC as the floor: an image model that does not beat it")
    print("has learned the patient's age from the retina, not the disease. The trainer")
    print("recomputes this baseline on each run's own split (--covariate-baseline).")
    if failed:
        print(f"\n[{'FAIL' if args.strict else 'warn'}] not trainable as encoded: {failed}")
        print("  If the raw values above are 1/2 or Portuguese tokens, extend _POS_TOKENS /")
        print("  _NEG_TOKENS in dataset.py -- do not edit the CSV.")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
