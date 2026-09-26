#!/usr/bin/env python3
"""
summarize_retinareach.py
========================
Turn per-seed RetinaReach runs (``run_retinareach.sh`` -> ``<dir>/seed<s>/``)
into the three deliverables of the plan (section 6):

1. **Threshold transfer with and without self-calibration** -- per source
   target and device: AUROC, sensitivity and flagged fraction at the shipped
   threshold under ``trained`` vs ``calibrated`` statistics, and the paired
   (same seed, same weights) calibrated-minus-trained delta with a t CI. This
   is immune to training nondeterminism, like every within-run contrast.
2. **The multi-seed capability table** -- per target x device x protocol, mean
   +/- SD over seeds of the image AUROC, the comparator AUROC and their paired
   delta; how many seeds each status; and the FINAL status: the most severe
   per-seed status (a safety gate takes the worst case), additionally demoted to
   NO_IMAGE_EVIDENCE when the mean delta does not clear
   ``max(margin, seed SD of the image AUROC)`` -- the measured noise floor that
   replaces the single-run default.
3. **How many captures a clinic needs** -- the calibration-size sweep pooled
   over seeds and repeats: AUROC and sensitivity at the shipped threshold vs N,
   with and without the source prior; plus the envelopes and the label-free
   device-recognition rate.

    python summarize_retinareach.py --dir exp_retinareach
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retinareach import STATUSES  # noqa: E402
from run_experiment import paired_stats  # noqa: E402

SEVERITY = {s: i for i, s in enumerate(STATUSES)}      # later = more severe


def md_table(df: pd.DataFrame) -> str:
    """GitHub markdown table without the optional ``tabulate`` dependency."""
    if df is None or len(df) == 0:
        return "(empty)"
    cell = lambda v: (f"{v:.3f}" if isinstance(v, (float, np.floating)) and v == v else
                      ("" if isinstance(v, float) else str(v)).replace("|", "/"))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load(d: str):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "seed*", "results.json"))):
        base = os.path.dirname(f)
        m = re.fullmatch(r"seed(\d+)", os.path.basename(base))
        if m is None:                         # e.g. seed0_old/: not a run of this sweep
            print(f"[skip] {base}: not a seed<N> run directory")
            continue
        s = int(m.group(1))
        cap = pd.read_csv(os.path.join(base, "capability.csv")).assign(seed=s)
        sweep_p = os.path.join(base, "calibration_sweep.csv")
        sweep = pd.read_csv(sweep_p).assign(seed=s) if os.path.exists(sweep_p) else pd.DataFrame()
        with open(f) as fh:
            runs[s] = {"results": json.load(fh), "cap": cap, "sweep": sweep}
    return runs


def transfer_table(cap: pd.DataFrame, source_targets: List[str]) -> List[dict]:
    rows = []
    for (t, dev), g in cap[cap["target"].isin(source_targets)].groupby(["target", "device"]):
        piv = {pr: g[g["protocol"] == pr].set_index("seed") for pr in ("trained", "calibrated")}
        row = {"target": t, "device": dev, "seeds": len(piv["calibrated"])}
        for m in ("auroc", "sensitivity", "flagged_fraction", "specificity"):
            for pr in ("trained", "calibrated"):
                v = piv[pr][m].dropna() if m in piv[pr] else pd.Series(dtype=float)
                row[f"{m}_{pr}"] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else \
                    (f"{v.mean():.3f}" if len(v) else "n/a")
            a = piv["calibrated"][m].dropna().to_dict() if m in piv["calibrated"] else {}
            b = piv["trained"][m].dropna().to_dict() if m in piv["trained"] else {}
            ps = paired_stats(a, b)
            if ps["n"]:
                row[f"{m}_delta"] = (f"{ps['mean_delta']:+.3f} [{ps['ci95'][0]:+.3f}, "
                                     f"{ps['ci95'][1]:+.3f}] {ps['n_positive']}/{ps['n']}"
                                     + (" *" if ps["significant"] else ""))
        rows.append(row)
    return rows


def capability_table(cap: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (t, dev, pr), g in cap.groupby(["target", "device", "protocol"], sort=False):
        counts = g["status"].value_counts().to_dict()
        worst = max(g["status"], key=lambda s: SEVERITY[s])
        auc, dl = g["auroc"].dropna(), g["delta"].dropna()
        margin = float(g["margin"].dropna().max()) if g["margin"].notna().any() else float("nan")
        floor = float(auc.std(ddof=1)) if len(auc) > 1 else float("nan")
        bar = np.nanmax([margin, floor]) if not (np.isnan(margin) and np.isnan(floor)) else float("nan")
        final, why = worst, "most severe per-seed status"
        if worst in ("SUPPORTED", "THRESHOLD_DRIFT") and len(dl) and dl.mean() < bar:
            final, why = "NO_IMAGE_EVIDENCE", (f"mean delta {dl.mean():+.3f} < noise floor "
                                               f"{bar:.3f} (seed SD {floor:.3f})")
        elif worst == "SUPPORTED":
            why = f"all seeds SUPPORTED; mean delta {dl.mean():+.3f} >= floor {bar:.3f}"
        out.append({"target": t, "device": dev, "protocol": pr, "final": final,
                    "seeds": len(g), "status_counts": ", ".join(f"{k}:{v}" for k, v in counts.items()),
                    "auroc": auc.mean() if len(auc) else np.nan,
                    "auroc_sd": floor, "meta_auroc": g["meta_auroc"].mean(),
                    "delta": dl.mean() if len(dl) else np.nan,
                    "delta_sd": dl.std(ddof=1) if len(dl) > 1 else np.nan,
                    "noise_floor": bar,
                    "sensitivity": g["sensitivity"].mean(),
                    "flagged_fraction": g["flagged_fraction"].mean(),
                    "pos_patients": g["pos_patients"].max(), "why": why})
    return pd.DataFrame(out)


def sweep_table(sweep: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    if sweep.empty:
        return sweep
    cols = [c for c in sweep.columns if c.split("_", 1)[0] in ("auroc", "sens", "flagged")
            and c.split("_", 1)[1] in targets]
    agg = sweep.groupby(["kind", "prior_strength", "n"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Multi-seed RetinaReach summary.")
    p.add_argument("--dir", default="exp_retinareach")
    a = p.parse_args(argv)
    runs = load(a.dir)
    if not runs:
        raise SystemExit(f"[fatal] no seed*/results.json under {a.dir}")
    seeds = sorted(runs)
    first = runs[seeds[0]]["results"]
    source_targets = first["args"]["source_targets"]
    cap = pd.concat([r["cap"] for r in runs.values()], ignore_index=True)
    sweep = pd.concat([r["sweep"] for r in runs.values()], ignore_index=True)

    tr = transfer_table(cap, source_targets)
    ct = capability_table(cap)
    sw = sweep_table(sweep, source_targets)
    env = {s: runs[s]["results"].get("envelope") for s in seeds}
    rec = {s: runs[s]["results"].get("device_recognition") for s in seeds}

    ff = lambda v: f"{v:.3f}"
    md = [f"# RetinaReach — {len(seeds)} seeds ({', '.join(map(str, seeds))})", "",
          "## 1. Threshold transfer: trained vs self-calibrated statistics", "",
          "Threshold fixed on the home validation split, never on the device it is applied to. "
          "Delta = calibrated − trained, paired by seed (t 95 % CI, seeds positive / n, * = CI "
          "excludes 0).", "",
          md_table(pd.DataFrame(tr)) if tr else "(no source targets)", "",
          "## 2. Capability table (final = most severe per-seed status, demoted when the mean "
          "image-minus-metadata delta does not clear the seed-SD noise floor)", "",
          md_table(ct), "",
          "## 3. Calibration size (handheld test; pooled over seeds and repeats)", "",
          md_table(sw) if len(sw) else "(no sweep)", "",
          "Envelope (95th pct same-camera distance) and label-free device recognition by seed:", "",
          "```", json.dumps({"envelope": env, "recognition": rec}, indent=1, default=float), "```"]
    os.makedirs(a.dir, exist_ok=True)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    ct.to_csv(os.path.join(a.dir, "capability_final.csv"), index=False)
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump({"seeds": seeds, "transfer": tr, "capability": ct.to_dict("records"),
                   "sweep": sw.to_dict("records") if len(sw) else []}, f, indent=2, default=float)
    print("\n".join(md[:4 + 1]))
    print(pd.DataFrame(tr).to_string(index=False) if tr else "")
    print("\n=== final capability ===")
    print(ct[["target", "device", "protocol", "final", "status_counts", "auroc", "meta_auroc",
              "delta", "noise_floor", "sensitivity"]].to_string(index=False, float_format=ff))
    print(f"\nwrote {a.dir}/summary.md, capability_final.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
