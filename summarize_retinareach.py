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
4. **Controls and mechanism** -- permuted-label probes and the quality-only
   AUROC per target, the ``shuffle_seed*`` negative-control runs (every
   source-target AUROC must sit at chance), camera decodability as-trained vs
   calibrated (paired by seed), AUROC on gradable vs ungradable images, and the
   vessel ablation (vessels removed vs the same area elsewhere).
5. **Unfamiliar cameras** -- the gate on each ``--unfamiliar`` camera from its
   own labels, and how often the phone's label-free lookup showed a target those
   labels do not support (unsafe) or hid one they do (missed), by N.

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


def _csv(base: str, name: str, seed: int) -> pd.DataFrame:
    path = os.path.join(base, name)
    return pd.read_csv(path).assign(seed=seed) if os.path.exists(path) else pd.DataFrame()


def load(d: str, prefix: str = "seed"):
    """Runs under ``<d>/<prefix><N>/``; ``prefix="shuffle_seed"`` loads the
    negative-control runs, which never enter the main tables."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, f"{prefix}*", "results.json"))):
        base = os.path.dirname(f)
        m = re.fullmatch(prefix + r"(\d+)", os.path.basename(base))
        if m is None:                         # e.g. seed0_old/: not a run of this sweep
            print(f"[skip] {base}: not a {prefix}<N> run directory")
            continue
        s = int(m.group(1))
        with open(f) as fh:
            res = json.load(fh)
        if res.get("shuffle_source_labels") and prefix == "seed":
            print(f"[skip] {base}: a shuffled-label control run; move it to shuffle_seed{s}/")
            continue
        runs[s] = {"results": res, "cap": _csv(base, "capability.csv", s),
                   "sweep": _csv(base, "calibration_sweep.csv", s),
                   "strata": _csv(base, "quality_strata.csv", s),
                   "unf": _csv(base, "capability_unfamiliar.csv", s),
                   "lookup": _csv(base, "unfamiliar_lookup.csv", s),
                   "ablation": _csv(base, "anatomy_ablation.csv", s),
                   "pertarget": _pertarget(base, s)}
    return runs


def _pertarget(base: str, seed: int) -> pd.DataFrame:
    """train_mbrset.py per-target results written next to a seed's run."""
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "pertarget_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        rows.append({"seed": seed, "target": r["task"], "finetuned_auroc": (r.get("test") or {}).get("auroc"),
                     "covariate_auroc": (r.get("covariate_baseline") or {}).get("auroc"),
                     "n_test": (r.get("test") or {}).get("n")})
    return pd.DataFrame(rows)


def shared_vs_pertarget(cap: pd.DataFrame, pt: pd.DataFrame, device: str) -> List[dict]:
    """Paired by seed on the same handheld test rows: fine-tuned per-target model
    minus the shared-trunk calibrated probe."""
    out = []
    for t, g in pt.groupby("target"):
        probe = cap[(cap["target"] == t) & (cap["device"] == device)
                    & (cap["protocol"] == "calibrated")].set_index("seed")["auroc"].dropna().to_dict()
        fine = g.set_index("seed")["finetuned_auroc"].dropna().to_dict()
        ps = paired_stats(fine, probe)
        out.append({"target": t, "seeds": ps["n"],
                    "probe_auroc": float(np.mean(list(probe.values()))) if probe else np.nan,
                    "finetuned_auroc": float(np.mean(list(fine.values()))) if fine else np.nan,
                    "finetuned_minus_probe": (f"{ps['mean_delta']:+.3f} [{ps['ci95'][0]:+.3f}, "
                                              f"{ps['ci95'][1]:+.3f}] {ps['n_positive']}/{ps['n']}"
                                              + (" *" if ps["significant"] else "")) if ps["n"] else "n/a",
                    "covariate_auroc": g["covariate_auroc"].mean()})
    return out


def _cat(runs, key) -> pd.DataFrame:
    frames = [r[key] for r in runs.values() if len(r[key])]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def controls_table(cap: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("shuffle_auroc", "quality_only_auroc", "meta_auroc", "auroc", "ece",
                        "mean_predicted", "prevalence") if c in cap.columns]
    c = cap[cap["protocol"] == "calibrated"].dropna(subset=["auroc"])
    return c.groupby(["target", "device"])[cols].mean().reset_index() if len(c) else c


def decode_table(runs) -> Dict[str, object]:
    by = {pr: {s: r["results"]["camera_decodability"][pr]["auroc"] for s, r in runs.items()
               if r["results"].get("camera_decodability")} for pr in ("trained", "calibrated")}
    ps = paired_stats(by["calibrated"], by["trained"])
    return {"trained": by["trained"], "calibrated": by["calibrated"], "paired": ps}


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
    fa = {s: runs[s]["results"].get("device_false_accept") for s in seeds}
    ctl = controls_table(cap)
    dec = decode_table(runs)
    strata = _cat(runs, "strata")
    st = (strata.groupby(["target", "device", "protocol", "stratum"])[["n", "auroc", "sensitivity"]]
          .mean().reset_index() if len(strata) else strata)
    shuf = load(a.dir, prefix="shuffle_seed")
    shuf_cap = _cat(shuf, "cap")
    shuf_t = (shuf_cap[shuf_cap["target"].isin(source_targets)]
              .groupby(["target", "device", "protocol"])["auroc"].mean().reset_index()
              if len(shuf_cap) else shuf_cap)
    unf, look = _cat(runs, "unf"), _cat(runs, "lookup")
    unf_ct = capability_table(unf) if len(unf) else unf
    unf_tr = transfer_table(unf, source_targets) if len(unf) else []
    lk = (look.groupby(["device", "target", "n"])[["shown_rate", "unsafe_rate", "missed_rate",
                                                   "inside_rate"]].mean().reset_index()
          if len(look) else look)
    dps = dec["paired"]
    pt = _cat(runs, "pertarget")
    svp = shared_vs_pertarget(cap, pt, first["args"]["target_name"]) if len(pt) else []
    ab = _cat(runs, "ablation")
    ab_t = pd.DataFrame()
    if len(ab):
        ab_t = ab.groupby(["target", "device"]).agg(
            seeds=("seed", "nunique"), auroc_intact=("auroc_intact", "mean"),
            auroc_vessels=("auroc_vessels", "mean"), auroc_control=("auroc_control", "mean"),
            vessels_minus_control=("vessels_minus_control", "mean"),
            vmc_sd=("vessels_minus_control", "std"),
            seeds_ci_below_0=("vmc_hi", lambda v: int((v < 0).sum()))).reset_index()

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
          "Envelope (95th pct same-camera distance), out-of-sample recognition and false "
          "acceptance by seed:", "",
          "```", json.dumps({"envelope": env, "recognition": rec, "false_accept": fa}, indent=1,
                            default=float), "```", "",
          "## 4. Controls and mechanism", "",
          "Per target (calibrated protocol, mean over seeds): permuted-label probe AUROC (must be "
          "~0.5), quality-flags-only AUROC, intake-form AUROC, image AUROC, ECE, mean predicted "
          "risk vs prevalence.", "",
          md_table(ctl), "",
          "Negative-control runs (source labels permuted in training; must sit at ~0.5): " +
          (f"{len(shuf)} run(s)" if len(shuf) else "none found (run with SHUFFLE=1)"), "",
          md_table(shuf_t) if len(shuf_t) else "", "",
          "Camera decodability from the trunk embedding (patient-grouped linear read-out AUROC; "
          "1.0 = the representation encodes the camera):", "",
          (f"trained {np.mean(list(dec['trained'].values())):.3f}, calibrated "
           f"{np.mean(list(dec['calibrated'].values())):.3f}; paired calibrated − trained "
           f"{dps['mean_delta']:+.3f} [{dps['ci95'][0]:+.3f}, {dps['ci95'][1]:+.3f}] "
           f"({dps['n']} seeds)") if dps["n"] else "(not recorded)", "",
          "Gradable vs ungradable images:", "", md_table(st) if len(st) else "(none)", "",
          "Vessel ablation (calibrated protocol; vessels_minus_control < 0 = the target leans on "
          "the vasculature more than on the same area of other retina; seeds_ci_below_0 = runs "
          "whose patient-bootstrap CI excludes 0):", "",
          md_table(ab_t) if len(ab_t) else "(not run)", "",
          "Shared trunk vs per-target models (handheld test rows, paired by seed; fine-tuned "
          "per-target train_mbrset.py model minus the shared-trunk calibrated probe):", "",
          md_table(pd.DataFrame(svp)) if svp else "(no PERTARGET runs)", "",
          "## 5. Unfamiliar cameras (labels used only to score)", "",
          md_table(unf_ct) if len(unf_ct) else "(no --unfamiliar camera in these runs)", "",
          md_table(pd.DataFrame(unf_tr)) if unf_tr else "", "",
          "Phone lookup vs the camera's own labels (mean over seeds and draws): shown = the "
          "target is displayed; unsafe = displayed but not SUPPORTED by the labels; missed = "
          "SUPPORTED but hidden.", "",
          md_table(lk) if len(lk) else ""]
    os.makedirs(a.dir, exist_ok=True)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    ct.to_csv(os.path.join(a.dir, "capability_final.csv"), index=False)
    with open(os.path.join(a.dir, "summary.json"), "w") as f:
        json.dump({"seeds": seeds, "transfer": tr, "capability": ct.to_dict("records"),
                   "sweep": sw.to_dict("records") if len(sw) else [],
                   "controls": ctl.to_dict("records") if len(ctl) else [],
                   "shuffle_runs": shuf_t.to_dict("records") if len(shuf_t) else [],
                   "decodability": dec,
                   "quality_strata": st.to_dict("records") if len(st) else [],
                   "anatomy_ablation": ab_t.to_dict("records") if len(ab_t) else [],
                   "shared_vs_pertarget": svp,
                   "unfamiliar_capability": unf_ct.to_dict("records") if len(unf_ct) else [],
                   "unfamiliar_transfer": unf_tr,
                   "unfamiliar_lookup": lk.to_dict("records") if len(lk) else []},
                  f, indent=2, default=float)
    print("\n".join(md[:4 + 1]))
    print(pd.DataFrame(tr).to_string(index=False) if tr else "")
    print("\n=== final capability ===")
    print(ct[["target", "device", "protocol", "final", "status_counts", "auroc", "meta_auroc",
              "delta", "noise_floor", "sensitivity"]].to_string(index=False, float_format=ff))
    print(f"\nwrote {a.dir}/summary.md, capability_final.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
