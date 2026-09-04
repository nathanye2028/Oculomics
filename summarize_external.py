#!/usr/bin/env python3
"""
summarize_external.py
=====================
Paired treatment-vs-control statistics on the EXTRA external datasets scored by
``score_external.py`` (files ``<condition>_seed<n>_on_<dataset>.json``), one
block per dataset: zero-shot AUROC and, when present, AdaBN AUROC, paired by
seed. Complements ``summarize_xfer.py`` (which covers the trainer's primary
external set).

    python summarize_external.py --dir exp_glaucoma --treatment kd --control ctrl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats          # noqa: E402

_PAT = re.compile(r"^(?P<cond>.+?)_seed(?P<seed>\d+)_on_(?P<ds>.+)\.json$")


def load(d: str) -> Dict[str, Dict[str, Dict[int, dict]]]:
    """{dataset: {condition: {seed: record}}}"""
    out: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_on_*.json"))):
        m = _PAT.match(os.path.basename(p))
        if not m:
            continue
        with open(p) as f:
            out.setdefault(m["ds"], {}).setdefault(m["cond"], {})[int(m["seed"])] = json.load(f)
    return out


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt(st, label):
    lo, hi = st["ci95"]
    if st["n"] < 2:
        return f"  {label:<30} delta={st['mean_delta']:+.4f}  no CI (n={st['n']})"
    verdict = "SIGNIFICANT" if st["significant"] else "inconclusive (CI includes 0)"
    return (f"  {label:<30} delta={st['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
            f"{st['n_positive']}/{st['n']} seeds+  {verdict}")


def summarize(root: str, control: str, treatment: str = None) -> dict:
    data = load(root)
    if not data:
        raise SystemExit(f"[fatal] no <cond>_seed<n>_on_<dataset>.json under {root}")
    L = [f"\n{'='*78}", f"EXTRA EXTERNAL SETS  --  {root}   control={control}"
         + (f"  treatment={treatment}" if treatment else ""), f"{'='*78}"]
    report = {}
    for ds, runs in data.items():
        if control not in runs:
            L.append(f"\n[{ds}] no {control!r} runs; skipped"); continue
        conds = [control] + ([treatment] if treatment and treatment in runs else [])
        seeds = sorted(set.intersection(*(set(runs[c]) for c in conds)))
        rec = {"seeds": seeds, "conditions": {}, "paired_zero_shot": None, "paired_bnadapt": None,
               "n_external": runs[control][seeds[0]].get("n_external") if seeds else None}
        r0 = runs[control][seeds[0]] if seeds else None
        prev = (r0["external_pos"] / r0["n_external"]) if r0 and r0.get("external_pos") is not None and r0.get("n_external") else None
        L += ["", f"--- {ds} ---  paired seeds={seeds}  n={rec['n_external']}"
              + (f"  prevalence={prev:.1%}" if prev is not None else ""),
              f"  {'condition':<10}{'zero-shot AUROC':<22}{'AdaBN AUROC':<22}"]
        for c in conds:
            zs = {s: runs[c][s]["external"]["auroc"] for s in seeds}
            ad = {s: runs[c][s]["external_bnadapt"]["auroc"] for s in seeds if runs[c][s].get("external_bnadapt")}
            rec["conditions"][c] = {"zero_shot": zs, "bnadapt": ad}
            zm, zsd = agg(zs.values()); am, asd = agg(ad.values())
            L.append(f"  {c:<10}{f'{zm:.4f}+/-{zsd:.4f}':<22}{(f'{am:.4f}+/-{asd:.4f}' if ad else '-'):<22}")
        if treatment in conds:
            st = paired_stats(rec["conditions"][treatment]["zero_shot"], rec["conditions"][control]["zero_shot"])
            rec["paired_zero_shot"] = {**st, "ci95": list(st["ci95"])}
            L.append(_fmt(st, f"{treatment}-{control} zero-shot"))
            if rec["conditions"][treatment]["bnadapt"] and rec["conditions"][control]["bnadapt"]:
                st = paired_stats(rec["conditions"][treatment]["bnadapt"], rec["conditions"][control]["bnadapt"])
                rec["paired_bnadapt"] = {**st, "ci95": list(st["ci95"])}
                L.append(_fmt(st, f"{treatment}-{control} AdaBN"))
        report[ds] = rec
    L += ["", "NB: zero-shot is the deployable number; AdaBN is transductive (statistics from the",
          "    scored images) and is reported separately. AUROC is rank-based; prevalence is context."]
    out = "\n".join(L)
    print(out)
    with open(os.path.join(root, "summary_external.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(root, "summary_external.json"), "w") as f:
        json.dump({"control": control, "treatment": treatment, "datasets": report}, f, indent=2)
    print(f"\n[info] wrote {root}/summary_external.md and summary_external.json")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise score_external.py results per dataset.")
    p.add_argument("--dir", default="exp_glaucoma")
    p.add_argument("--control", default="ctrl")
    p.add_argument("--treatment", default="kd")
    args = p.parse_args()
    if not os.path.isdir(args.dir):
        print(f"[fatal] {args.dir} is not a directory", file=sys.stderr); return 2
    summarize(args.dir, args.control, args.treatment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
