#!/usr/bin/env python3
"""
summarize_ophthalmic.py
=======================
Per-label summary of the BRSET ophthalmic runs written by ``run_ophthalmic.sh``:
one multi-label model (``multi_seed<n>.json``, task ``ophthalmic``) against one
dedicated binary model per label (``single_<label>_seed<n>.json``). In-domain on
BRSET only -- mBRSET has none of these labels, so there is no external block.

The paired question per label, paired by seed (same patient split):
**does joint multi-label training help or hurt this label** versus a model
trained for it alone? Rare labels (retinal detachment, vascular occlusion) are
where sharing a trunk is expected to help; the CI says whether it did.

    python summarize_ophthalmic.py --dir exp_ophthalmic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats          # noqa: E402  (same statistics as every other experiment)


def load(d: str) -> Dict[str, Dict[int, dict]]:
    """{condition: {seed: record}}; condition is 'multi' or 'single_<label>'."""
    out: Dict[str, Dict[int, dict]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        base = os.path.basename(p)[: -len(".json")]
        cond, _, seed = base.rpartition("_seed")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(p) as f:
            out.setdefault(cond, {})[seed] = json.load(f)
    return out


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt(st, label):
    lo, hi = st["ci95"]
    if st["n"] < 2:
        return f"  {label:<26} delta={st['mean_delta']:+.4f}  no CI (n={st['n']})"
    verdict = "SIGNIFICANT" if st["significant"] else "inconclusive (CI includes 0)"
    return (f"  {label:<26} delta={st['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
            f"{st['n_positive']}/{st['n']} seeds+  {verdict}")


def summarize(root: str, labels: Optional[List[str]] = None, multi: str = "multi",
              single_prefix: str = "single_") -> dict:
    runs = load(root)
    if multi not in runs and not any(c.startswith(single_prefix) for c in runs):
        raise SystemExit(f"[fatal] no {multi}_seed<n>.json or {single_prefix}<label>_seed<n>.json in {root}")
    m_runs = runs.get(multi, {})
    if labels is None:
        labels = list(next(iter(m_runs.values()))["label_names"]) if m_runs else \
            sorted(c[len(single_prefix):] for c in runs if c.startswith(single_prefix))
    L = [f"\n{'='*80}", f"BRSET OPHTHALMIC LABELS: multi-label head vs one model per label  --  {root}",
         f"  multi seeds={sorted(m_runs)}   labels={labels}", f"{'='*80}",
         f"\n{'label':<26}{'single AUROC':<20}{'multi AUROC':<20}{'multi-single':<14}{'prev':>7}{'n_test':>8}"]
    report: Dict[str, dict] = {"labels": {}, "multi_macro_auroc": None}
    for lab in labels:
        s_runs = runs.get(f"{single_prefix}{lab}", {})
        single = {s: r["test"]["auroc"] for s, r in s_runs.items()}
        multi_l = {s: (r["test"].get("per_label_auroc") or {}).get(lab) for s, r in m_runs.items()}
        multi_l = {s: v for s, v in multi_l.items() if v is not None and v == v}
        seeds = sorted(set(single) & set(multi_l))
        prev, n_test = None, None
        src = s_runs or m_runs
        if src:
            r0 = src[sorted(src)[0]]
            tp = r0.get("test_pos")
            tp = tp.get(lab) if isinstance(tp, dict) else tp
            n_test = r0.get("n_test")
            prev = (tp / n_test) if tp is not None and n_test else None
        st = paired_stats(multi_l, single) if seeds else None
        sm, ssd = agg(single.values()); mm, msd = agg(multi_l.values())
        delta = f"{st['mean_delta']:+.4f}" if st else "-"
        report["labels"][lab] = {"single_auroc": single, "multi_auroc": multi_l, "paired_seeds": seeds,
                                 "paired_multi_minus_single": ({**st, "ci95": list(st["ci95"])} if st else None),
                                 "prevalence": prev, "n_test": n_test}
        L.append(f"{lab:<26}{(f'{sm:.4f}+/-{ssd:.4f}' if single else '-'):<20}"
                 f"{(f'{mm:.4f}+/-{msd:.4f}' if multi_l else '-'):<20}"
                 f"{delta:<14}"
                 f"{(f'{prev:.1%}' if prev is not None else '-'):>7}{(n_test or 0):>8}")
    L += ["", "-" * 80, "PAIRED multi minus single (same seed, same patient split):"]
    for lab, rec in report["labels"].items():
        st = rec["paired_multi_minus_single"]
        L.append(_fmt(st, lab) if st else f"  {lab:<26} not paired (missing single or multi runs)")
    if m_runs:
        mac = {s: r["test"]["auroc"] for s, r in m_runs.items()}
        mm, msd = agg(mac.values())
        report["multi_macro_auroc"] = mac
        L += ["", f"multi-label macro AUROC over its labels: {mm:.4f}+/-{msd:.4f} (n={len(mac)} seeds)"]
    L += ["", "NB: in-domain on BRSET (tabletop, one site). A label with a prevalence under ~1%",
          "    has a handful of positives in the test split; its AUROC is wide regardless of",
          "    the model. Quote the CI, not the point estimate."]
    out = "\n".join(L)
    print(out)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[info] wrote {root}/summary.md and summary.json")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise run_ophthalmic.sh results per label.")
    p.add_argument("--dir", default="exp_ophthalmic")
    p.add_argument("--labels", nargs="+", default=None, help="Labels (default: from the multi run).")
    p.add_argument("--multi", default="multi", help="Condition prefix of the multi-label runs.")
    p.add_argument("--single-prefix", default="single_", help="Prefix of the per-label runs.")
    args = p.parse_args()
    if not os.path.isdir(args.dir):
        print(f"[fatal] {args.dir} is not a directory", file=sys.stderr)
        return 2
    summarize(args.dir, args.labels, args.multi, args.single_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
