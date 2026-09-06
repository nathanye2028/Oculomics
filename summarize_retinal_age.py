#!/usr/bin/env python3
"""
summarize_retinal_age.py
========================
Aggregate ``train_retinal_age.py`` runs named ``<condition>_seed<n>.json``
(``run_retinal_age.sh`` writes ``student_seed<n>`` and optionally
``teacher_seed<n>``): mean ± SD per condition of the in-domain healthy MAE, the
whole-test MAE, the mBRSET MAE zero-shot and after AdaBN, Pearson r, the
non-healthy-minus-healthy corrected gap, and the **MAE-by-age-bin** table
averaged over seeds. If two conditions share seeds, their MAE difference is
paired by seed (``run_experiment.paired_stats``).

It also pools the per-run predictions tables into ``<dir>/predictions_pooled.csv``
— per image, the mean prediction / gap / corrected gap across seeds (a seed
ensemble) with ``n_seeds`` — which is the input for the disease-association
step. Every run's own table (``<ckpt-dir>/<run>_predictions.csv``) stays as is.

    python summarize_retinal_age.py --dir exp_retinal_age
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats               # noqa: E402
from train_retinal_age import AGE_BIN_LABELS          # noqa: E402

SETS = [("test_healthy", "BRSET test, healthy"), ("test_all", "BRSET test, all"),
        ("test_nonhealthy", "BRSET test, non-healthy"), ("unseen_nonhealthy", "BRSET non-healthy, never trained"),
        ("external", "mBRSET zero-shot"), ("external_bnadapt", "mBRSET + AdaBN")]


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        base = os.path.basename(p)[:-5]
        cond, _, seed = base.rpartition("_seed")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(p) as f:
            r = json.load(f)
        if r.get("task") != "retinal_age":
            continue
        out.setdefault(cond, {})[seed] = r
    return out


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), 0.0, 0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(m, sd, n, prec=2):
    return "-" if n == 0 else (f"{m:.{prec}f} ± {sd:.{prec}f}" if n > 1 else f"{m:.{prec}f}")


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise retinal-age runs (mean ± SD over seeds, "
                                            "MAE by age bin, pooled predictions).")
    p.add_argument("--dir", default="exp_retinal_age")
    p.add_argument("--no-pool", action="store_true", help="Skip the pooled predictions CSV.")
    args = p.parse_args()

    runs = load(args.dir)
    if not runs:
        print(f"[fatal] no retinal_age result JSONs in {args.dir}", file=sys.stderr)
        return 2
    L = [f"\n{'='*78}", f"RETINAL AGE: {args.dir}", f"{'='*78}"]
    for cond, by_seed in runs.items():
        seeds = sorted(by_seed)
        r0 = by_seed[seeds[0]]
        L.append(f"\n## {cond}   seeds={seeds}   backbone={r0.get('backbone')}   "
                 f"healthy={r0.get('healthy')}   params={r0.get('params_m')}M")
        c = r0.get("cohort", {})
        L.append(f"   cohort: {c.get('n_healthy_images')}/{c.get('n_images')} healthy images, "
                 f"{c.get('n_healthy_patients')}/{c.get('n_patients')} patients; "
                 f"excluded {c.get('exclusions')}")
        L.append(f"\n   {'set':<36}{'MAE (y)':<16}{'r':<16}{'mean gap':<16}{'corrected gap':<16}{'n':>6}")
        for key, label in SETS:
            recs = [by_seed[s].get(key) for s in seeds]
            recs = [x for x in recs if x and x.get("n")]
            if not recs:
                continue
            mae = agg([x["mae"] for x in recs]); r = agg([x["r"] for x in recs])
            g = agg([x["mean_gap"] for x in recs]); gc = agg([x.get("mean_gap_corrected") for x in recs])
            L.append(f"   {label:<36}{fmt(*mae):<16}{fmt(*r, prec=3):<16}{fmt(*g):<16}{fmt(*gc):<16}"
                     f"{recs[0]['n']:>6}")
        # non-healthy minus healthy corrected gap, per seed then averaged
        deltas = []
        for s in seeds:
            a, b = by_seed[s].get("test_nonhealthy") or {}, by_seed[s].get("test_healthy") or {}
            if a.get("n") and b.get("n"):
                deltas.append(a["mean_gap_corrected"] - b["mean_gap_corrected"])
        if deltas:
            m, sd, n = agg(deltas)
            L.append(f"\n   non-healthy minus healthy corrected gap (BRSET test): {fmt(m, sd, n)} y "
                     f"over {n} seed(s)  [descriptive; the association step tests this properly]")
        # by-age-bin tables
        for key, label in (("test_healthy", "BRSET test healthy"), ("external", "mBRSET zero-shot"),
                           ("external_bnadapt", "mBRSET + AdaBN")):
            recs = [by_seed[s].get(key) for s in seeds]
            recs = [x for x in recs if x and x.get("by_bin")]
            if not recs:
                continue
            L.append(f"\n   MAE by age bin — {label} (mean over {len(recs)} seed(s); n from seed {seeds[0]})")
            L.append(f"   {'bin':<8}" + "".join(f"{lab:>12}" for lab in AGE_BIN_LABELS))
            L.append(f"   {'n':<8}" + "".join(f"{recs[0]['by_bin'].get(lab, {}).get('n', 0):>12}"
                                              for lab in AGE_BIN_LABELS))
            for stat in ("mae", "mean_gap"):
                row = []
                for lab in AGE_BIN_LABELS:
                    m, sd, n = agg([x["by_bin"].get(lab, {}).get(stat) for x in recs])
                    row.append("-" if n == 0 else (f"{m:+.1f}" if stat == "mean_gap" else f"{m:.1f}"))
                L.append(f"   {stat:<8}" + "".join(f"{v:>12}" for v in row))
        bc = [by_seed[s].get("bias_correction") for s in seeds]
        bc = [x for x in bc if x]
        if bc:
            a, b = agg([x["a"] for x in bc]), agg([x["b"] for x in bc])
            L.append(f"\n   bias correction gap = a + b*age (fit on healthy val): "
                     f"a={fmt(*a)}  b={fmt(*b, prec=4)}  (a negative b = regression to the mean)")

    # paired condition contrast on MAE (teacher - student etc.)
    conds = sorted(runs)
    if len(conds) >= 2:
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                a, b = conds[i], conds[j]
                for key, label in (("test_healthy", "BRSET healthy MAE"), ("external", "mBRSET MAE"),
                                   ("external_bnadapt", "mBRSET+AdaBN MAE")):
                    da = {s: runs[a][s][key]["mae"] for s in runs[a] if runs[a][s].get(key)}
                    db = {s: runs[b][s][key]["mae"] for s in runs[b] if runs[b][s].get(key)}
                    ps = paired_stats(db, da)
                    if ps["n"]:
                        lo, hi = ps["ci95"]
                        L.append(f"\n   paired {b} − {a} on {label}: {ps['mean_delta']:+.2f} y  "
                                 f"95% CI [{lo:+.2f}, {hi:+.2f}]  {ps['n_positive']}/{ps['n']} seeds positive"
                                 f"{'  (significant)' if ps['significant'] else ''}  "
                                 f"(negative = {b} better)")

    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write("```\n" + text.strip("\n") + "\n```\n")
    print(f"\n[info] wrote {os.path.join(args.dir, 'summary.md')}")

    if not args.no_pool:
        frames = []
        for cond, by_seed in runs.items():
            for s, r in by_seed.items():
                pth = r.get("predictions_csv")
                if pth and os.path.isfile(pth):
                    df = pd.read_csv(pth); df["condition"] = cond; df["seed"] = s
                    frames.append(df)
                else:
                    print(f"[warn] {cond}_seed{s}: predictions CSV missing ({pth})", file=sys.stderr)
        if frames:
            allp = pd.concat(frames, ignore_index=True)
            num = [c for c in ("pred_age", "gap", "gap_corrected", "pred_age_bnadapt", "gap_bnadapt",
                               "gap_bnadapt_corrected") if c in allp.columns]
            keys = ["condition", "dataset", "file"]
            meta = [c for c in allp.columns if c not in num + keys + ["seed"]]
            pooled = allp.groupby(keys, as_index=False).agg(
                **{c: (c, "mean") for c in num}, n_seeds=("seed", "nunique"),
                **{c: (c, "first") for c in meta})
            out = os.path.join(args.dir, "predictions_pooled.csv")
            pooled.to_csv(out, index=False)
            print(f"[info] wrote {out} ({len(pooled)} rows; per-image mean over seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
