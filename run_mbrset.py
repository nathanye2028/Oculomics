"""
run_mbrset.py
=============
GCG vs control on **mBRSET** (5,164 smartphone fundus images) — the
well-powered counterpart to the IDRiD segmentation benchmark.

Runs the same MobileNetV3-Small backbone with gating ON vs OFF across seeds,
then reports **paired** per-seed deltas with 95% CIs (same statistics as
``run_experiment.py``, reused directly).

Primary metric is **AUROC** on a patient-grouped held-out test split.

    python run_mbrset.py --root <mBRSET dir> --task dr_referable --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_mbrset.py")
sys.path.insert(0, HERE)
from run_experiment import paired_stats  # noqa: E402  (identical statistics)

CONDITIONS = [("gcg", []), ("nogcg", ["--no-gcg"])]
METRICS = ["auroc", "acc", "f1", "kappa"]


def main() -> int:
    p = argparse.ArgumentParser(description="GCG vs control on mBRSET classification.")
    p.add_argument("--root", required=True)
    p.add_argument("--task", default="dr_referable",
                   choices=["dr_referable", "dr_binary", "dr_grade", "edema",
                            "quality", "artifacts"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gcg-variant", default="baseline")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--out-dir", default="exp_mbrset")
    p.add_argument("--ckpt-dir", default="ck_mbrset")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    res = {c: {m: {} for m in METRICS} for c, _ in CONDITIONS}
    failures = []

    total = len(CONDITIONS) * len(args.seeds)
    i = 0
    for cond, flags in CONDITIONS:
        for seed in args.seeds:
            i += 1
            run_name = f"{cond}_seed{seed}"
            rj = os.path.join(args.out_dir, f"{run_name}.json")
            cmd = [sys.executable, TRAINER, "--root", args.root, "--task", args.task,
                   "--seed", str(seed), "--epochs", str(args.epochs),
                   "--patience", str(args.patience), "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr), "--image-size", str(args.image_size),
                   "--num-workers", str(args.num_workers), "--gcg-variant", args.gcg_variant,
                   *(["--nondeterministic"] if args.nondeterministic else []),
                   "--ckpt-dir", args.ckpt_dir, "--run-name", run_name,
                   "--results-json", rj] + flags
            print(f"\n[{i}/{total}] === {run_name} ===\n  {' '.join(cmd)}", flush=True)
            if subprocess.run(cmd).returncode != 0 or not os.path.exists(rj):
                print(f"  !! failed — skipping"); failures.append(run_name); continue
            r = json.load(open(rj))
            for m in METRICS:
                res[cond][m][seed] = r["test"][m]

    lines = [f"\n{'='*72}",
             f"GCG vs CONTROL on mBRSET  task={args.task}  seeds={args.seeds}  epochs={args.epochs}",
             f"{'='*72}",
             f"{'metric':<10}{'control (no-gcg)':<24}{'GCG':<24}{'delta':>9}"]
    for m in METRICS:
        cv, gv = list(res["nogcg"][m].values()), list(res["gcg"][m].values())
        if not cv or not gv:
            continue
        cm, cs = statistics.mean(cv), (statistics.pstdev(cv) if len(cv) > 1 else 0.0)
        gm, gs = statistics.mean(gv), (statistics.pstdev(gv) if len(gv) > 1 else 0.0)
        lines.append(f"{m:<10}{f'{cm:.4f} +/- {cs:.4f}':<24}{f'{gm:.4f} +/- {gs:.4f}':<24}{gm-cm:>+9.4f}")

    lines += ["", "-" * 72, "PAIRED analysis (per-seed GCG-minus-control):"]
    verdict = "inconclusive"
    for m in METRICS:
        ps = paired_stats(res["gcg"][m], res["nogcg"][m])
        lo, hi = ps["ci95"]
        sig = "  *sig" if ps["significant"] else ""
        lines.append(f"  {m:<7} delta={ps['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                     f"{ps['n_positive']}/{ps['n']} seeds+{sig}")
        if m == "auroc":
            if ps["significant"] and ps["mean_delta"] > 0:
                verdict = (f"GCG improves AUROC by {ps['mean_delta']:+.4f} "
                           f"(95% CI excludes 0, paired n={ps['n']})")
            elif ps["significant"]:
                verdict = f"GCG reduces AUROC by {ps['mean_delta']:.4f} (95% CI excludes 0)"
            elif ps["n"] and ps["n_positive"] == ps["n"]:
                verdict = f"GCG positive in all {ps['n']} seeds but CI includes 0 -> add seeds"
    if failures:
        lines.append(f"\n(!) failed runs: {failures}")
    lines.append(f"\nverdict (AUROC): {verdict}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("```\n" + report + "\n```\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"task": args.task, "seeds": args.seeds, "results": res,
                   "paired": {m: paired_stats(res["gcg"][m], res["nogcg"][m]) for m in METRICS},
                   "failures": failures}, f, indent=2)
    print(f"\n[info] wrote {args.out_dir}/summary.md and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
