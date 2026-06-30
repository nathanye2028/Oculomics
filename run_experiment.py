"""
run_experiment.py
=================
The headline experiment: **does Guided Context Gating beat the control on
IDRiD lesion segmentation?**

Runs the *same* GCG-U-Net backbone with gating ON vs OFF, across several seeds
(the IDRiD test set is only 27 images, so a single-run gap is noise — we need
mean +/- std over seeds), then prints a per-lesion comparison table with the
GCG-minus-control delta.

Each run is a fresh `train_idrid.py` subprocess (clean process, no memory
build-up, identical code path you'd run by hand). Results are collected from
the per-run JSON each subprocess writes.

Examples
--------
    # Local (MPS / single GPU), 3 seeds, full convergence:
    python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled --amp

    # Quick end-to-end sanity (tiny, just checks the harness + table):
    python run_experiment.py --seeds 0 --epochs 2 --image-size 128 --quick

    # Multi-GPU per run (cloud): launch each condition via torchrun
    python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled \
        --amp --launcher torchrun --nproc 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_idrid.py")

# The two conditions under test: identical backbone, gating is the only variable.
CONDITIONS = [("gcg", []), ("nogcg", ["--no-gcg"])]


def build_cmd(args, cond_flags, seed, results_json, run_name):
    base = [sys.executable, TRAINER]
    if args.launcher == "torchrun":
        base = ["torchrun", f"--nproc_per_node={args.nproc}",
                f"--master_port={args.master_port}", TRAINER]
    cmd = base + [
        "--arch", "gcg_unet",
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--loss", args.loss,
        "--lesions", *args.lesions,
        "--num-workers", str(args.num_workers),
        "--ckpt-dir", args.ckpt_dir,
        "--run-name", run_name,
        "--results-json", results_json,
        "--gcg-variant", args.gcg_variant,
    ]
    if args.patch_size > 0:
        cmd += ["--patch-size", str(args.patch_size)]
    else:
        cmd += ["--image-size", str(args.image_size)]
    if args.eval_tiled:
        cmd += ["--eval-tiled"]
    if args.amp:
        cmd += ["--amp"]
    if args.warmup_epochs > 0:
        cmd += ["--warmup-epochs", str(args.warmup_epochs)]
    if args.pretrained:
        cmd += ["--pretrained"]
    return cmd


def aggregate(per_lesion_runs):
    """{lesion: [dice,...]} -> {lesion: (mean, std)}."""
    out = {}
    for lesion, vals in per_lesion_runs.items():
        if not vals:
            out[lesion] = (float("nan"), float("nan"))
        else:
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out[lesion] = (m, s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="GCG vs control benchmark on IDRiD.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=0)
    p.add_argument("--eval-tiled", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--loss", default="focal_tversky")
    p.add_argument("--gcg-variant", default="baseline",
                   choices=["baseline", "attention", "cbam", "se", "none"],
                   help="GCG block for the 'gcg' condition (vs the --no-gcg control).")
    p.add_argument("--pretrained", action="store_true",
                   help="ImageNet-pretrained encoder (strongly recommended: only 54 train images).")
    p.add_argument("--lesions", nargs="+", default=["MA", "HE", "EX", "SE"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--launcher", default="python", choices=["python", "torchrun"])
    p.add_argument("--nproc", type=int, default=2, help="GPUs per run when --launcher torchrun.")
    p.add_argument("--master-port", type=int, default=29570)
    p.add_argument("--quick", action="store_true", help="Tiny run to validate the harness.")
    args = p.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.image_size = min(args.image_size, 128)
        args.patch_size = 0

    os.makedirs(args.out_dir, exist_ok=True)
    # results[cond][lesion] = [dice across seeds]; means[cond] = [mean dice across seeds]
    results = {c: {les: [] for les in args.lesions} for c, _ in CONDITIONS}
    means = {c: [] for c, _ in CONDITIONS}
    failures = []

    total = len(CONDITIONS) * len(args.seeds)
    i = 0
    for cond, flags in CONDITIONS:
        for seed in args.seeds:
            i += 1
            run_name = f"{cond}_seed{seed}"
            rj = os.path.join(args.out_dir, f"{run_name}.json")
            cmd = build_cmd(args, flags, seed, rj, run_name) + flags
            print(f"\n[{i}/{total}] === {run_name} ===\n  {' '.join(cmd)}", flush=True)
            ret = subprocess.run(cmd).returncode
            if ret != 0 or not os.path.exists(rj):
                print(f"  !! run failed (exit {ret}) — skipping", flush=True)
                failures.append(run_name)
                continue
            with open(rj) as f:
                r = json.load(f)
            for les in args.lesions:
                results[cond][les].append(r["test_dice"].get(les, float("nan")))
            means[cond].append(r["test_mean"])

    # ---- comparison table ----
    agg = {c: aggregate(results[c]) for c, _ in CONDITIONS}
    mean_stats = {c: aggregate({"mean": means[c]})["mean"] for c, _ in CONDITIONS}

    lines = []
    lines.append(f"\n{'='*64}")
    lines.append(f"GCG vs CONTROL on IDRiD  (seeds={args.seeds}, epochs={args.epochs})")
    lines.append(f"{'='*64}")
    lines.append(f"{'lesion':<8}{'control (no-gcg)':<22}{'GCG':<22}{'delta':>8}")
    for les in args.lesions:
        cm, cs = agg["nogcg"][les]
        gm, gs = agg["gcg"][les]
        lines.append(f"{les:<8}{f'{cm:.3f} +/- {cs:.3f}':<22}{f'{gm:.3f} +/- {gs:.3f}':<22}{gm-cm:>+8.3f}")
    cm, cs = mean_stats["nogcg"]; gm, gs = mean_stats["gcg"]
    lines.append("-" * 64)
    lines.append(f"{'mean':<8}{f'{cm:.3f} +/- {cs:.3f}':<22}{f'{gm:.3f} +/- {gs:.3f}':<22}{gm-cm:>+8.3f}")
    if failures:
        lines.append(f"\n(!) failed runs: {failures}")
    verdict = ("GCG helps" if gm - cm > max(cs, gs) else
               "inconclusive (delta within std)" if gm >= cm else "GCG does not help")
    lines.append(f"\nverdict (mean): {verdict}")
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("```\n" + report + "\n```\n")
    # machine-readable summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"agg": {c: agg[c] for c, _ in CONDITIONS},
                   "mean": mean_stats, "seeds": args.seeds,
                   "epochs": args.epochs, "failures": failures}, f, indent=2)
    print(f"\n[info] wrote {args.out_dir}/summary.md and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
