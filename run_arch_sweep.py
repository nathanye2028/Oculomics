#!/usr/bin/env python3
"""
run_arch_sweep.py
=================
**Does the architecture change accuracy?** — the missing half of the mobile story.

``export_coreml.py`` answers "what does it cost on device"; this answers "what does
it score". Together they give the Dice-vs-latency frontier that a mobile-first
project should be reporting.

Design: **one factor at a time.** Every variant differs from ``baseline`` (the
architecture every pre-2026-08 result was measured with) in exactly one respect,
so a delta is attributable:

    baseline    mobilenetv3      dense      lat=0     <- reference
    lateral     mobilenetv3      dense      lat=256   <- the 1x1 deep projection
    separable   mobilenetv3      separable  lat=0     <- depthwise-separable fuse convs
    mnv4_m      mobilenetv4_m    dense      lat=0     <- the encoder
    evit_b1     efficientvit_b1  dense      lat=0     <- the encoder

GCG is held FIXED across variants (on by default), because this sweep asks about
architecture, not about gating. Keep using ``run_experiment.py`` for GCG-vs-control.

Comparisons are **paired by seed** against ``baseline`` and reuse
``run_experiment``'s statistics, so the numbers are computed the same way as the
GCG table.

Statistical power — read this before trusting a result
-----------------------------------------------------
IDRiD's test set is **27 images**, and its val split is **11**. The existing GCG
runs produced 95% CIs spanning +/-0.06 to +/-0.13 mean Dice on 3 seeds. An
architecture effect smaller than that is *unmeasurable* on IDRiD alone, and you
will get another "CI includes 0".

FGADR gives a 367-image test split. **Score on FGADR** (the default when FGADR is
in ``--datasets``) or accept that IDRiD-only results are directional at best.

    # Stage 1 — cheap ordering + end-to-end plumbing check (IDRiD only, underpowered)
    python run_arch_sweep.py --datasets idrid --seeds 0 1 2 --epochs 120 \
        --patch-size 512 --eval-tiled --amp

    # Stage 2 — the number you actually report (FGADR-scored; expensive)
    python run_arch_sweep.py --datasets idrid fgadr --seeds 0 1 2 --epochs 200 \
        --patch-size 512 --eval-tiled --amp --variants baseline lateral mnv4_m

    # See the commands and the cost table without training anything
    python run_arch_sweep.py --dry-run

Cost columns (params, GMAC) are computed locally with no GPU and no training.
**GMAC is not latency** on Apple silicon — see model_seg.py's module docstring;
get real latency from ``export_coreml.py`` on macOS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import aggregate, paired_stats          # noqa: E402

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_idrid.py")
BASELINE = "baseline"

# label -> (encoder, decoder, lateral_channels). Each differs from BASELINE in
# exactly one field; keep that property when adding rows.
VARIANTS = {
    "baseline":  ("mobilenetv3",     "dense",     0),
    "lateral":   ("mobilenetv3",     "dense",     256),
    "separable": ("mobilenetv3",     "separable", 0),
    "mnv4_m":    ("mobilenetv4_m",   "dense",     0),
    "evit_b1":   ("efficientvit_b1", "dense",     0),
}


def cost_of(label: str, num_classes: int, size: int = 512):
    """(params_M, GMAC) for a variant. CPU-only, untrained, no download."""
    import torch
    import torch.nn as nn
    from model_seg import build_model

    # Some cluster CPUs lack the ISA NNPACK wants, and torch then logs a warning
    # per conv per forward — hundreds of lines that bury the cost table this
    # function exists to print. We only need shapes here, so turn it off.
    try:
        torch.backends.nnpack.set_flags(False)
    except Exception:                       # noqa: BLE001 - cosmetic only
        pass

    enc, dec, lat = VARIANTS[label]
    net = build_model(arch="gcg_unet", num_classes=num_classes, pretrained=False,
                      use_gcg=True, encoder=enc, decoder=dec,
                      lateral_channels=None if lat < 0 else lat).eval()
    tally = []

    def hook(m, _i, o):
        tally.append(m.in_channels // m.groups * m.out_channels
                     * m.kernel_size[0] * m.kernel_size[1] * o.shape[-2] * o.shape[-1])

    handles = [m.register_forward_hook(hook)
               for m in net.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        net(torch.randn(1, 3, size, size))
    for h in handles:
        h.remove()
    return sum(p.numel() for p in net.parameters()) / 1e6, sum(tally) / 1e9


def build_cmd(args, label, seed, results_json, run_name):
    enc, dec, lat = VARIANTS[label]
    base = [sys.executable, TRAINER]
    if args.launcher == "torchrun":
        base = ["torchrun", f"--nproc_per_node={args.nproc}",
                f"--master_port={args.master_port}", TRAINER]
    cmd = base + [
        "--arch", "gcg_unet",
        "--encoder", enc, "--decoder", dec, "--lateral-channels", str(lat),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--loss", args.loss,
        "--lesions", *args.lesions,
        "--num-workers", str(args.num_workers),
        "--ckpt-dir", args.ckpt_dir,
        "--run-name", run_name,
        "--results-json", results_json,
        "--gcg-variant", args.gcg_variant,
        "--datasets", *args.datasets,
    ]
    if args.no_gcg:
        cmd += ["--no-gcg"]
    if args.patch_size > 0:
        cmd += ["--patch-size", str(args.patch_size)]
    else:
        cmd += ["--image-size", str(args.image_size)]
    if args.eval_tiled:
        cmd += ["--eval-tiled"]
    if args.amp:
        cmd += ["--amp"]
    if args.init_encoder:
        cmd += ["--init-encoder", args.init_encoder]
    if args.init_weights:
        cmd += ["--init-weights", args.init_weights]
    return cmd


def pick_metric(args, result: dict):
    """Which test split to read. FGADR's 367 images >> IDRiD's 27 for power."""
    want = args.metric
    if want == "auto":
        want = "fgadr" if result.get("fgadr_test_dice") else "idrid"
    if want == "fgadr":
        d = result.get("fgadr_test_dice")
        if not d:
            raise KeyError("run has no fgadr_test_dice (was FGADR in --datasets?)")
        return d, "FGADR (367 imgs)"
    return result["test_dice"], "IDRiD (27 imgs)"


def main() -> int:
    p = argparse.ArgumentParser(description="Architecture sweep: does the arch change Dice?")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS), help="Subset to run (baseline is always included).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--datasets", nargs="+", default=["idrid"],
                   choices=["idrid", "eophtha", "retlesion", "fgadr"])
    p.add_argument("--metric", default="auto", choices=["auto", "idrid", "fgadr"],
                   help="Which test split to compare on. 'auto' prefers FGADR when present.")
    p.add_argument("--lesions", nargs="+", default=["MA", "HE", "EX", "SE"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=0)
    p.add_argument("--eval-tiled", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--loss", default="focal_tversky")
    p.add_argument("--gcg-variant", default="baseline",
                   help="Held fixed across variants; this sweep is about architecture.")
    p.add_argument("--no-gcg", action="store_true",
                   help="Run the whole sweep with gating off instead of on.")
    p.add_argument("--init-encoder", default=None)
    p.add_argument("--init-weights", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ckpt-dir", default="ck_arch")
    p.add_argument("--out-dir", default="exp_arch")
    p.add_argument("--launcher", default="python", choices=["python", "torchrun"])
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cost table and the commands; train nothing.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse a results JSON that already exists (resume a partial sweep).")
    args = p.parse_args()

    # Fail on run 0, not run 9. Children are spawned with sys.executable, so
    # launching this script with the wrong interpreter (conda base instead of
    # .venv) propagates that interpreter to every run and each one dies on
    # `import numpy`. Nine identical tracebacks scroll past, then a summary of
    # NaNs gets written that looks like a completed sweep with missing data.
    try:
        import torch                                          # noqa: F401
    except ImportError as e:
        venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
        print(f"[fatal] this interpreter cannot import torch: {sys.executable}\n"
              f"        ({e})\n"
              f"        Every run would be spawned with it and fail identically.\n"
              f"        Re-run with the project venv:\n"
              f"          {venv} {os.path.basename(__file__)} ...", file=sys.stderr)
        return 2

    variants = [BASELINE] + [v for v in args.variants if v != BASELINE]
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- cost table: no GPU, no training, always printed ------------------- #
    print(f"\n{'='*74}\nARCHITECTURE COST (untrained; {args.image_size if args.patch_size <= 0 else args.patch_size}px)"
          f"\n{'='*74}")
    print(f"{'variant':<12}{'encoder':<17}{'dec':<11}{'lat':>5}{'params':>9}{'GMAC':>8}")
    costs = {}
    for v in variants:
        enc, dec, lat = VARIANTS[v]
        try:
            pm, gm = cost_of(v, len(args.lesions),
                             args.patch_size if args.patch_size > 0 else args.image_size)
            costs[v] = {"params_m": round(pm, 3), "gmac": round(gm, 3)}
            print(f"{v:<12}{enc:<17}{dec:<11}{lat:>5}{pm:8.2f}M{gm:8.2f}")
        except Exception as e:            # e.g. timm missing
            costs[v] = {"error": str(e)}
            print(f"{v:<12}{enc:<17}{dec:<11}{lat:>5}   [cost unavailable: {e}]")
    print("\nNB: GMAC is NOT latency here — this net is bandwidth-bound. Measure with")
    print("    export_coreml.py --image-size <n> --quantize fp16   (see README).")

    if len(args.seeds) < 2:
        print("\n[warn] <2 seeds: no CI is computable. Use at least 3.")
    if "fgadr" not in args.datasets and args.metric != "idrid":
        print("\n[warn] IDRiD-only: 27 test / 11 val images. Prior 3-seed CIs spanned")
        print("       +/-0.06..0.13 Dice, so only very large effects will resolve.")
        print("       Add 'fgadr' to --datasets for a 367-image test split.")

    total = len(variants) * len(args.seeds)
    if args.dry_run:
        print(f"\n{'='*74}\nDRY RUN — {total} runs would execute\n{'='*74}")
        for v in variants:
            for s in args.seeds:
                rn = f"{v}_seed{s}"
                cmd = build_cmd(args, v, s, os.path.join(args.out_dir, f"{rn}.json"), rn)
                print(f"\n[{v} seed{s}]\n  {' '.join(cmd)}")
        return 0

    # ---- run ---------------------------------------------------------------- #
    results, means, failures = {}, {}, []
    i = 0
    for v in variants:
        results[v] = {les: {} for les in args.lesions}
        means[v] = {}
        for s in args.seeds:
            i += 1
            rn = f"{v}_seed{s}"
            rj = os.path.join(args.out_dir, f"{rn}.json")
            if args.skip_existing and os.path.isfile(rj):
                # Reuse only if the stored run matches the CURRENT config —
                # a same-named JSON from an older sweep with different
                # epochs/datasets/loss would silently pollute the paired stats.
                with open(rj) as f:
                    old_r = json.load(f)
                enc, dec, lat = VARIANTS[v]
                expect = {"epochs": args.epochs, "datasets": args.datasets,
                          "lesions": args.lesions, "loss": args.loss,
                          "encoder": enc, "decoder": dec,
                          "patch_size": (args.patch_size if args.patch_size > 0 else None),
                          "use_gcg": not args.no_gcg, "seed": s}
                mismatch = {k: (old_r.get(k), want) for k, want in expect.items()
                            if old_r.get(k) != want}
                if mismatch:
                    print(f"\n[{i}/{total}] === {rn} === existing {rj} was run with a "
                          f"DIFFERENT config {mismatch}; re-running.", flush=True)
                    cmd = build_cmd(args, v, s, rj, rn)
                    if subprocess.run(cmd).returncode != 0 or not os.path.isfile(rj):
                        print(f"[fail] {rn}")
                        failures.append(rn)
                        continue
                else:
                    print(f"\n[{i}/{total}] === {rn} === (reusing existing {rj})", flush=True)
            else:
                cmd = build_cmd(args, v, s, rj, rn)
                print(f"\n[{i}/{total}] === {rn} ===\n  {' '.join(cmd)}", flush=True)
                if subprocess.run(cmd).returncode != 0 or not os.path.isfile(rj):
                    print(f"[fail] {rn}")
                    failures.append(rn)
                    continue
            with open(rj) as f:
                r = json.load(f)
            try:
                dice, split = pick_metric(args, r)
            except KeyError as e:
                print(f"[fail] {rn}: {e}")
                failures.append(rn)
                continue
            for les in args.lesions:
                val = dice.get(les, float("nan"))
                if val == val:                      # skip NaN (lesion absent from split)
                    results[v][les][s] = val
            present = [dice[k] for k in args.lesions if k in dice and dice[k] == dice[k]]
            if len(present) < len(args.lesions):
                print(f"[warn] {rn}: only {len(present)}/{len(args.lesions)} lesions scoreable; "
                      "mean is over the scoreable ones")
            means[v][s] = sum(present) / max(len(present), 1)

    # ---- report ------------------------------------------------------------- #
    split_name = "unknown"
    for v in variants:
        if means[v]:
            rj = os.path.join(args.out_dir, f"{v}_seed{sorted(means[v])[0]}.json")
            with open(rj) as f:
                split_name = pick_metric(args, json.load(f))[1]
            break

    L = [f"\n{'='*74}",
         f"ARCHITECTURE SWEEP on {split_name}  (seeds={args.seeds}, epochs={args.epochs},",
         f"  train={'+'.join(args.datasets)}, gcg={'off' if args.no_gcg else args.gcg_variant})",
         f"{'='*74}",
         f"{'variant':<12}" + "".join(f"{les:<14}" for les in args.lesions)
         + f"{'MEAN':<16}{'GMAC':>7}{'params':>9}"]
    for v in variants:
        agg = aggregate({les: list(results[v][les].values()) for les in args.lesions})
        row = f"{v:<12}"
        for les in args.lesions:
            m, s = agg[les]
            row += f"{f'{m:.3f}+/-{s:.3f}':<14}"
        mv = list(means[v].values())
        mm = statistics.mean(mv) if mv else float("nan")
        ms = statistics.pstdev(mv) if len(mv) > 1 else 0.0
        c = costs.get(v, {})
        row += (f"{f'{mm:.3f}+/-{ms:.3f}':<16}"
                f"{c.get('gmac', float('nan')):7.2f}{c.get('params_m', float('nan')):8.2f}M")
        L.append(row)

    L += ["-" * 74,
          f"PAIRED vs {BASELINE} (same seeds; mean Dice):"]
    paired = {}
    for v in variants:
        if v == BASELINE:
            continue
        ps = paired_stats(means[v], means[BASELINE])
        paired[v] = {**ps, "ci95": list(ps["ci95"])}
        lo, hi = ps["ci95"]
        if ps["n"] < 2:
            # paired_stats collapses the interval to the point estimate at n=1;
            # printing it as a CI would read as a precise result from one run.
            L.append(f"  {v:<11} delta={ps['mean_delta']:+.3f}  no CI (n={ps['n']}; "
                     f"needs >=2 seeds)  {ps['n_positive']}/{ps['n']} seeds+")
            continue
        verdict = "SIGNIFICANT" if ps["significant"] else "inconclusive (CI includes 0)"
        L.append(f"  {v:<11} delta={ps['mean_delta']:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
                 f"{ps['n_positive']}/{ps['n']} seeds+  {verdict}")
    if failures:
        L.append(f"\n(!) failed runs: {failures}")
    L.append("\nNB: a 'significant' Dice delta on IDRiD's 27-image test set is still a")
    L.append("    27-image result. Confirm anything you plan to report on FGADR.")

    out = "\n".join(L)
    print(out)

    # A summary of NaNs is worse than no summary: it survives on disk, reads like
    # a finished sweep with missing cells, and there is nothing in the file itself
    # to say every run crashed. Refuse to write one.
    if not any(means[v] for v in variants):
        print(f"\n[fatal] every run failed — refusing to write a summary of NaNs to "
              f"{args.out_dir}/. Fix the failure above and re-run; the first traceback "
              f"in this log is the real error.", file=sys.stderr)
        return 2

    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"variants": {v: VARIANTS[v] for v in variants},
                   "costs": costs, "split": split_name,
                   "per_lesion": {v: {k: dict(x) for k, x in results[v].items()} for v in results},
                   "means": means, "paired_vs_baseline": paired,
                   "seeds": args.seeds, "epochs": args.epochs,
                   "datasets": args.datasets, "failures": failures}, f, indent=2)
    print(f"\n[info] wrote {args.out_dir}/summary.md and summary.json")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
