#!/usr/bin/env python3
"""
precision_check.py
==================
**Does the sweep's arithmetic destroy the microaneurysm channel?**

Reduced precision does not fail loudly. It fails as one lesion class quietly
going to zero while the others look fine, which is indistinguishable from "that
backbone just can't learn MA" unless you go looking. It already happened once in
this repo: TF32 (10-bit mantissa) took mobilenetv4_m's MA Dice from 0.974 to
0.000 with HE and EX untouched, because MA is the sparsest class and therefore
carries the smallest gradients in the batch.

``--amp`` is fp16, whose mantissa is also 10 bits, and every recommended command
in ``run_arch_sweep.py`` and the README passes it. GradScaler defends against
gradient *underflow*, which is a different failure from mantissa truncation, so
AMP is not automatically safe just because TF32 was fixed — it is untested. This
script tests it, on three patches, before a 15-run sweep spends the GPU-hours.

What it runs
------------
Per encoder, the same fixed batch and seed under four arithmetic settings:

    fp32       tf32 off, amp off    <- reference; every Dice in this repo
    tf32       tf32 on,  amp off    <- the known failure, kept as a positive control
    amp        tf32 off, amp fp16   <- WHAT THE SWEEP ACTUALLY RUNS
    amp_tf32   tf32 on,  amp fp16   <- what the sweep ran before the TF32 fix

Read the **MA column against fp32**. A large drop in MA with HE/EX intact is the
signature; a uniform drop across lesions is ordinary noise. The positive control
matters: if the ``tf32`` row does *not* reproduce the known collapse, the harness
is not measuring what it claims (wrong GPU generation, TF32 unavailable) and the
``amp`` row proves nothing either.

Run
---
    # the check that gates the sweep (~10 min on one GPU)
    python precision_check.py

    # everything the sweep will train, all four settings
    python precision_check.py --encoders mobilenetv3 mobilenetv4_m efficientvit_b1

    # just the untested case, no positive control
    python precision_check.py --modes fp32 amp

Exit code is non-zero if any encoder shows a precision-attributable MA collapse.
TF32 only exists on Ampere+ (A100/RTX30xx and later); on older cards the tf32
rows are identical to fp32 by construction and the control is uninformative.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "overfit_test.py")

# label -> (allow_tf32, amp)
MODES = {
    "fp32":     (False, False),
    "tf32":     (True,  False),
    "amp":      (False, True),
    "amp_tf32": (True,  True),
}


def run_one(args, encoder, mode, out_json):
    tf32, amp = MODES[mode]
    cmd = [sys.executable, RUNNER,
           "--encoder", encoder,
           "--decoder", args.decoder,
           "--lateral-channels", str(args.lateral_channels),
           "--lesions", *args.lesions,
           "--patch-size", str(args.patch_size),
           "--batch-size", str(args.batch_size),
           "--steps", str(args.steps),
           "--lr", str(args.lr),
           "--seed", str(args.seed),
           "--json", out_json]
    if args.root:
        cmd += ["--root", args.root]
    if tf32:
        cmd += ["--allow-tf32"]
    if amp:
        cmd += ["--amp"]
    if args.no_gcg:
        cmd += ["--no-gcg"]
    print(f"\n{'='*74}\n[{encoder} / {mode}]  {' '.join(cmd)}\n{'='*74}", flush=True)
    rc = subprocess.run(cmd).returncode
    # A FAIL exit (rc=1) is a legitimate outcome here — that is the collapse we
    # are hunting. Only a missing JSON means the run itself broke.
    if not os.path.isfile(out_json):
        print(f"[fail] {encoder}/{mode} produced no JSON (rc={rc})")
        return None
    with open(out_json) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Precision matrix on the overfit test.")
    p.add_argument("--encoders", nargs="+", default=["mobilenetv3", "mobilenetv4_m"],
                   help="mobilenetv3 is the reference; mobilenetv4_m is the one TF32 broke.")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--root", default=None, help="IDRiD root; omit to fetch via kagglehub.")
    p.add_argument("--lesions", nargs="+", default=["MA", "HE", "EX", "SE"])
    p.add_argument("--decoder", default="dense", choices=["dense", "separable"])
    p.add_argument("--lateral-channels", type=int, default=-1)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--out-dir", default="exp_precision")
    p.add_argument("--drop-thresh", type=float, default=0.25,
                   help="Flag a lesion whose Dice falls this far below its fp32 value.")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    modes = [m for m in MODES if m in args.modes]        # keep canonical order
    results = {}
    for enc in args.encoders:
        results[enc] = {}
        for mode in modes:
            oj = os.path.join(args.out_dir, f"{enc}_{mode}.json")
            if args.skip_existing and os.path.isfile(oj):
                with open(oj) as f:
                    old_r = json.load(f)
                # Reuse only when the stored run matches the current config; a
                # same-named JSON from an older sweep (different steps/patch)
                # would silently pollute the matrix under the current header.
                expect = {"steps": args.steps, "encoder": enc,
                          "patch_size": args.patch_size, "batch_size": args.batch_size}
                mismatch = {k: (old_r.get(k), want) for k, want in expect.items()
                            if k in old_r and old_r.get(k) != want}
                if mismatch:
                    print(f"[skip-existing] {oj} config differs {mismatch}; re-running")
                    results[enc][mode] = run_one(args, enc, mode, oj)
                else:
                    print(f"[skip] reusing {oj}")
                    results[enc][mode] = old_r
            else:
                results[enc][mode] = run_one(args, enc, mode, oj)

    # ---- report ------------------------------------------------------------ #
    L = [f"\n{'='*78}",
         "PRECISION MATRIX — overfit test on a fixed IDRiD batch",
         f"  {args.steps} steps, patch={args.patch_size}px, batch={args.batch_size}, "
         f"seed={args.seed}, decoder={args.decoder}",
         "  Dice near 1.0 is the only correct answer here: the model is being asked to",
         "  memorise three patches it can see. Anything low is a defect, not a hard task.",
         f"{'='*78}"]

    suspicious = []
    for enc in args.encoders:
        ref = results[enc].get("fp32")
        L.append(f"\n{enc}")
        L.append(f"  {'mode':<10}" + "".join(f"{l:>9}" for l in args.lesions)
                 + f"{'mean':>9}{'skipped':>9}  {'vs fp32':<20}")
        for mode in modes:
            r = results[enc].get(mode)
            if r is None:
                L.append(f"  {mode:<10}{'[run failed]':>40}")
                continue
            row = f"  {mode:<10}"
            for l in args.lesions:
                # Unannotated lesions carry a meaningless Dice; do not read them.
                row += f"{r['dice'][l]:9.3f}" if r["present"][l] else f"{'  -':>9}"
            row += f"{r['mean_present_dice']:9.3f}{r.get('scaler_skipped_steps', 0):9d}  "

            note = "(reference)"
            if ref and mode != "fp32":
                drops = [(l, ref["dice"][l] - r["dice"][l]) for l in args.lesions
                         if r["present"][l] and ref["present"][l]
                         and ref["dice"][l] - r["dice"][l] > args.drop_thresh]
                if drops:
                    note = "DROP " + " ".join(f"{l}-{d:.2f}" for l, d in drops)
                    suspicious.append((enc, mode, drops))
                else:
                    note = "ok"
            row += note
            L.append(row)

    # The TF32 positive control is a property of the RUN, not of each encoder:
    # the known collapse is mobilenetv4_m-specific, and mobilenetv3 tolerating
    # TF32 is the documented, expected outcome. Warning per-encoder would flag
    # that healthy row as a failed control every time.
    if "tf32" in modes and "MA" in args.lesions:
        reproduced = [
            enc for enc in args.encoders
            if results[enc].get("fp32") and results[enc].get("tf32")
            and results[enc]["fp32"]["present"].get("MA")
            and results[enc]["fp32"]["dice"]["MA"] - results[enc]["tf32"]["dice"]["MA"]
            > args.drop_thresh
        ]
        L.append("")
        if reproduced:
            L.append(f"  [control] tf32 reproduced the MA collapse on {', '.join(reproduced)} —")
            L.append("            the harness detects precision damage, so the 'ok' rows above")
            L.append("            are meaningful negatives rather than a blind spot.")
        else:
            L.append("  [control] tf32 did NOT reproduce the known MA collapse on ANY encoder.")
            L.append("            Either this card predates TF32 (needs Ampere+) or the harness")
            L.append("            is not measuring what it claims. Every 'ok' above is then weak")
            L.append("            evidence — do not clear --amp on the strength of it.")

    L.append(f"\n{'-'*78}")
    if suspicious:
        L.append("VERDICT: precision-attributable Dice loss found:")
        for enc, mode, drops in suspicious:
            L.append(f"  {enc} / {mode}: " + ", ".join(f"{l} -{d:.3f}" for l, d in drops))
        L.append("")
        L.append("If the 'amp' row is implicated, do NOT run the sweep with --amp for that")
        L.append("encoder. Drop --amp (costs memory and speed, not correctness), or switch")
        L.append("autocast to bfloat16 — 8-bit mantissa but full fp32 exponent range, which")
        L.append("removes the underflow that GradScaler is patching over in the first place.")
    else:
        L.append("VERDICT: no precision-attributable Dice loss at the "
                 f"{args.drop_thresh} threshold.")
        L.append("--amp is safe for these encoders on this GPU. Note this is a 3-patch")
        L.append("memorisation test; it rules out gross numeric failure, not a small")
        L.append("generalisation cost.")

    out = "\n".join(L)
    print(out)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"encoders": args.encoders, "modes": modes,
                   "results": results, "suspicious": suspicious,
                   "drop_thresh": args.drop_thresh, "steps": args.steps,
                   "seed": args.seed}, f, indent=2)
    print(f"\n[info] wrote {args.out_dir}/summary.md and summary.json")
    return 1 if suspicious else 0


if __name__ == "__main__":
    raise SystemExit(main())
