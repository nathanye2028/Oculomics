#!/usr/bin/env python3
"""
operating_point_sweep.py
========================
Does a tabletop-calibrated operating point transfer to smartphone images — and
does label-free device adaptation (AdaBN) restore it? Asked of EVERY checkpoint
in a sweep, paired by seed, so the answer is a distribution rather than one
anecdote.

Why this exists
---------------
``evaluate_deploy.py`` on one checkpoint (kd_seed1) showed AUROC dropping only
0.09 from BRSET to mBRSET while **sensitivity at the BRSET-calibrated threshold
halved (0.88 -> 0.54)**: the model's probabilities run systematically lower on
phone captures. A screening tool ships a threshold, not an AUROC, so this is the
deployment-relevant failure — and it is exactly the kind of statistic shift
AdaBN corrects. This script measures both halves of that claim across all
``ctrl_seed*.pt`` / ``kd_seed*.pt`` checkpoints:

  1. threshold chosen on the checkpoint's OWN in-domain validation split
     (rebuilt from the recorded seed, as train_mbrset.py built it) for
     sensitivity >= --target-sens, using the zero-shot weights;
  2. external set scored at that threshold, zero-shot                (ships today);
  3. external set scored at that SAME threshold after AdaBN on the external
     images (labels never used)                                       (ships with device calibration);
  4. for reference only, the *oracle* threshold that would hit the target
     sensitivity on the external set itself — this uses external labels and
     is never a deployment number; it just says how far off the source
     threshold is.

Nothing is retrained. ~1-2 min per checkpoint on a GPU (three passes over the
external set + one over val).

    python operating_point_sweep.py --ckpt-dir ck_kd_v4_384_v2 \\
        --root <BRSET> --external-root <mBRSET> --out exp_kd_v4_384_v2/operating_point.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brset_dataset import load_any                                        # noqa: E402
from dataset import MBRSETDataset, stratified_split                       # noqa: E402
from evaluate_deploy import binary_report, operating_point, torch_scores  # noqa: E402
from fundus_utils import seed_worker                                      # noqa: E402
from model import MBRSETClassifier                                        # noqa: E402
from run_experiment import paired_stats                                   # noqa: E402
from train_mbrset import adapt_bn, backbone_kwargs_for, pick_device       # noqa: E402

_NAME = re.compile(r"^(?P<cond>.+?)_seed(?P<seed>\d+)\.pt$")


def build_from_checkpoint(ck: dict, num_classes: int) -> MBRSETClassifier:
    a = ck.get("args", {}) or {}
    backbone = a.get("backbone", "mobilenetv3_small")
    use_gcg = ck.get("use_gcg")
    if use_gcg is None:                      # pre-2026-09 checkpoints stored only the flag
        use_gcg = (not a.get("no_gcg", False)) and not backbone.startswith("timm:")
    m = MBRSETClassifier(num_classes=num_classes, pretrained=False, use_gcg=use_gcg,
                         gcg_variant=a.get("gcg_variant", "baseline"), backbone=backbone,
                         backbone_kwargs=backbone_kwargs_for(backbone, int(a.get("image_size", 224))))
    m.load_state_dict(ck["model"])
    return m.eval()


def auroc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def score_one(path: str, args, device) -> dict:
    ck = torch.load(path, map_location="cpu")
    a = ck.get("args", {}) or {}
    task = a.get("task", "dr_referable")
    seed = int(a.get("seed", 0))
    img_size = int(a.get("image_size", 224))
    train_dataset = a.get("dataset", "mbrset")

    # In-domain val split exactly as the trainer built it (same seed, same grouping).
    src = load_any(args.root, train_dataset, image_ext=args.image_ext)
    splits = stratified_split(src["df"], task=task, val_frac=0.10, test_frac=0.20,
                              group_col="patient", seed=seed)
    val_ds = MBRSETDataset(csv=splits["val"], images_dir=src["images_dir"], task=task, split="val",
                           image_size=img_size, drop_missing_files=True, fov_crop=True)
    ext = load_any(args.external_root, args.external_dataset, image_ext=args.image_ext)
    ext_ds = MBRSETDataset(csv=ext["df"], images_dir=ext["images_dir"], task=task, split="val",
                           image_size=img_size, drop_missing_files=True, fov_crop=True)
    if val_ds.num_classes != 2 or ext_ds.num_classes != 2:
        raise SystemExit(f"[fatal] operating points need a binary task; {task} is not.")

    dl = lambda ds, **kw: DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, pin_memory=(device.type == "cuda"), **kw)
    val_loader, ext_loader = dl(val_ds, shuffle=False), dl(ext_ds, shuffle=False)

    model = build_from_checkpoint(ck, 2).to(device)

    # 1. source-calibrated threshold (zero-shot weights, in-domain val)
    vs, vy = torch_scores(model, val_loader, device, tta=args.tta)
    thr, vsens, vspec = operating_point(vs, vy, target_sens=args.target_sens)

    # 2. external, zero-shot, at the source threshold
    es, ey = torch_scores(model, ext_loader, device, tta=args.tta)
    zero = {"auroc": auroc(es, ey), **binary_report(es, ey, thr)}

    # 3. external after AdaBN on the external IMAGES, same source threshold
    g = torch.Generator(); g.manual_seed(seed)
    bn_loader = dl(ext_ds, shuffle=True, generator=g)
    adapted, n_bn = adapt_bn(model, bn_loader, device, max_batches=args.bn_adapt_batches)
    if n_bn == 0:
        adapt = None
    else:
        as_, ay = torch_scores(adapted, ext_loader, device, tta=args.tta)
        adapt = {"auroc": auroc(as_, ay), **binary_report(as_, ay, thr)}

    # 4. oracle (uses external labels) — reference only
    thr_or, _, _ = operating_point(es, ey, target_sens=args.target_sens)
    oracle_zero = binary_report(es, ey, thr_or)
    oracle_adapt = None
    if adapt is not None:
        thr_or_a, _, _ = operating_point(as_, ay, target_sens=args.target_sens)
        oracle_adapt = {"threshold": thr_or_a, **binary_report(as_, ay, thr_or_a)}

    return {"ckpt": path, "task": task, "seed": seed, "image_size": img_size,
            "backbone": a.get("backbone", "mobilenetv3_small"), "train_dataset": train_dataset,
            "external_dataset": args.external_dataset, "n_val": int(len(vy)), "n_external": int(len(ey)),
            "target_sens": args.target_sens,
            "source_threshold": float(thr), "val_sens": vsens, "val_spec": vspec,
            "zero_shot": zero, "bn_adapted": adapt, "bn_layers": n_bn,
            "oracle_zero_shot": {"threshold": float(thr_or), **oracle_zero},
            "oracle_bn_adapted": oracle_adapt}


def _agg(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def summarise(rows, args) -> str:
    by = {}
    for r in rows:
        m = _NAME.match(os.path.basename(r["ckpt"]))
        if not m:
            continue
        by.setdefault(m["cond"], {})[int(m["seed"])] = r
    L = [f"\n{'='*84}",
         f"OPERATING POINT TRANSFER  (threshold: in-domain val, sensitivity >= {args.target_sens}; "
         f"external = {args.external_dataset})",
         f"{'='*84}",
         f"{'condition':<8}{'n':>3}{'thr':>8}{'  zero-shot sens/spec/AUROC':<30}{'  AdaBN sens/spec/AUROC':<28}{'  oracle thr (ref.)':<20}"]
    per_cond_delta = {}
    for cond in sorted(by):
        rs = by[cond]; seeds = sorted(rs)
        zs = [rs[s]["zero_shot"]["sensitivity"] for s in seeds]
        zp = [rs[s]["zero_shot"]["specificity"] for s in seeds]
        za = [rs[s]["zero_shot"]["auroc"] for s in seeds]
        has_adapt = all(rs[s]["bn_adapted"] for s in seeds)
        thr = [rs[s]["source_threshold"] for s in seeds]
        othr = [rs[s]["oracle_zero_shot"]["threshold"] for s in seeds]
        zs_m, zs_s = _agg(zs); zp_m, _ = _agg(zp); za_m, _ = _agg(za)
        line = (f"{cond:<8}{len(seeds):>3}{_agg(thr)[0]:>8.3f}"
                f"  {zs_m:.3f}±{zs_s:.3f} / {zp_m:.3f} / {za_m:.3f}   ")
        if has_adapt:
            a_s = [rs[s]["bn_adapted"]["sensitivity"] for s in seeds]
            a_p = [rs[s]["bn_adapted"]["specificity"] for s in seeds]
            a_a = [rs[s]["bn_adapted"]["auroc"] for s in seeds]
            as_m, as_s = _agg(a_s); ap_m, _ = _agg(a_p); aa_m, _ = _agg(a_a)
            line += f"  {as_m:.3f}±{as_s:.3f} / {ap_m:.3f} / {aa_m:.3f}   "
            per_cond_delta[cond] = paired_stats(dict(zip(seeds, a_s)), dict(zip(seeds, zs)))
        else:
            line += f"  {'(no BN layers)':<28}"
        line += f"  {_agg(othr)[0]:.3f}"
        L.append(line)

    L += ["", "-" * 84, "PAIRED within run: AdaBN minus zero-shot SENSITIVITY at the source threshold"]
    for cond, st in per_cond_delta.items():
        lo, hi = st["ci95"]
        L.append(f"  {cond:<8} delta={st['mean_delta']:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
                 f"{st['n_positive']}/{st['n']} seeds+  {'SIGNIFICANT' if st['significant'] else 'n.s.'}")

    if args.treatment in by and args.control in by:
        seeds = sorted(set(by[args.treatment]) & set(by[args.control]))
        L += ["", "-" * 84, f"PAIRED {args.treatment} - {args.control} (same seed)"]
        for label, key, block in (("zero-shot sensitivity", "sensitivity", "zero_shot"),
                                  ("AdaBN sensitivity", "sensitivity", "bn_adapted"),
                                  ("AdaBN AUROC", "auroc", "bn_adapted")):
            try:
                t = {s: by[args.treatment][s][block][key] for s in seeds}
                c = {s: by[args.control][s][block][key] for s in seeds}
            except (KeyError, TypeError):
                continue
            st = paired_stats(t, c); lo, hi = st["ci95"]
            L.append(f"  {label:<22} delta={st['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                     f"{st['n_positive']}/{st['n']} seeds+  {'SIGNIFICANT' if st['significant'] else 'n.s.'}")

    L += ["", "NB: 'oracle thr' uses EXTERNAL labels and is a reference for how far the source",
          "    threshold is off — never a deployment number. AdaBN uses external IMAGES only."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="Operating-point transfer across a sweep's checkpoints.")
    p.add_argument("--ckpt-dir", required=True, help="Dir holding <cond>_seed<n>.pt files.")
    p.add_argument("--root", required=True, help="Root of the dataset the checkpoints were TRAINED on.")
    p.add_argument("--external-root", required=True)
    p.add_argument("--external-dataset", default="mbrset", choices=["mbrset", "brset"])
    p.add_argument("--conditions", nargs="+", default=["ctrl", "kd"],
                   help="Checkpoint name prefixes to score (teachers have no BN; skipped by default).")
    p.add_argument("--treatment", default="kd")
    p.add_argument("--control", default="ctrl")
    p.add_argument("--target-sens", type=float, default=0.90)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--bn-adapt-batches", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--out", required=True, help="Results JSON (summary .md written alongside).")
    args = p.parse_args()

    device = pick_device()
    paths = []
    for pth in sorted(glob.glob(os.path.join(args.ckpt_dir, "*_seed*.pt"))):
        m = _NAME.match(os.path.basename(pth))
        if m and m["cond"] in args.conditions:
            paths.append(pth)
    if not paths:
        raise SystemExit(f"[fatal] no {args.conditions} checkpoints under {args.ckpt_dir}")
    print(f"[info] device={device}  checkpoints={len(paths)}  external={args.external_dataset} "
          f"@ {args.external_root}")

    rows = []
    for i, pth in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(pth)} ...", flush=True)
        r = score_one(pth, args, device)
        z, ad = r["zero_shot"], r["bn_adapted"]
        print(f"      thr={r['source_threshold']:.3f}  zero-shot sens={z['sensitivity']:.3f} "
              f"spec={z['specificity']:.3f} AUROC={z['auroc']:.4f}"
              + (f"  | AdaBN sens={ad['sensitivity']:.3f} spec={ad['specificity']:.3f} "
                 f"AUROC={ad['auroc']:.4f}" if ad else "  | (no BN layers)"), flush=True)
        rows.append(r)

    report = summarise(rows, args)
    print(report)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "runs": rows}, f, indent=2)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("```\n" + report + "\n```\n")
    print(f"\n[info] wrote {args.out} and {os.path.splitext(args.out)[0]}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
