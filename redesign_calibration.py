#!/usr/bin/env python3
"""
redesign_calibration.py
=======================
Why does self-calibration LOWER sensitivity at the shipped threshold, and which
label-free change fixes it? Runs on finished ``train_retinareach.py`` checkpoints
-- no retraining, minutes per checkpoint on a GPU::

    python redesign_calibration.py --ckpt ck_retinareach/seed0.pt ck_retinareach/seed3.pt \\
        --out exp_retinareach/redesign

The observation (lab box, seeds 0 and 3)
----------------------------------------
Full AdaBN raised handheld referable-DR AUROC (0.894 -> 0.935, 0.906 -> 0.923) but
sensitivity at the tabletop-fixed threshold FELL (0.638 -> 0.572, 0.645 -> 0.487);
the same on ODIR-5K and for edema, and the same with 16 or 4,132 calibration
captures -- a systematic shift, not estimation noise.

The hypothesis
--------------
AdaBN re-centres every feature on the new clinic's own average. mBRSET and ODIR
have ~2.5x BRSET's referable-DR prevalence, so part of the disease signal is
normalised away and every score drops: AdaBN cannot tell a camera shift from a
prevalence shift.

Experiments (every setting scored at the SAME shipped threshold)
----------------------------------------------------------------
A  Mechanism. Calibration pools of fixed size at controlled referable-DR
   prevalence (0, the source's rate, the camera's natural rate, 0.35) on every
   camera -- including the tabletop camera itself, where only prevalence
   changes. Labels build the pools and nothing else; the device never sees them.
   Sensitivity falling with prevalence on the tabletop too = prevalence, not the
   camera, drives the drop.
B1 Shallow-only recalibration: the first K BatchNorm layers (fractions of depth)
   recalibrated, the rest kept at the tabletop statistics. K = 0 is "tabletop
   statistics on the new camera", K = all is full AdaBN.
B2 The source prior, in-forward (``retinareach.calibrate_blended``):
   alpha in {0.25, 0.5, 0.75} (0 = AdaBN, 1 = tabletop statistics).
B3 Re-estimated threshold: the clinic's prevalence estimated from the model's own
   outputs on the unlabelled pool (EM, Saerens, Latinne & Decaestecker 2002, on
   outputs Platt-calibrated on tabletop validation), then the threshold that keeps
   the EXPECTED sensitivity at the target under those soft labels. The estimated
   prevalence is reported next to the true pool prevalence.

Outputs: ``<out>/redesign.csv`` (one row per checkpoint x experiment x camera x
setting x repeat x target) and a printed summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brset_dataset import load_any  # noqa: E402
from capability_gate import multitask_split  # noqa: E402
from fundus_utils import seed_worker  # noqa: E402
from metadata_model import _auc, load_table  # noqa: E402
from retinareach import (bn_layers, build_from_checkpoint, calibrate_blended,  # noqa: E402
                         set_bn_stats, stats_from_lists)
from train_mbrset import pick_device  # noqa: E402
from train_retinareach import MultiTargetDataset, parse_unfamiliar, predict  # noqa: E402


# --------------------------------------------------------------------------- #
# Label-free prevalence and threshold (B3)
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(scores: np.ndarray, y: np.ndarray):
    """1-D logistic map from model probability to calibrated probability, fit on
    labelled source validation outputs."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(scores)[:, None], y)
    return lambda s: lr.predict_proba(_logit(s)[:, None])[:, 1]


def em_prevalence(q: np.ndarray, prior_src: float, iters: int = 200, tol: float = 1e-7):
    """Saerens et al. (2002): EM re-estimate of the class prior on unlabelled
    data from posteriors ``q`` calibrated at prior ``prior_src``. Returns
    (estimated prior, adjusted posteriors)."""
    pi = prior_src
    for _ in range(iters):
        a = (pi / prior_src) * q
        b = ((1 - pi) / (1 - prior_src)) * (1 - q)
        post = a / np.clip(a + b, 1e-12, None)
        new = float(post.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi, post


def expected_sensitivity_threshold(scores: np.ndarray, soft: np.ndarray, target: float) -> float:
    """Highest threshold whose EXPECTED sensitivity under soft labels ``soft``
    (sum of soft positives at or above it / all soft positives) is >= target."""
    order = np.argsort(-scores)
    s, w = scores[order], soft[order]
    cum = np.cumsum(w) / max(w.sum(), 1e-12)
    k = int(np.searchsorted(cum, target))
    return float(s[min(k, len(s) - 1)])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def prevalence_subset(y: np.ndarray, prev: float, size: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """``size`` labelled indices with exactly round(prev * size) positives
    (shrinking ``size`` if the pool cannot supply them); None if impossible."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(prev * size))
    if n_pos > len(pos) or size - n_pos > len(neg):
        size = int(min(len(pos) / prev if prev > 0 else np.inf,
                       len(neg) / (1 - prev) if prev < 1 else np.inf, size))
        n_pos = int(round(prev * size))
    if size < 2:
        return None
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, size - n_pos, replace=False)])
    return rng.permutation(idx)


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), p[ok]
    if len(np.unique(y)) < 2:
        return {}
    flag = p >= thr
    lg = _logit(p)
    return {"auroc": _auc(y, p), "sensitivity": float(flag[y == 1].mean()),
            "specificity": float((~flag[y == 0]).mean()), "flagged": float(flag.mean()),
            "logit_neg": float(lg[y == 0].mean()), "logit_pos": float(lg[y == 1].mean()),
            "threshold": float(thr)}


def load_device(root: str, dataset: str, targets: Sequence[str], strat: str, split_seed: int,
                image_size: int) -> Dict[str, object]:
    df, _ = load_table(root, dataset)
    split = multitask_split(df, strat, seed=split_seed)
    img = load_any(root, dataset)["images_dir"]
    mk = lambda part: MultiTargetDataset(split[part], img, targets, "val", image_size)
    tr, va, te = mk("train"), mk("val"), mk("test")
    return {"pool": ConcatDataset([tr, va]),
            "pool_y": np.concatenate([tr.label_array(), va.label_array()]),
            "val": va, "test": te}


# --------------------------------------------------------------------------- #
# One checkpoint
# --------------------------------------------------------------------------- #
def run_checkpoint(path: str, a, dev) -> List[dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    net = build_from_checkpoint(ck).to(dev)
    targets = net.targets
    src_targets = list(ck["config"]["source_targets"])
    k_primary = targets.index(src_targets[0])
    S, T = args["source_name"], args["target_name"]
    B, size = int(args.get("calib_batch", 16)), int(ck["config"].get("image_size", 384))
    thr_cal, thr_tr = ck["thresholds"]["calibrated"], ck["thresholds"]["trained"]
    tab = stats_from_lists(ck["profiles"][S]["bn_stats"])          # tabletop-calibrated statistics
    seed = int(args.get("seed", 0))
    rng = np.random.default_rng(seed)
    specs = [(S, a.source_root or args["source_root"], args["source_dataset"]),
             (T, a.target_root or args["target_root"], args["target_dataset"])]
    if not a.no_unfamiliar:
        specs += [(n, r, d) for r, d, n in (parse_unfamiliar(u) for u in args.get("unfamiliar") or [])]
    devs = {n: load_device(r, d, targets, src_targets[0], int(args["split_seed"]), size)
            for n, r, d in specs}
    loaders = {n: DataLoader(v["test"], batch_size=64, num_workers=a.num_workers,
                             worker_init_fn=seed_worker) for n, v in devs.items()}
    print(f"\n=== {path} (seed {seed}): cameras {list(devs)}; threshold dr {thr_cal.get(src_targets[0]):.3f} ===")

    def calib_loader(ds, s):
        g = torch.Generator(); g.manual_seed(s)
        return DataLoader(ds, batch_size=B, shuffle=True, generator=g, num_workers=a.num_workers,
                          worker_init_fn=seed_worker)

    def score(model, dname, thr: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        q = predict(model, loaders[dname], dev)
        y = devs[dname]["test"].label_array()
        return {t: metrics(y[:, targets.index(t)], q[:, targets.index(t)], thr.get(t, np.nan))
                for t in src_targets if thr.get(t) is not None}

    rows: List[dict] = []

    def add(exp, dname, setting, res, **extra):
        for t, m in res.items():
            if m:
                rows.append({"ckpt": os.path.basename(path), "seed": seed, "experiment": exp,
                             "device": dname, "setting": setting, "target": t, **m, **extra})

    tab_model = build_from_checkpoint(ck).to(dev)
    set_bn_stats(tab_model, tab)
    names = [n for n, _ in bn_layers(net)]
    y_src_pool = devs[S]["pool_y"][:, k_primary]
    src_prev = float(np.nanmean(y_src_pool))

    # baselines on every camera
    for dname in devs:
        add("baseline", dname, "as trained", score(net, dname, thr_tr))
        add("baseline", dname, "tabletop statistics", score(tab_model, dname, thr_cal))
        full, _ = calibrate_blended(tab_model, calib_loader(devs[dname]["pool"], seed), dev)
        add("baseline", dname, "AdaBN (full pool)", score(full, dname, thr_cal),
            pool_prev=float(np.nanmean(devs[dname]["pool_y"][:, k_primary])))
        del full

    # A: prevalence-controlled pools (labels build the pools only)
    for dname, d in devs.items():
        y = d["pool_y"][:, k_primary]
        nat = float(np.nanmean(y))
        for prev in sorted({0.0, round(src_prev, 3), round(nat, 3), 0.35}):
            for r in range(a.repeats):
                idx = prevalence_subset(y, prev, a.pool_size, rng)
                if idx is None:
                    continue
                cal, info = calibrate_blended(tab_model, calib_loader(Subset(d["pool"], idx), seed + r), dev)
                add("A_prevalence", dname, f"pool prevalence {prev:.3f}", score(cal, dname, thr_cal),
                    pool_prev=prev, repeat=r, n_images=info.n_images)
                del cal

    others = [n for n in devs if n != S]
    # B1: shallow-only; B2: in-forward prior -- on the natural full pool
    for dname in others:
        pool = devs[dname]["pool"]
        for frac in a.depth_fractions:
            k = int(round(frac * len(names)))
            cal, _ = calibrate_blended(tab_model, calib_loader(pool, seed), dev, adapt=names[:k])
            add("B1_shallow", dname, f"first {k}/{len(names)} BN layers", score(cal, dname, thr_cal),
                depth_fraction=frac, layers=k)
            del cal
        for alpha in a.alphas:
            cal, _ = calibrate_blended(tab_model, calib_loader(pool, seed), dev, alpha=alpha)
            add("B2_prior", dname, f"alpha {alpha:.2f}", score(cal, dname, thr_cal), alpha=alpha)
            del cal

    # B3: EM prevalence + expected-sensitivity threshold (label-free)
    va = devs[S]["val"]
    q_val = predict(tab_model, DataLoader(va, batch_size=64, num_workers=a.num_workers), dev)
    y_val = va.label_array()
    for dname in others:
        pool = devs[dname]["pool"]
        full, _ = calibrate_blended(tab_model, calib_loader(pool, seed), dev)
        for base_name, model in (("tabletop statistics", tab_model), ("AdaBN (full pool)", full)):
            q_pool = predict(model, DataLoader(pool, batch_size=64, num_workers=a.num_workers), dev)
            thr_new = {}
            est = {}
            for t in src_targets:
                k = targets.index(t)
                ok = ~np.isnan(y_val[:, k])
                if len(np.unique(y_val[ok, k])) < 2:
                    continue
                cal_fn = platt(q_val[ok, k], y_val[ok, k].astype(int))
                pi_s = float(y_val[ok, k].mean())
                pi_hat, post = em_prevalence(cal_fn(q_pool[:, k]), pi_s)
                thr_new[t] = expected_sensitivity_threshold(q_pool[:, k], post, a.target_sens)
                est[t] = pi_hat
            res = score(model, dname, thr_new)
            for t, m in res.items():
                if m:
                    true_prev = float(np.nanmean(devs[dname]["pool_y"][:, targets.index(t)]))
                    rows.append({"ckpt": os.path.basename(path), "seed": seed,
                                 "experiment": "B3_threshold", "device": dname,
                                 "setting": f"{base_name} + EM threshold", "target": t, **m,
                                 "est_prev": est.get(t), "pool_prev": true_prev})
        del full
    return rows


def summarise(df: pd.DataFrame, primary: str) -> str:
    d = df[df["target"] == primary]
    out = []
    for exp in ("baseline", "A_prevalence", "B1_shallow", "B2_prior", "B3_threshold"):
        e = d[d["experiment"] == exp]
        if e.empty:
            continue
        cols = ["auroc", "sensitivity", "specificity", "flagged", "logit_neg"]
        cols += [c for c in ("pool_prev", "est_prev") if c in e and e[c].notna().any()]
        g = e.groupby(["device", "setting"], sort=False)[cols].mean()
        out.append(f"\n--- {exp} ({primary}; mean over checkpoints and repeats) ---\n"
                   + g.to_string(float_format=lambda v: f"{v:.3f}"))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Why self-calibration moves the threshold, and label-free fixes.")
    p.add_argument("--ckpt", nargs="+", required=True, help="train_retinareach.py checkpoints")
    p.add_argument("--out", default="exp_retinareach/redesign")
    p.add_argument("--source-root", default=None, help="override the checkpoint's recorded roots")
    p.add_argument("--target-root", default=None)
    p.add_argument("--no-unfamiliar", action="store_true")
    p.add_argument("--pool-size", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--depth-fractions", type=float, nargs="+",
                   default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--target-sens", type=float, default=None,
                   help="default: the checkpoint's own --target-sens")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    a = p.parse_args(argv)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for path in a.ckpt:
        if a.target_sens is None:
            a.target_sens = float(torch.load(path, map_location="cpu",
                                             weights_only=False)["args"].get("target_sens", 0.85))
        rows += run_checkpoint(path, a, dev)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "redesign.csv"), index=False)
    df = pd.DataFrame(rows)
    primary = df["target"].iloc[0] if len(df) else "dr_referable"
    txt = summarise(df, "dr_referable" if (df["target"] == "dr_referable").any() else primary)
    print(txt)
    with open(os.path.join(a.out, "redesign_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/redesign.csv and redesign_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
