#!/usr/bin/env python3
"""
probe_folds.py
==============
Cross-fitted systemic probes: every handheld patient tested once.

The capability table gates each systemic probe on the handheld TEST split, a
fifth of the cohort, so four targets are UNDERPOWERED (7-16 positive test
patients) and the rest are judged on 40-180. The trunk is frozen and was trained
on the tabletop set only, so a probe can be fit K-fold out of fold on ALL
handheld rows without any handheld label reaching the trunk: K patient-grouped
folds, each probe fit on K-1 of them (its C and its sensitivity threshold chosen
inside those K-1 by inner patient-grouped folds) and scored on the K-th. The
metadata comparator is cross-fit on the same folds, so the paired image-minus-
metadata delta keeps the same rows. On mBRSET that gives nephropathy 46,
neuropathy 55, myocardial infarction 98 and smoking 81 positive patients instead
of 7-16.

Inside a training run: ``train_retinareach.py --probe-folds 5`` (rows with
protocol ``calibrated_oof5`` in capability.csv). On a finished checkpoint, no
retraining, minutes on a GPU::

    python probe_folds.py --ckpt ck_rr_canon/seed0.pt --target-root <mBRSET> --folds 5

writes ``<out>/capability_probes_oof.csv`` (the gate rows),
``predictions_<device>_probes_oof<K>.csv`` (file, patient, fold, score and
threshold per probe) and ``probe_folds.json`` next to the run's results
(``--out``, default the checkpoint's recorded ``--out``); ``summarize_retinareach.py``
picks the rows up. The threshold a row is gated at is the median of the folds'
inner thresholds; the gate's sensitivity is secondary for these targets (the
image-minus-metadata test comes first).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capability_gate import (full_chart_auroc, gate_cell, metadata_comparator,  # noqa: E402
                             quality_only_auroc, quality_strata)
from metadata_model import _auc, threshold_at_sensitivity  # noqa: E402

DEFAULT_CS = (1e-3, 1e-2, 1e-1, 1.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _fit(Z: np.ndarray, y: np.ndarray, C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed).fit(Z, y)


def cross_fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5,
                    Cs: Sequence[float] = DEFAULT_CS, seed: int = 0, n_shuffle: int = 0,
                    target_sens: float = 0.85, inner_folds: int = 3, min_pos: int = 10) -> Dict[str, object]:
    """One probe target, K-fold out of fold over every labelled row.

    ``X`` [N, D] embeddings, ``y`` [N] labels (NaN = unlabelled), ``groups`` [N]
    patient ids. Per outer fold: standardise on the training folds, pick C by
    inner patient-grouped CV AUROC, fix the threshold at ``target_sens`` on the
    inner out-of-fold probabilities of that C, refit on the training folds, score
    the held-out fold. ``n_shuffle`` permuted-label refits per fold (the chosen C)
    give the negative control. Returns full-length ``scores`` / ``fold`` arrays
    (NaN / -1 off the labelled rows), the per-fold C and thresholds, their median
    ``threshold``, and ``shuffle`` (mean AUROC, SE, count).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y, dtype=float)
    n_all = len(y)
    out: Dict[str, object] = {"fitted": False, "reason": None, "folds": int(folds),
                              "fold": np.full(n_all, -1), "scores": np.full(n_all, np.nan),
                              "C": [], "thresholds": [], "threshold": float("nan"),
                              "shuffle": {"auroc": float("nan"), "se": float("nan"), "n": 0},
                              "n": int((~np.isnan(y)).sum()), "pos": int(np.nansum(y))}
    ok = ~np.isnan(y)
    yl, Xl, gl = y[ok].astype(int), np.asarray(X)[ok], np.asarray(groups)[ok]
    if yl.sum() < min_pos or len(np.unique(yl)) < 2:
        out["reason"] = f"{int(yl.sum())} positive images < {min_pos}"
        return out
    if len(np.unique(gl)) < folds:
        out["reason"] = f"{len(np.unique(gl))} patients cannot fill {folds} folds"
        return out
    fold = np.full(len(yl), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
                                .split(Xl, yl, gl)):
        fold[te] = k
    oof, thr_k, c_k, sh = np.full(len(yl), np.nan), [], [], []
    for k in range(folds):
        tr, te = fold != k, fold == k
        if len(np.unique(yl[tr])) < 2 or not te.any():
            continue
        mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6
        Ztr, Zte, ytr, gtr = (Xl[tr] - mu) / sd, (Xl[te] - mu) / sd, yl[tr], gl[tr]
        n_in = int(min(inner_folds, len(np.unique(gtr[ytr == 1])), len(np.unique(gtr[ytr == 0]))))
        best = None
        for C in Cs:
            inner = np.full(len(ytr), np.nan)
            if n_in >= 2:
                for itr, ite in StratifiedGroupKFold(n_in, shuffle=True, random_state=seed + 1).split(Ztr, ytr, gtr):
                    if len(np.unique(ytr[itr])) == 2:
                        inner[ite] = _fit(Ztr[itr], ytr[itr], C, seed).decision_function(Ztr[ite])
            m = ~np.isnan(inner)
            auc = _auc(ytr[m], inner[m]) if m.any() and len(np.unique(ytr[m])) == 2 else float("nan")
            if best is None or (auc == auc and (best[0] != best[0] or auc > best[0])):
                best = (auc, C, inner)
        _, C, inner = best
        m = ~np.isnan(inner)
        thr_k.append(threshold_at_sensitivity(ytr[m], _sigmoid(inner[m]), target_sens) if m.any()
                     else float("nan"))
        c_k.append(float(C))
        clf = _fit(Ztr, ytr, C, seed)
        oof[te] = _sigmoid(clf.decision_function(Zte))
        if n_shuffle > 0 and len(np.unique(yl[te])) == 2:
            rng = np.random.default_rng(seed + 100 + k)
            for _ in range(n_shuffle):
                sh.append(_auc(yl[te], _fit(Ztr, rng.permutation(ytr), C, seed).decision_function(Zte)))
    out["fold"][ok], out["scores"][ok] = fold, oof
    out.update(fitted=bool(np.isfinite(oof).any()), C=c_k, thresholds=thr_k,
               threshold=float(np.nanmedian(thr_k)) if thr_k and np.isfinite(thr_k).any() else float("nan"))
    if sh:
        out["shuffle"] = {"auroc": float(np.mean(sh)),
                          "se": float(np.std(sh, ddof=1) / np.sqrt(len(sh))) if len(sh) > 1 else float("nan"),
                          "n": len(sh)}
    return out


def oof_comparator(df: pd.DataFrame, fold: np.ndarray, task: str, feature_sets: Sequence[str],
                   seed: int = 0, cv_repeats: int = 5) -> Dict[str, object]:
    """``capability_gate.metadata_comparator`` cross-fit on ``fold`` (-1 = rows
    without this target's label, used for fitting only): scores aligned to
    ``df``'s rows, the chosen feature set / model (the most frequent across folds)
    and the mean CV SD, in the comparator's own layout."""
    scores = np.full(len(df), np.nan)
    picks, sds, means, feats = [], [], [], []
    for k in sorted(set(fold[fold >= 0].tolist())):
        comp = metadata_comparator(df[fold != k].reset_index(drop=True),
                                   df[fold == k].reset_index(drop=True), task, feature_sets,
                                   seed=seed, cv_repeats=cv_repeats)
        if comp["reason"] is not None:
            return {"scores": scores, "feature_set": None, "model": None, "cv_mean": float("nan"),
                    "cv_sd": float("nan"), "features": [], "reason": comp["reason"]}
        scores[fold == k] = comp["scores"]
        picks.append((comp["feature_set"], comp["model"]))
        sds.append(comp["cv_sd"]); means.append(comp["cv_mean"]); feats = comp["features"]
    if not picks:
        return {"scores": scores, "feature_set": None, "model": None, "cv_mean": float("nan"),
                "cv_sd": float("nan"), "features": [], "reason": "no fold to cross-fit on"}
    fs, mk = max(set(picks), key=picks.count)
    return {"scores": scores, "feature_set": fs, "model": mk, "cv_mean": float(np.nanmean(means)),
            "cv_sd": float(np.nanmean(sds)), "features": feats, "reason": None}


def _mean_over_folds(fn, df: pd.DataFrame, fold: np.ndarray, task: str, seed: int) -> float:
    vals = [fn(df[fold != k].reset_index(drop=True), df[fold == k].reset_index(drop=True), task, seed)
            for k in sorted(set(fold[fold >= 0].tolist()))]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def gate_probe_folds(df: pd.DataFrame, Z: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                     probe_targets: Sequence[str], col_of: Dict[str, int], device: str, a,
                     audits: Dict[str, dict]) -> Tuple[List[dict], List[dict], Optional[pd.DataFrame], Dict[str, dict]]:
    """Every probe target cross-fit and gated on all of ``df``'s rows. ``Z`` [N, D]
    calibrated embeddings, ``Y`` [N, K] labels (column ``col_of[target]``),
    ``groups`` patient ids; ``a`` carries the trainer's gate settings
    (``probe_folds``, ``probe_inner_folds``, ``probe_C``, ``seed``, ``n_shuffle``,
    ``min_probe_pos``, ``target_sens``, ``sens_tolerance``, ``gate_margin``,
    ``min_test_pos_patients``, ``n_boot``, ``gate_feature_sets``, ``comparator_seed``,
    ``cv_repeats``). Returns (capability rows, quality-strata rows, the per-row
    predictions table, per-target info)."""
    K = int(a.probe_folds)
    protocol = f"calibrated_oof{K}"
    rows, strata, info = [], [], {}
    pred = pd.DataFrame({"file": df["file"].to_numpy(), "patient": np.asarray(groups)})
    for t in probe_targets:
        au = audits.get(t, {"ok": False, "reason": "no audit"})
        if not au.get("ok"):
            continue
        res = cross_fit_probe(Z, Y[:, col_of[t]], groups, folds=K, Cs=a.probe_C, seed=a.seed,
                              n_shuffle=a.n_shuffle, target_sens=a.target_sens,
                              inner_folds=a.probe_inner_folds, min_pos=a.min_probe_pos)
        info[t] = {k: v for k, v in res.items() if k not in ("fold", "scores")}
        if not res["fitted"]:
            print(f"  {t:<22} not cross-fit: {res['reason']}")
            continue
        fold, scores = res["fold"], res["scores"]
        comp = oof_comparator(df, fold, t, a.gate_feature_sets, seed=a.comparator_seed,
                              cv_repeats=a.cv_repeats)
        full = _mean_over_folds(full_chart_auroc, df, fold, t, a.comparator_seed)
        qonly = _mean_over_folds(quality_only_auroc, df, fold, t, a.comparator_seed)
        row = gate_cell(t, device, protocol, df, scores, res["threshold"], comp,
                        target_sens=a.target_sens, sens_tolerance=a.sens_tolerance,
                        gate_margin=a.gate_margin, min_pos_patients=a.min_test_pos_patients,
                        n_boot=a.n_boot, seed=a.seed, audit=au, full_chart=full, quality_only=qonly,
                        shuffle=res["shuffle"] if res["shuffle"]["n"] else None)
        row.update(design=f"oof{K}", folds=K, probe_C=json.dumps(res["C"]),
                   fold_thresholds=json.dumps([float(v) for v in res["thresholds"]]))
        rows.append(row)
        strata += [dict(r, device=device, protocol=protocol)
                   for r in quality_strata(df, t, scores, res["threshold"])]
        pred[f"fold_{t}"], pred[f"p_{t}"], pred[f"threshold_{t}"] = fold, scores, res["threshold"]
        print(f"  {t:<22} {row['status']:<18} AUROC {row.get('auroc', float('nan')):.3f} vs metadata "
              f"{row.get('meta_auroc', float('nan')):.3f}  delta {row.get('delta', float('nan')):+.3f} "
              f"[{row.get('delta_lo', float('nan')):+.3f}, {row.get('delta_hi', float('nan')):+.3f}]  "
              f"positive patients {row['pos_patients']}  (C {res['C']})")
    return rows, strata, (pred if len(pred.columns) > 2 else None), info


def settings_from_checkpoint(ck: dict, **overrides) -> SimpleNamespace:
    """The trainer's gate settings as recorded in the checkpoint, overridable."""
    args = ck["args"]
    s = SimpleNamespace(
        probe_folds=5, probe_inner_folds=3, probe_C=list(args.get("probe_C", DEFAULT_CS)),
        seed=int(args.get("seed", 0)), n_shuffle=int(args.get("n_shuffle", 10)),
        min_probe_pos=int(args.get("min_probe_pos", 10)), target_sens=float(args.get("target_sens", 0.85)),
        sens_tolerance=float(args.get("sens_tolerance", 0.10)), gate_margin=float(args.get("gate_margin", 0.02)),
        min_test_pos_patients=int(args.get("min_test_pos_patients", 20)), n_boot=int(args.get("n_boot", 1000)),
        gate_feature_sets=list(args.get("gate_feature_sets", ["age_sex", "clinical"])),
        comparator_seed=int(args.get("comparator_seed", 0)), cv_repeats=int(args.get("cv_repeats", 5)))
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch
    from torch.utils.data import DataLoader
    from brset_dataset import load_any
    from capability_gate import encoding_audit
    from fundus_utils import seed_worker
    from metadata_model import load_table
    from retinareach import build_from_checkpoint
    from train_mbrset import pick_device
    from train_retinareach import MultiTargetDataset, predict
    p = argparse.ArgumentParser(description="Cross-fitted systemic probes on a finished RetinaReach checkpoint.")
    p.add_argument("--ckpt", required=True, help="train_retinareach.py checkpoint")
    p.add_argument("--target-root", default=None, help="handheld dataset root (default: the checkpoint's)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=None)
    p.add_argument("--n-shuffle", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--out", default=None, help="run directory to write into (default: the checkpoint's --out)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    a = p.parse_args(argv)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    T, root, ds_name = args["target_name"], a.target_root or args["target_root"], args["target_dataset"]
    out = a.out or args["out"]
    os.makedirs(out, exist_ok=True)
    s = settings_from_checkpoint(ck, probe_folds=a.folds, probe_inner_folds=a.inner_folds,
                                 n_shuffle=a.n_shuffle, n_boot=a.n_boot)
    net = build_from_checkpoint(ck, stats=T).to(dev)        # the handheld profile's statistics
    targets, probes = net.targets, list(net.probe_targets)
    df, csv = load_table(root, ds_name)
    audits = {t: encoding_audit(df, t) for t in probes}
    ds = MultiTargetDataset(df, load_any(root, ds_name)["images_dir"], targets, "val",
                            int(ck["config"].get("image_size", 384)))
    print(f"[data] {T}: {ds_name} @ {csv}: {len(ds)} images on disk / {len(np.unique(ds.patients))} "
          f"patients; probes {probes}; {s.probe_folds} folds")
    Z = predict(net, DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                                worker_init_fn=seed_worker), dev, embed=True)
    rows, strata, pred, info = gate_probe_folds(ds.df, Z, ds.label_array(), ds.patients, probes,
                                                {t: net.index(t) for t in probes}, T, s, audits)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "capability_probes_oof.csv"), index=False)
        pd.DataFrame(strata).to_csv(os.path.join(out, "quality_strata_probes_oof.csv"), index=False)
    if pred is not None:
        pred.to_csv(os.path.join(out, f"predictions_{T}_probes_oof{s.probe_folds}.csv"), index=False)
    with open(os.path.join(out, "probe_folds.json"), "w") as f:
        json.dump({"ckpt": a.ckpt, "settings": vars(s), "info": info, "rows": rows}, f, indent=2, default=float)
    print(f"\n[done] {out}/capability_probes_oof.csv, predictions_{T}_probes_oof{s.probe_folds}.csv, "
          f"probe_folds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
