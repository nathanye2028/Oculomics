#!/usr/bin/env python3
"""
score_once.py
=============
RetinaReach's once-only external test: the frozen checkpoints scored on a dataset
the build never touched (Messidor-2), exactly once, with a ledger that refuses a
second look.

    # minutes, spends nothing: data, checkpoints, one forward pass, the ledger's state
    python score_once.py --dataset messidor2 --root <Messidor2> \\
        --checkpoints ck_rr_canon/seed*.pt --preflight
    # the look itself, after the analysis freeze, with the primary result named first
    python score_once.py --dataset messidor2 --root <Messidor2> \\
        --checkpoints ck_rr_canon/seed*.pt --primary calibrated:em:dr_or_edema \\
        --calibration full --plan analysis_plan.md --out exp_messidor2_once

What one look does, per frozen checkpoint
-----------------------------------------
* **Calibration on the camera's own pool, never on a patient it then scores.** The
  patients fall into ``--folds`` (5) patient-grouped folds; each fold is scored
  with statistics re-estimated, labels unread, on the other four -- about the 80 %
  pool an unfamiliar camera gets in a training run -- so every patient is scored
  once and every calibration is inductive.
* **Protocols.** ``trained``: as-trained statistics, the shipped threshold.
  ``calibrated``: ``--calibration full`` = AdaBN on every BN layer (the trainer's
  calibrated protocol); ``shallow`` = the first ``--shallow-frac`` of the layers
  re-estimated from the tabletop profile (the redesign's candidate). Calibrated
  scores are read at the shipped threshold and at the label-free ``em`` and
  ``anchor`` thresholds re-set from the same fold's unlabelled pool.
* **Gate rows** (capability_gate.gate_cell) per target and protocol at the shipped
  threshold. Messidor-2 carries no metadata and an external set's labels fit
  nothing -- they only score -- so the comparator is chance.
* **Operating points** per threshold method and **per-1,000 referral outcomes**
  per patient (referral_outcomes.py: refer if any gradable eye fires, retake when
  none is gradable; rules ``dr_referable`` and ``dr_or_edema``), against the
  IDx-DR pivotal targets.

The ledger (``--ledger``; default ``external_test_ledger.json`` beside this script)
-----------------------------------------------------------------------------------
Before any score exists the look is OPENED: dataset, root, the checkpoints'
SHA-256, the code revision, the plan file's SHA-256, ``--primary``, the
calibration mode, host and time. Nothing about performance is printed until every
checkpoint is scored; then the outputs are written and the look is marked
COMPLETED with its headline numbers. A completed look refuses every later run on
that dataset, and there is no flag to override it: deleting the entry is a
decision to disclose, and git history shows it once the ledger is committed
(commit it right after the look). A look that was opened but never completed (a
crash, so nothing was shown) may be retried, with the same checkpoints only.

Outputs (``--out``): ``once.json`` (provenance, primary result, summary),
``capability_<dataset>.csv``, ``operating_point_<dataset>.csv``,
``referral_<dataset>.csv``, ``seed<s>/predictions_<dataset>_<protocol>.csv``.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capability_gate import encoding_audit, format_table, gate_cell  # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metadata_model import load_table, target_vector  # noqa: E402
from referral_outcomes import RULES, eye_table, patient_outcomes  # noqa: E402
from retinareach import (THRESHOLD_METHODS, ThresholdReference, bn_layers,  # noqa: E402
                         build_from_checkpoint, calibrate_blended, label_free_threshold,
                         self_calibrate, stats_from_lists)
from train_retinareach import ONCE_ONLY_DATASETS, MultiTargetDataset, predict  # noqa: E402

DEFAULT_LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_test_ledger.json")
PROTOCOL_METHODS = (("trained", "shipped"),) + tuple(("calibrated", m) for m in THRESHOLD_METHODS)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_revision() -> Dict[str, object]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", here, "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"commit": rev or None, "uncommitted_changes": dirty}
    except Exception:  # noqa: BLE001 - provenance is best effort; the hashes are what bind
        return {"commit": None, "uncommitted_changes": None}


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
def read_ledger(path: str) -> dict:
    if not os.path.exists(path):
        return {"about": "RetinaReach once-only external tests: one completed look per dataset",
                "looks": []}
    with open(path) as f:
        return json.load(f)


def write_ledger(path: str, ledger: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, path)


def check_ledger(ledger: dict, dataset: str, hashes: Sequence[str]) -> Tuple[bool, str, int]:
    """(allowed, why, attempt). A completed look at ``dataset`` refuses; an opened,
    never-completed one (nothing was shown) allows a retry with the SAME
    checkpoints only."""
    prior = [l for l in ledger.get("looks", []) if l.get("dataset") == dataset]
    done = [l for l in prior if l.get("completed_utc")]
    if done:
        return False, (f"{dataset} was already scored: look opened {done[0]['opened_utc']}, completed "
                       f"{done[0]['completed_utc']} with {len(done[0]['checkpoints'])} checkpoint(s). "
                       f"It is the once-only external test, so this script will not score it again. "
                       f"A second look is a decision to make, and disclose, outside this script."), 0
    if prior and any(sorted(l["checkpoints"].values()) != sorted(hashes) for l in prior):
        return False, (f"an earlier look at {dataset} (opened {prior[-1]['opened_utc']}) never completed "
                       f"and used DIFFERENT checkpoints; a retry must score the same frozen files"), 0
    return True, ("retry of a look that never completed (no result was shown)" if prior
                  else "first look"), len(prior) + 1


def open_look(path: str, entry: dict) -> None:
    ledger = read_ledger(path)
    ledger.setdefault("looks", []).append(entry)
    write_ledger(path, ledger)


def complete_look(path: str, opened_utc: str, dataset: str, update: dict) -> None:
    ledger = read_ledger(path)
    for l in ledger["looks"]:
        if l.get("dataset") == dataset and l.get("opened_utc") == opened_utc:
            l.update(update, completed_utc=_now())
            break
    else:                                       # pragma: no cover - the entry was written above
        raise RuntimeError(f"open look {opened_utc} for {dataset} vanished from {path}")
    write_ledger(path, ledger)


# --------------------------------------------------------------------------- #
# Checkpoints, data, folds
# --------------------------------------------------------------------------- #
def load_checkpoints(paths: Sequence[str]) -> List[Tuple[str, dict]]:
    """The frozen checkpoints, which must be one sweep: same backbone, targets,
    image size and source camera (seeds may differ)."""
    out = [(p, torch.load(p, map_location="cpu", weights_only=False)) for p in paths]
    key = lambda ck: (ck["config"]["backbone"], tuple(ck["config"]["source_targets"]),
                      int(ck["config"].get("image_size", 384)), ck["args"].get("source_camera"),
                      int(ck["args"].get("split_seed", 42)))
    keys = {key(ck) for _, ck in out}
    if len(keys) != 1:
        raise SystemExit(f"[fatal] the checkpoints are not one sweep (backbone, targets, size, "
                         f"source camera, split seed differ): {sorted(keys)}")
    seeds = [int(ck["args"].get("seed", -1)) for _, ck in out]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"[fatal] two checkpoints share a training seed: {seeds}")
    return out


def make_folds(df: pd.DataFrame, strat_task: str, folds: int, seed: int) -> np.ndarray:
    """Fold index per row, patient-grouped, stratified on ``strat_task`` (missing
    labels form their own stratum)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = target_vector(df, strat_task)
    strata = np.where(np.isnan(y), -1, y).astype(int)
    groups = df["patient"].astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise SystemExit(f"[fatal] {len(np.unique(groups))} patients cannot fill {folds} folds")
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
                                .split(df, strata, groups)):
        fold[te] = k
    return fold


def _calibrate(model, ck: dict, pool, a, dev, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(pool, batch_size=a.calib_batch or int(ck["args"].get("calib_batch", 16)),
                        shuffle=True, generator=g, num_workers=a.num_workers,
                        worker_init_fn=seed_worker)
    if a.calibration == "full":
        return self_calibrate(model, loader, dev)[0]
    names = [n for n, _ in bn_layers(model)]
    k = int(round(a.shallow_frac * len(names)))
    source = ck["args"].get("source_name", "tabletop")
    tab = stats_from_lists(ck["profiles"][source]["bn_stats"])
    return calibrate_blended(model, loader, dev, adapt=names[:k], source=tab)[0]


def score_checkpoint(ck: dict, ds, fold: np.ndarray, a, dev) -> Dict[str, object]:
    """Scores [N, K] under both protocols and the per-row thresholds of every
    (protocol, method), for one frozen checkpoint. Reads no label."""
    model = build_from_checkpoint(ck).to(dev)
    seed = int(ck["args"].get("seed", 0))
    seed_everything(seed)
    dl = lambda d: DataLoader(d, batch_size=a.batch_size, num_workers=a.num_workers,
                              worker_init_fn=seed_worker)
    n, src = len(ds), list(ck["config"]["source_targets"])
    q = {"trained": predict(model, dl(ds), dev), "calibrated": np.full((n, len(model.targets)), np.nan)}
    thr = {(p, m): {t: np.full(n, np.nan) for t in src} for p, m in PROTOCOL_METHODS}
    refs = {t: ThresholdReference.from_dict(r) for t, r in (ck.get("threshold_reference") or {}).items()}
    for t in src:
        thr[("trained", "shipped")][t][:] = ck["thresholds"]["trained"].get(t, np.nan)
    for k in range(int(fold.max()) + 1):
        test_idx, pool_idx = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        print(f"    fold {k + 1}/{fold.max() + 1}: calibrating on {len(pool_idx)} unlabelled images, "
              f"scoring {len(test_idx)}", flush=True)
        cal = _calibrate(model, ck, Subset(ds, list(pool_idx)), a, dev, seed + k)
        q["calibrated"][test_idx] = predict(cal, dl(Subset(ds, list(test_idx))), dev)
        q_pool = predict(cal, dl(Subset(ds, list(pool_idx))), dev)
        del cal
        for t in src:
            j = model.index(t)
            thr[("calibrated", "shipped")][t][test_idx] = ck["thresholds"]["calibrated"].get(t, np.nan)
            for m in THRESHOLD_METHODS[1:]:
                if t in refs:
                    thr[("calibrated", m)][t][test_idx] = label_free_threshold(q_pool[:, j], refs[t], m)[0]
    return {"q": q, "thr": thr, "targets": list(model.targets), "source_targets": src, "seed": seed}


# --------------------------------------------------------------------------- #
# Reading the scores (only after every checkpoint is scored)
# --------------------------------------------------------------------------- #
def _op(y: np.ndarray, p: np.ndarray, thr: np.ndarray, g: np.ndarray, n_boot: int, seed: int) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(thr)
    y, flag, g = y[ok].astype(int), (p[ok] >= thr[ok]), g[ok]
    pos, neg = y == 1, y == 0
    out = {"n": int(ok.sum()), "pos": int(pos.sum()),
           "sensitivity": float(flag[pos].mean()) if pos.any() else float("nan"),
           "specificity": float((~flag[neg]).mean()) if neg.any() else float("nan"),
           "ppv": float(y[flag].mean()) if flag.any() else float("nan"),
           "flagged_fraction": float(flag.mean()) if len(flag) else float("nan"),
           "threshold_mean": float(np.nanmean(thr[ok])) if ok.any() else float("nan")}
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    sens = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) if members else []
        if len(idx) and pos[idx].any():
            sens.append(float(flag[idx][pos[idx]].mean()))
    out["sens_ci"] = ([float(np.percentile(sens, 2.5)), float(np.percentile(sens, 97.5))] if sens
                      else [float("nan")] * 2)
    return out


def read_scores(scored: List[dict], df: pd.DataFrame, a, ck0: dict) -> Dict[str, pd.DataFrame]:
    args0 = ck0["args"]
    gate, ops, refs = [], [], []
    g = df["patient"].astype(str).to_numpy()
    for s in scored:
        seed = s["seed"]
        for t in s["source_targets"]:
            j, y = s["targets"].index(t), target_vector(df, t)
            for proto in ("trained", "calibrated"):
                thr0 = s["thr"][(proto, "shipped")][t]            # one value on every row
                row = gate_cell(t, a.dataset, proto, df, s["q"][proto][:, j], float(np.nanmean(thr0)),
                                {"reason": "external set: its labels fit nothing; comparator = chance"},
                                target_sens=float(args0.get("target_sens", 0.85)),
                                sens_tolerance=float(args0.get("sens_tolerance", 0.10)),
                                gate_margin=float(args0.get("gate_margin", 0.02)),
                                min_pos_patients=int(args0.get("min_test_pos_patients", 20)),
                                n_boot=a.n_boot, seed=seed, audit=encoding_audit(df, t))
                gate.append(dict(row, seed=seed))
            for proto, m in PROTOCOL_METHODS:
                ops.append({"seed": seed, "target": t, "protocol": proto, "method": m,
                            **_op(y, s["q"][proto][:, j], s["thr"][(proto, m)][t], g, a.n_boot, seed)})
        for proto, m in PROTOCOL_METHODS:
            pred = pd.DataFrame({"file": df["file"], "patient": df["patient"].astype(str)})
            for t in s["source_targets"]:
                pred[f"p_{t}"] = s["q"][proto][:, s["targets"].index(t)]
                pred[f"threshold_{t}"] = s["thr"][(proto, m)][t]
            for rule in RULES:
                eyes = eye_table(pred, df, rule)
                if eyes is None:
                    continue
                o = patient_outcomes(eyes, n_boot=a.n_boot, seed=seed, prevalences=a.prevalence)
                o.update(seed=seed, protocol=proto, method=m, rule=rule,
                         meets_targets=bool(o.get("sensitivity", 0) >= a.target_sens
                                            and o.get("specificity", 0) >= a.target_spec))
                refs.append(o)
    return {"gate": pd.DataFrame(gate), "ops": pd.DataFrame(ops), "referral": pd.DataFrame(refs)}


def primary_summary(tabs: Dict[str, pd.DataFrame], primary: str) -> dict:
    proto, method, rule = primary.split(":")
    r = tabs["referral"]
    r = r[(r["protocol"] == proto) & (r["method"] == method) & (r["rule"] == rule)]
    if r.empty:
        return {"primary": primary, "reason": "no rows for the primary protocol / method / rule"}
    return {"primary": primary, "seeds": int(r["seed"].nunique()),
            "sensitivity_mean": float(r["sensitivity"].mean()),
            "specificity_mean": float(r["specificity"].mean()),
            "sensitivity_by_seed": dict(zip(r["seed"].astype(int).astype(str), r["sensitivity"])),
            "specificity_by_seed": dict(zip(r["seed"].astype(int).astype(str), r["specificity"])),
            "seeds_meeting_targets": int(r["meets_targets"].sum()),
            "per1000_missed_mean": float(r["per1000_missed"].mean()),
            "per1000_unnecessary_mean": float(r["per1000_unnecessary"].mean()),
            "per1000_retakes_mean": float(r["per1000_retakes"].mean())}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_primary(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    parts = s.split(":")
    ok = len(parts) == 3 and (parts[0], parts[1]) in PROTOCOL_METHODS and parts[2] in RULES
    if not ok:
        raise SystemExit(f"[fatal] --primary {s!r}: expected PROTOCOL:METHOD:RULE, one of "
                         f"{[f'{p}:{m}:<rule>' for p, m in PROTOCOL_METHODS]} with rule in {list(RULES)}")
    return s


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RetinaReach's once-only external test.")
    p.add_argument("--dataset", required=True, choices=list(ONCE_ONLY_DATASETS))
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="the frozen train_retinareach.py checkpoints (one sweep; globs allowed)")
    p.add_argument("--primary", default=None,
                   help="PROTOCOL:METHOD:RULE named before the look, e.g. calibrated:em:dr_or_edema "
                        "(required unless --preflight)")
    p.add_argument("--calibration", default="full", choices=["full", "shallow"])
    p.add_argument("--shallow-frac", type=float, default=0.5)
    p.add_argument("--plan", default=None, help="the dated analysis plan; its SHA-256 is recorded")
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--out", default=None, help="output dir (default exp_<dataset>_once)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--calib-batch", type=int, default=0, help="default: the checkpoint's")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--target-spec", type=float, default=0.825)
    p.add_argument("--prevalence", type=float, nargs="*", default=[0.055, 0.156])
    p.add_argument("--preflight", action="store_true",
                   help="check data, checkpoints, one forward pass and the ledger; score nothing")
    a = p.parse_args(argv)
    a.primary = _parse_primary(a.primary)
    if not a.preflight and a.primary is None:
        raise SystemExit("[fatal] name the primary result before the look: --primary "
                         "PROTOCOL:METHOD:RULE (e.g. calibrated:em:dr_or_edema)")
    if a.plan and not os.path.isfile(a.plan):
        raise SystemExit(f"[fatal] --plan {a.plan} does not exist")
    paths = sorted({q for c in a.checkpoints for q in (glob.glob(c) or [c])})
    missing = [q for q in paths if not os.path.isfile(q)]
    if missing:
        raise SystemExit(f"[fatal] checkpoint(s) not found: {missing}")
    hashes = {q: sha256(q) for q in paths}
    ledger = read_ledger(a.ledger)
    allowed, why, attempt = check_ledger(ledger, a.dataset, list(hashes.values()))
    print(f"[ledger] {a.ledger}: {why}")
    if not allowed:
        raise SystemExit(f"[refused] {why}")

    from train_mbrset import pick_device
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    cks = load_checkpoints(paths)
    ck0 = cks[0][1]
    df_all, csv = load_table(a.root, a.dataset)
    from brset_dataset import load_any
    img_dir = load_any(a.root, a.dataset)["images_dir"]
    size = int(ck0["config"].get("image_size", 384))
    model0 = build_from_checkpoint(ck0)
    ds = MultiTargetDataset(df_all, img_dir, model0.targets, "val", size)
    df = ds.df.reset_index(drop=True)
    print(f"[data] {a.dataset} @ {csv}: {len(df_all)} rows, {len(ds)} images on disk, "
          f"{df['patient'].nunique()} patients; checkpoints {len(cks)} "
          f"(seeds {[int(ck['args'].get('seed', -1)) for _, ck in cks]}), {ck0['config']['backbone']} @ {size}")
    if len(ds) == 0:
        raise SystemExit(f"[fatal] no {a.dataset} image found under {img_dir}")
    src = list(ck0["config"]["source_targets"])
    fold = make_folds(df, src[0], a.folds, int(ck0["args"].get("split_seed", 42)))

    if a.preflight:
        model0 = model0.to(dev)
        x = next(iter(DataLoader(Subset(ds, list(range(min(len(ds), 8)))), batch_size=8)))["image"]
        with torch.no_grad():
            out = torch.sigmoid(model0(x.to(dev)))
        cal = _calibrate(model0, ck0, Subset(ds, list(range(min(len(ds), 16)))), a, dev, 0)
        stats_ok = all(torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all()
                       for _, m in bn_layers(cal))
        refs = sorted(ck0.get("threshold_reference") or {})
        problems = ([] if torch.isfinite(out).all() else ["non-finite model output"]) + \
                   ([] if stats_ok else ["non-finite calibration statistics"]) + \
                   ([] if all(t in ck0["thresholds"]["calibrated"] for t in src)
                    else ["a source target has no shipped threshold"])
        print(f"[preflight] folds of {np.bincount(fold).tolist()} images; thresholds for {src}; "
              f"label-free references for {refs}; forward pass and calibration on {dev}: "
              f"{'finite' if not problems else problems}")
        if problems:
            print("PREFLIGHT FAILED")
            return 1
        print(f"PREFLIGHT OK -- nothing was scored; the {a.dataset} look is unspent")
        return 0

    out_dir = a.out or f"exp_{a.dataset}_once"
    os.makedirs(out_dir, exist_ok=True)
    opened = _now()
    entry = {"dataset": a.dataset, "root": os.path.realpath(a.root), "opened_utc": opened,
             "attempt": attempt, "checkpoints": hashes, "primary": a.primary,
             "calibration": a.calibration,
             "shallow_frac": a.shallow_frac if a.calibration == "shallow" else None,
             "folds": a.folds, "plan": ({"path": os.path.abspath(a.plan), "sha256": sha256(a.plan)}
                                        if a.plan else None),
             "code": code_revision(), "host": socket.gethostname(), "out": os.path.abspath(out_dir)}
    open_look(a.ledger, entry)
    print(f"[ledger] look OPENED {opened} (attempt {attempt}); no result is shown until every "
          f"checkpoint is scored")

    scored = []
    for path, ck in cks:
        print(f"\n[score] {path} (seed {ck['args'].get('seed')}), calibration {a.calibration}", flush=True)
        scored.append(score_checkpoint(ck, ds, fold, a, dev))
    tabs = read_scores(scored, df, a, ck0)
    for s in scored:
        sd = os.path.join(out_dir, f"seed{s['seed']}")
        os.makedirs(sd, exist_ok=True)
        for proto in ("trained", "calibrated"):
            pr = pd.DataFrame({"file": df["file"], "patient": df["patient"].astype(str), "fold": fold})
            for t in s["targets"]:
                pr[f"p_{t}"] = s["q"][proto][:, s["targets"].index(t)]
            for (p_, m), d in s["thr"].items():
                if p_ == proto:
                    for t, v in d.items():
                        pr[f"threshold_{t}" + ("" if m == "shipped" else f"__{m}")] = v
            pr.to_csv(os.path.join(sd, f"predictions_{a.dataset}_{proto}.csv"), index=False)
    tabs["gate"].to_csv(os.path.join(out_dir, f"capability_{a.dataset}.csv"), index=False)
    tabs["ops"].to_csv(os.path.join(out_dir, f"operating_point_{a.dataset}.csv"), index=False)
    tabs["referral"].to_csv(os.path.join(out_dir, f"referral_{a.dataset}.csv"), index=False)
    prim = primary_summary(tabs, a.primary)
    summary = {"dataset": a.dataset, "look": entry, "primary": prim,
               "gate_status_by_seed": tabs["gate"].groupby(["target", "protocol"])["status"]
               .apply(list).to_dict() if len(tabs["gate"]) else {}}
    summary["gate_status_by_seed"] = {f"{k[0]}|{k[1]}": v for k, v in summary["gate_status_by_seed"].items()}
    with open(os.path.join(out_dir, "once.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    complete_look(a.ledger, opened, a.dataset, {"primary_result": prim,
                                                "outputs": sorted(os.listdir(out_dir))})

    print(f"\n=== {a.dataset}: capability gate per seed (comparator = chance) ===")
    from capability_gate import TABLE_COLS
    print(format_table(tabs["gate"].to_dict("records"), cols=["seed"] + list(TABLE_COLS)))
    print(f"\n=== {a.dataset}: operating points (mean over seeds) ===")
    print(tabs["ops"].groupby(["target", "protocol", "method"], sort=False)
          [["sensitivity", "specificity", "ppv", "flagged_fraction"]].mean()
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n=== {a.dataset}: referral outcomes per patient (mean over seeds) ===")
    cols = ["sensitivity", "specificity", "per1000_missed", "per1000_unnecessary", "per1000_retakes"]
    print(tabs["referral"].groupby(["rule", "protocol", "method"], sort=False)[cols].mean()
          .to_string(float_format=lambda v: f"{v:.3f}" if abs(v) <= 1 else f"{v:.1f}"))
    if "sensitivity_mean" in prim:
        print(f"\nPRIMARY ({a.primary}): sensitivity {prim['sensitivity_mean']:.3f}, specificity "
              f"{prim['specificity_mean']:.3f} (mean of {prim['seeds']} seeds); targets met in "
              f"{prim['seeds_meeting_targets']}/{prim['seeds']} seeds; per 1,000 screened "
              f"{prim['per1000_missed_mean']:.1f} missed, {prim['per1000_unnecessary_mean']:.1f} "
              f"unnecessary referrals, {prim['per1000_retakes_mean']:.1f} retakes")
    print(f"\n[ledger] look COMPLETED; {a.dataset} will not be scored again. Commit {a.ledger} "
          f"and {out_dir}/ now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
