#!/usr/bin/env python3
"""
train_retinareach.py
====================
Build and evaluate one **RetinaReach** model (``retinareach.py``) end to end:
train on the tabletop source, self-calibrate to the handheld device from its
unlabelled images, fit the systemic probes, fix every threshold on home data,
and emit the per-target, per-device capability table the phone ships.

    python train_retinareach.py --source-root <BRSET> --target-root <mBRSET> \\
        --image-size 384 --epochs 30 --ema-decay 0.999 --seed 0 \\
        --out exp_retinareach/seed0 --ckpt ck_retinareach/seed0.pt

Pipeline (one seed)
-------------------
1. **Splits.** One patient-grouped 70/10/20 partition per dataset
   (``capability_gate.multitask_split``, ``--split-seed`` 42, fixed across
   seeds): every head and the metadata comparator share it.
2. **Stage 1 -- source trunk.** MobileNetV4 + the source-target heads
   (``dr_referable``, ``edema``) on BRSET train; masked BCE (a row missing one
   label still trains the others), sampler balanced on the first source target
   (the DR recipe of record), warmup-cosine, optional EMA; selection on mean
   BRSET-val AUROC. mBRSET contributes nothing here -- no image, no label.
3. **Self-calibration.** AdaBN on each device's *pool* (train+val images,
   labels unread): ``tabletop`` and ``handheld`` profiles. The test split is
   never in a pool, so calibration is inductive -- calibrate once, then screen
   patients the calibration never saw, as a clinic would.
4. **Stage 2 -- systemic probes.** For each probe target, a standardised L2
   logistic regression on the *calibrated* handheld embeddings (train rows; C
   by val AUROC), folded into its row of the head. The trunk stays frozen.
5. **Thresholds.** Fixed on the home validation split at ``--target-sens``,
   never on the device they are applied to: source targets on BRSET val, under
   each protocol; probes on mBRSET val.
6. **Evaluation.** Every target x device x protocol --
   ``trained`` (as-trained BN statistics: what ships today) and ``calibrated``
   (each device's own AdaBN statistics: RetinaReach). AUROC with a
   patient-cluster CI, patient-level AUROC, sens/spec/PPV/flagged fraction at the
   shipped threshold, the oracle threshold for reference.
7. **Capability gate** (``capability_gate.gate_cell``) per cell against the
   strongest intake-form metadata model on the same test rows.
8. **Calibration-size sweep.** How many unlabelled captures does a clinic
   need? Random N-image subsets of the handheld pool (``--calib-sizes`` x
   ``--calib-repeats``), with and without the Schneider prior
   (``--prior-strengths``): handheld test AUROC, sensitivity at the shipped
   threshold, flagged fraction.
9. **Envelopes and recognition.** Each profile's envelope = 95th percentile of
   the same-camera distance of N-capture calibrations drawn from its own pool
   (``--envelope-repeats``). Recognition is then scored OUT OF SAMPLE on
   N-capture draws from each device's test patients (labels unread): right
   camera and inside its envelope, and how often the other camera's profile
   would falsely accept it. The printed device report uses fresh test
   captures too (``--demo-n``), never a profile compared with itself.
10. **Controls and mechanism checks** (plan 5.7-5.8). Every gate row carries
   the quality-only AUROC (image-quality flags alone as predictors), Brier /
   ECE / mean predicted risk; every probe a permuted-label refit
   (``--n-shuffle``, must sit at 0.5); ``--shuffle-source-labels`` makes the
   whole run a negative control. ``quality_strata.csv``: AUROC and sensitivity
   on gradable vs ungradable images. Camera decodability: a patient-grouped
   linear read-out of camera identity from the trunk embedding, as-trained vs
   calibrated -- does self-calibration remove the camera from the representation?
   Vessel ablation (``vessel_ablation.py``, ``anatomy_ablation.csv``): every test
   image scored intact, with the vasculature inpainted, and with the same mask
   rotated 90 degrees; ``vessels_minus_control`` (paired, patient bootstrap)
   says whether a target leans on the vessels more than on other retina.
11. **Unfamiliar cameras** (``--unfamiliar ROOT DATASET [NAME]``, e.g. ODIR-5K):
   a labelled camera the model is not validated on. It builds nothing; its
   labels only score. The full gate on it (what a validation study would find)
   is compared with what the phone would SHOW after N unlabelled captures:
   ``unsafe_rate`` (shown but not supported) and ``missed_rate``.

Outputs (``--out``): ``results.json``, ``capability.csv``,
``calibration_sweep.csv``, ``quality_strata.csv``,
``capability_unfamiliar.csv`` + ``unfamiliar_lookup.csv`` (with --unfamiliar),
``predictions_<device>_<protocol>.csv``, ``split_<device>.csv``,
``profiles.json`` (what ``export_retinareach.py`` ships). ``--ckpt`` holds the weights (as-trained statistics + probes) with the
config, thresholds, profiles and capability rows.

Everything paired within a run (calibrated vs trained, image vs metadata) is
immune to training nondeterminism; anything across runs needs the multi-seed
summary (``summarize_retinareach.py``).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capability_gate import (GATE_FEATURE_SETS, decodability, encoding_audit,  # noqa: E402
                             format_table, full_chart_auroc, gate_cell, label_matrix,
                             metadata_comparator, multitask_split, paired_delta_bootstrap,
                             quality_only_auroc, quality_strata)
from vessel_ablation import ablate_normalised  # noqa: E402
from dataset import DeviceAug, LabelSpec, MBRSETDataset  # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metadata_model import FEATURE_SETS, _auc, load_table, threshold_at_sensitivity  # noqa: E402
from retinareach import (DEFAULT_BACKBONE, PROBE_TARGETS, SOURCE_TARGETS,  # noqa: E402
                         DeviceProfile, RetinaReachNet, camera_distance,
                         device_capability, get_bn_stats, self_calibrate, set_bn_stats,
                         stats_to_lists)
from train_mbrset import ModelEMA, pick_device  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class MultiTargetDataset(MBRSETDataset):
    """``MBRSETDataset`` with a [K] float label vector (NaN = missing) instead of
    one task, and every row kept: an image unlabelled for one target still
    trains the others and still counts as an unlabelled calibration capture.
    Same decode / FOV crop / resize / normalisation code as every other run, so
    the two devices differ only in their pixels."""

    def __init__(self, df: pd.DataFrame, images_dir: str, targets: Sequence[str], split: str,
                 image_size: int, device_aug: bool = False) -> None:
        self._rr_targets = list(targets)
        files = df["file"].astype(str)
        if files.str.contains("/", regex=False).any():      # public sets: nested folders
            ok = files.map(lambda f: os.path.isfile(os.path.join(images_dir, f)))
        else:
            present = set(os.listdir(images_dir)) if os.path.isdir(images_dir) else set()
            ok = files.isin(present)
        df = df[ok.to_numpy()].reset_index(drop=True)
        super().__init__(csv=df, images_dir=images_dir, task="retinareach", split=split,
                         image_size=image_size, drop_missing_labels=False,
                         drop_missing_files=False, fov_crop=True, device_aug=device_aug)
        self.df = df
        self.patients = df["patient"].to_numpy()

    def label_array(self) -> np.ndarray:
        """An independent numpy COPY of the [N, K] labels. Never a ``.numpy()``
        view: with spawn-started DataLoader workers (macOS) torch moves the
        tensor's storage into shared memory the first time a loader pickles the
        dataset, and a view taken earlier is left pointing at freed memory
        (measured: validation AUROC silently undefined on the Mac)."""
        return self.labels.numpy().copy()

    def _resolve_labels(self, df, task, label_col, label_map):
        spec = LabelSpec(source_cols=(), fn=lambda r: np.nan, dtype=torch.float32,
                         num_classes=len(self._rr_targets), multilabel=True,
                         description="RetinaReach target vector")
        return spec, label_matrix(df, self._rr_targets)


def balanced_weights(y: np.ndarray) -> torch.Tensor:
    """Per-row sampler weights balancing one binary column; rows missing it
    weigh like the majority class."""
    pos, neg = np.nansum(y == 1), np.nansum(y == 0)
    if pos == 0 or neg == 0:
        return torch.ones(len(y), dtype=torch.double)
    w = np.where(y == 1, 0.5 / pos, 0.5 / neg)
    w = np.where(np.isnan(y), 0.5 / max(pos, neg), w)
    return torch.as_tensor(w, dtype=torch.double)


@torch.no_grad()
def predict(model: nn.Module, loader, device, embed: bool = False, transform=None) -> np.ndarray:
    """Sigmoid probabilities [N, K] (or pooled embeddings [N, D]); ``transform``
    (normalised batch -> normalised batch) perturbs the inputs, e.g. an ablation."""
    model.eval()
    out = []
    for b in loader:
        # non_blocking only on CUDA: on MPS (torch 2.8) a non-blocking copy of a
        # strided CPU tensor delivers garbage -- measured, every label batch.
        x = b["image"].to(device, non_blocking=device.type == "cuda")
        if transform is not None:
            x = transform(x)
        out.append((model.embed(x) if embed else torch.sigmoid(model(x))).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def with_stats(model: nn.Module, stats) -> nn.Module:
    m = copy.deepcopy(model)
    set_bn_stats(m, stats)
    return m.eval()


# --------------------------------------------------------------------------- #
# Stage 2: linear probes on the calibrated embedding
# --------------------------------------------------------------------------- #
def fit_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
              Cs: Sequence[float], seed: int = 0) -> Dict[str, object]:
    """Standardised, class-balanced L2 logistic regression; C chosen on val
    AUROC; returned already folded into raw-embedding units so it drops straight
    into the network's head row: logit = w . z + b."""
    from sklearn.linear_model import LogisticRegression
    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-6
    Z_tr, Z_va = (X_tr - mu) / sd, (X_va - mu) / sd
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(Z_tr, y_tr)
        a = _auc(y_va, clf.decision_function(Z_va)) if len(np.unique(y_va)) == 2 else float("nan")
        if best is None or (a == a and (best[0] != best[0] or a > best[0])):
            best = (a, C, clf)
    val_auc, C, clf = best
    w = clf.coef_.ravel() / sd
    b = float(clf.intercept_[0] - np.sum(clf.coef_.ravel() * mu / sd))
    return {"weight": w, "bias": b, "C": float(C), "val_auroc": float(val_auc),
            "n_train": int(len(y_tr)), "pos_train": int(y_tr.sum())}


def probe_shuffle(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                  C: float, n: int, seed: int = 0) -> Dict[str, float]:
    """Negative control (plan 5.8): the same probe refit ``n`` times on
    PERMUTED training labels, scored on the real test labels. The mean must sit
    at 0.5; one drifting toward the real AUROC means the pipeline leaks."""
    from sklearn.linear_model import LogisticRegression
    if n <= 0 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "se": float("nan"), "n": 0}
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    rng = np.random.default_rng(seed)
    aucs = [_auc(y_te, LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                          random_state=seed)
                 .fit(Z_tr, rng.permutation(y_tr)).decision_function(Z_te)) for _ in range(n)]
    return {"auroc": float(np.mean(aucs)),
            "se": float(np.std(aucs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), "n": n}


def run_preflight(a, model, dev, train_loader, loaders, datasets: Dict[str, dict], audits,
                  targets: Sequence[str], nS: int) -> int:
    """Minutes, not hours: everything the full run depends on, checked before the
    GPU time is spent. Data (roots, encodings, images on disk, test-split power
    per target and device, unfamiliar cameras), the device (one training step:
    finite, plausible loss; one calibration on the target pool: finite
    statistics), the pretrained weights (already loaded by the time this runs),
    disk, and a wall-clock estimate. Returns 0 when nothing blocks the run."""
    import shutil
    problems, notes = [], []
    print("\n=== PREFLIGHT ===")
    for name, d in datasets.items():
        test = d["test"]
        y = test.label_array()
        print(f"  {name}: pool {len(d['pool'])} / test {len(test)} images on disk")
        for k, t in enumerate(targets):
            au = audits[name][t]
            if not au["ok"]:
                continue
            ok = ~np.isnan(y[:, k])
            pos_pat = len(set(test.patients[ok & (y[:, k] == 1)]))
            flag = "  UNDERPOWERED" if pos_pat < a.min_test_pos_patients else ""
            print(f"    {t:<22} test rows {int(ok.sum()):>5}  positives {int(np.nansum(y[ok, k])):>4}  "
                  f"positive patients {pos_pat:>4}{flag}")
    for spec in a.unfamiliar:
        root_u, ds_u, name_u = parse_unfamiliar(spec)
        try:
            df_u, csv_u = load_table(root_u, ds_u)
            from brset_dataset import load_any
            dir_u = load_any(root_u, ds_u)["images_dir"]
            n_img = len(MultiTargetDataset(df_u, dir_u, targets, "val", a.image_size))
            ok_t = [t for t in targets if encoding_audit(df_u, t)["ok"]]
            print(f"  unfamiliar {name_u}: {len(df_u)} rows, {n_img} images on disk; labelled: {ok_t}")
            if n_img == 0:
                problems.append(f"unfamiliar camera {name_u}: no images under {dir_u}")
            if not ok_t:
                problems.append(f"unfamiliar camera {name_u}: no target is labelled")
        except Exception as e:  # noqa: BLE001 - report every cause, keep checking the rest
            problems.append(f"unfamiliar camera {name_u}: {e}")

    # one training step on the real device
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    dev_aug = DeviceAug() if (a.gpu_aug if a.gpu_aug is not None else dev.type == "cuda") else None
    cuda = dev.type == "cuda"
    losses, t_steps = [], []
    it = iter(train_loader)
    for step in range(3):
        b = next(it)
        t0 = time.time()
        x = b["image"].to(dev, non_blocking=cuda)
        y = b["label"][:, :nS].contiguous().to(dev, non_blocking=cuda)
        if dev_aug is not None:
            x = dev_aug(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x)[:, :nS]
        mask = ~torch.isnan(y)
        loss = (F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), reduction="none")
                * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        t_steps.append(time.time() - t0)
    bad = [v for v in losses if not (0.0 <= v < 20.0)]
    print(f"  training steps on {dev}: losses {[round(v, 3) for v in losses]}  "
          f"({a.batch_size / np.mean(t_steps[1:]):.0f} img/s after warm-up, incl. transfer)")
    if bad:
        problems.append(f"implausible training loss {bad} on {dev}: labels or inputs are corrupted")
    model.eval()
    t0 = time.time()
    tgt_pool = datasets[a.target_name]["pool"]
    n_cal = min(len(tgt_pool), 2 * a.calib_batch)
    cal, info = self_calibrate(model, DataLoader(Subset(tgt_pool, list(range(n_cal))),
                                                 batch_size=a.calib_batch, num_workers=0), dev)
    vec = torch.cat([torch.cat([m_, v_]) for m_, v_ in get_bn_stats(cal).values()])
    eval_rate = n_cal / max(time.time() - t0, 1e-6)
    print(f"  self-calibration on {n_cal} {a.target_name} images: {info.n_bn_layers} BN layers, "
          f"statistics finite: {bool(torch.isfinite(vec).all())}")
    if not torch.isfinite(vec).all():
        problems.append("self-calibration produced non-finite BatchNorm statistics")
    free_gb = shutil.disk_usage(os.path.abspath(a.out)).free / 1e9
    print(f"  free disk at {a.out}: {free_gb:.1f} GB")
    if free_gb < 5:
        problems.append(f"only {free_gb:.1f} GB free at {a.out}")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
        if free < 0.5 * total:
            notes.append("less than half the GPU memory is free: another job is on this card")

    # wall-clock estimate from the measured rates
    train_rate = a.batch_size / np.mean(t_steps[1:])
    n_train = len(train_loader) * a.batch_size
    tgt_test = len(datasets[a.target_name]["test"])
    sizes = [n for n in a.calib_sizes]
    eval_imgs = (sum(len(d["pool"]) + 4 * len(d["test"]) for d in datasets.values())
                 + len(sizes) * a.calib_repeats * (len(a.prior_strengths) * tgt_test)
                 + 2 * (a.envelope_repeats + a.calib_repeats) * sum(sizes))
    train_min = a.epochs * (n_train / train_rate + len(datasets[a.source_name]["val"]) / eval_rate) / 60
    eval_min = eval_imgs / eval_rate / 60
    print(f"  estimate: stage 1 <= {train_min:.0f} min ({a.epochs} epochs, early stopping may cut it), "
          f"evaluation ~{eval_min:.0f} min, metadata models a few min per device "
          f"(+ unfamiliar cameras)")
    for n_ in notes:
        print(f"  [note] {n_}")
    if problems:
        print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
        return 1
    print("PREFLIGHT OK")
    return 0


def parse_unfamiliar(spec: Sequence[str]) -> tuple:
    if len(spec) not in (2, 3):
        raise SystemExit(f"[fatal] --unfamiliar takes ROOT DATASET [NAME], got {list(spec)}")
    return spec[0], spec[1], spec[2] if len(spec) == 3 else spec[1]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] = None) -> int:
    p = argparse.ArgumentParser(description="Train + evaluate one RetinaReach model.")
    p.add_argument("--source-root", required=True, help="tabletop dataset (labels + images)")
    p.add_argument("--source-dataset", default="brset", choices=["brset", "mbrset"])
    p.add_argument("--target-root", required=True, help="handheld dataset (labels + images)")
    p.add_argument("--target-dataset", default="mbrset", choices=["brset", "mbrset"])
    p.add_argument("--source-name", default="tabletop")
    p.add_argument("--target-name", default="handheld")
    p.add_argument("--source-targets", nargs="+", default=list(SOURCE_TARGETS))
    p.add_argument("--probe-targets", nargs="+", default=list(PROBE_TARGETS))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--imbalance", default="sampler", choices=["sampler", "none"],
                   help="'sampler': batches balanced on the first source target")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--gpu-aug", dest="gpu_aug", action="store_true", default=None)
    p.add_argument("--no-gpu-aug", dest="gpu_aug", action="store_false")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0, help="training seed (init, sampler, augmentation)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="patient split seed, fixed across training seeds")
    # calibration / probes / thresholds / gate
    p.add_argument("--calib-batch", type=int, default=16,
                   help="calibration batch size -- must equal the exported collector's")
    p.add_argument("--calib-sizes", type=int, nargs="*", default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--calib-repeats", type=int, default=5,
                   help="draws per N for the calibration-size sweep and for recognition")
    p.add_argument("--envelope-repeats", type=int, default=20,
                   help="pool draws per N that define each profile's envelope")
    p.add_argument("--unfamiliar", nargs="+", action="append", default=[],
                   metavar="ROOT DATASET [NAME]",
                   help="a camera the model is NOT validated on, with labels for checking only "
                        "(repeatable), e.g. --unfamiliar <ODIR-5K> odir. Its images never build "
                        "a profile; the phone's label-free report is compared with what its "
                        "labels show")
    p.add_argument("--n-shuffle", type=int, default=10,
                   help="permuted-label refits per probe (negative control)")
    p.add_argument("--shuffle-source-labels", action="store_true",
                   help="negative control run: permute the source training labels across "
                        "images; every source-target number must then sit at chance")
    p.add_argument("--ablation-frac", type=float, default=0.12,
                   help="vessel ablation: fraction of the field of view removed as vessel "
                        "(and, rotated, as the control); 0 disables")
    p.add_argument("--preflight", action="store_true",
                   help="check data, encodings, power, device, disk and timing, then exit "
                        "(minutes; run before a sweep)")
    p.add_argument("--demo-n", type=int, default=64,
                   help="fresh handheld captures for the printed device report")
    p.add_argument("--prior-strengths", type=float, nargs="+", default=[0.0, 32.0])
    p.add_argument("--probe-C", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--min-probe-pos", type=int, default=10,
                   help="positive training images needed to fit a probe at all")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--sens-tolerance", type=float, default=0.10)
    p.add_argument("--gate-margin", type=float, default=0.02)
    p.add_argument("--gate-feature-sets", nargs="+", default=list(GATE_FEATURE_SETS),
                   choices=list(FEATURE_SETS))
    p.add_argument("--min-test-pos-patients", type=int, default=20)
    p.add_argument("--cv-repeats", type=int, default=5)
    p.add_argument("--comparator-seed", type=int, default=0,
                   help="metadata comparator seed, fixed across training seeds")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="exp_retinareach/seed0")
    p.add_argument("--ckpt", default="ck_retinareach/seed0.pt")
    p.add_argument("--log-every", type=int, default=100)
    a = p.parse_args(argv)

    if os.path.realpath(a.source_root) == os.path.realpath(a.target_root):
        raise SystemExit("[fatal] --source-root and --target-root are the same directory")
    if a.calib_batch < 2:
        raise SystemExit("[fatal] --calib-batch must be >= 2 (train-mode BN needs a batch)")
    seed_everything(a.seed, deterministic=not a.nondeterministic)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    t0 = time.time()

    # ---- data ------------------------------------------------------------ #
    from brset_dataset import load_any
    tables, splits, img_dirs, audits = {}, {}, {}, {}
    for name, root, ds in ((a.source_name, a.source_root, a.source_dataset),
                           (a.target_name, a.target_root, a.target_dataset)):
        df, csv = load_table(root, ds)
        img_dirs[name] = load_any(root, ds)["images_dir"]
        tables[name] = df
        splits[name] = multitask_split(df, a.source_targets[0], seed=a.split_seed)
        audits[name] = {t: encoding_audit(df, t) for t in a.source_targets + a.probe_targets}
        print(f"[data] {name:<9} {ds} @ {csv}: {len(df)} rows / {df['patient'].nunique()} patients; "
              + " ".join(f"{k}={len(v)}" for k, v in splits[name].items()))
        for t, au in audits[name].items():
            if not au["ok"]:
                print(f"         {t:<22} not usable on {name}: {au['reason']}")
        pd.concat([v.assign(split=k)[["file", "patient", "split"]] for k, v in splits[name].items()]
                  ).to_csv(os.path.join(a.out, f"split_{name}.csv"), index=False)
    S, T = a.source_name, a.target_name
    bad_src = [t for t in a.source_targets if not audits[S][t]["ok"]]
    if bad_src:
        raise SystemExit(f"[fatal] source targets {bad_src} are not usable on {S}: "
                         f"{[audits[S][t]['reason'] for t in bad_src]}")
    probes = [t for t in a.probe_targets if audits[T][t]["ok"]]
    dropped = sorted(set(a.probe_targets) - set(probes))
    if dropped:
        print(f"[warn] probe targets not usable on {T} (NOT_VALIDATED everywhere): {dropped}")
    targets = list(a.source_targets) + list(a.probe_targets)

    gpu_aug = a.gpu_aug if a.gpu_aug is not None else dev.type == "cuda"
    mk = lambda name, part, split: MultiTargetDataset(
        splits[name][part], img_dirs[name], targets, split, a.image_size,
        device_aug=gpu_aug and split == "train")
    src_train, src_val, src_test = mk(S, "train", "train"), mk(S, "val", "val"), mk(S, "test", "val")
    tgt_train, tgt_val, tgt_test = mk(T, "train", "val"), mk(T, "val", "val"), mk(T, "test", "val")
    src_pool = torch.utils.data.ConcatDataset([mk(S, "train", "val"), src_val])
    tgt_pool = torch.utils.data.ConcatDataset([tgt_train, tgt_val])
    for nm, d in (("source train", src_train), ("source pool", src_pool), ("target pool", tgt_pool),
                  ("source test", src_test), ("target test", tgt_test)):
        if len(d) == 0:
            raise SystemExit(f"[fatal] {nm} has no images on disk (check the images dir)")
    print(f"[data] images on disk: {S} train/val/test {len(src_train)}/{len(src_val)}/{len(src_test)}, "
          f"{T} pool/test {len(tgt_pool)}/{len(tgt_test)}")

    g = torch.Generator(); g.manual_seed(a.seed)
    dl = lambda d, **kw: DataLoader(d, batch_size=kw.pop("bs", a.batch_size),
                                    num_workers=a.num_workers, worker_init_fn=seed_worker,
                                    pin_memory=dev.type == "cuda", **kw)
    if a.shuffle_source_labels:
        nS_ = len(a.source_targets)
        perm = np.random.default_rng(a.seed + 777).permutation(len(src_train))
        src_train.labels[:, :nS_] = src_train.labels[perm, :nS_].clone()
        print("[control] source training labels PERMUTED across images: every source-target "
              "number from this run should sit at chance")
    y_src_train = src_train.label_array()
    sampler = (WeightedRandomSampler(balanced_weights(y_src_train[:, 0]), len(src_train),
                                     replacement=True, generator=g)
               if a.imbalance == "sampler" else None)
    train_loader = dl(src_train, sampler=sampler, shuffle=sampler is None, drop_last=True,
                      generator=g, persistent_workers=a.num_workers > 0)
    loaders = {(S, "val"): dl(src_val), (S, "test"): dl(src_test),
               (T, "train"): dl(tgt_train), (T, "val"): dl(tgt_val), (T, "test"): dl(tgt_test)}

    def calib_loader(ds, seed):
        gg = torch.Generator(); gg.manual_seed(seed)
        return DataLoader(ds, batch_size=a.calib_batch, shuffle=True, generator=gg,
                          num_workers=a.num_workers, worker_init_fn=seed_worker)

    # ---- stage 1: trunk + source heads on the tabletop source ------------- #
    model = RetinaReachNet(a.source_targets, a.probe_targets, backbone=a.backbone,
                           pretrained=not a.no_pretrained, dropout=a.dropout).to(dev)
    nS = len(a.source_targets)
    print(f"[model] {a.backbone}: {sum(q.numel() for q in model.parameters()) / 1e6:.3f} M params, "
          f"embedding {model.feat_dim}, heads {model.targets}")
    if a.preflight:
        return run_preflight(a, model, dev, train_loader, loaders,
                             {S: {"pool": src_pool, "test": src_test, "val": src_val},
                              T: {"pool": tgt_pool, "test": tgt_test, "val": tgt_val}},
                             audits, targets, nS)
    ema = ModelEMA(model, a.ema_decay) if a.ema_decay > 0 else None
    use_amp = a.amp if a.amp is not None else dev.type == "cuda"
    amp_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda")
    dev_aug = DeviceAug() if gpu_aug else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    warm = max(0, min(a.warmup_epochs, a.epochs - 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - warm))
    sched = (torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm), cos],
        milestones=[warm]) if warm > 0 else cos)
    y_src_val = src_val.label_array()[:, :nS]

    def val_score(m) -> float:
        pv = predict(m, loaders[(S, "val")], dev)[:, :nS]
        aucs = [_auc(y[~np.isnan(y)].astype(int), q[~np.isnan(y)])
                for y, q in zip(y_src_val.T, pv.T) if len(np.unique(y[~np.isnan(y)])) == 2]
        aucs = [x for x in aucs if x == x]
        return float(np.mean(aucs)) if aucs else float("nan")

    cuda = dev.type == "cuda"
    best, best_ep, since, best_state = -1.0, -1, 0, None
    print(f"\n=== stage 1: {S} trunk + heads {a.source_targets}, up to {a.epochs} epochs ===")
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb, te = 0.0, 0, time.time()
        for b in train_loader:
            x = b["image"].to(dev, non_blocking=cuda)
            # the label slice is strided: never a non-blocking copy off CUDA (see predict)
            y = b["label"][:, :nS].contiguous().to(dev, non_blocking=cuda)
            if dev_aug is not None:
                x = dev_aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)[:, :nS]
            mask = ~torch.isnan(y)
            bce = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y),
                                                     reduction="none")
            loss = (bce * mask).sum() / mask.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            run = run + loss.detach(); nb += 1
            if a.log_every and nb % a.log_every == 0:
                print(f"    step {nb}/{len(train_loader)} loss={float(run) / nb:.4f}", flush=True)
        em = ema.ema if ema is not None else model
        v = val_score(em)
        tag = ""
        if v == v and v > best:
            best, best_ep, since = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in em.state_dict().items()}
            tag = "  <- best"
        else:
            since += 1
        print(f"  epoch {ep:3d}/{a.epochs} loss={float(run) / max(nb, 1):.4f} "
              f"val mean AUROC={v:.4f} [{time.time() - te:.0f}s]{tag}", flush=True)
        sched.step()
        if a.patience and since >= a.patience:
            print(f"  early stop: no val gain for {a.patience} epochs")
            break
    if best_state is None:          # val AUROC undefined every epoch: keep the last weights
        best_state = {k: t.detach().cpu().clone()
                      for k, t in (ema.ema if ema is not None else model).state_dict().items()}
        print("[warn] val AUROC was never defined; keeping the final weights")
    model.load_state_dict(best_state)
    model.eval()
    print(f"[stage1] selected epoch {best_ep} (val mean AUROC {best:.4f})")

    # ---- self-calibration: one profile per device, from its pool ---------- #
    trained_stats = get_bn_stats(model)
    cal_src, info_s = self_calibrate(model, calib_loader(src_pool, a.seed), dev)
    cal_tgt, info_t = self_calibrate(model, calib_loader(tgt_pool, a.seed), dev)
    stats = {S: get_bn_stats(cal_src), T: get_bn_stats(cal_tgt)}
    del cal_src, cal_tgt
    print(f"[calib] {S}: {info_s.n_images} imgs / {info_s.n_batches} batches; "
          f"{T}: {info_t.n_images} imgs / {info_t.n_batches} batches; {info_t.n_bn_layers} BN layers")
    print(f"[calib] camera distance trained->{S} {camera_distance(trained_stats, stats[S]):.4f}, "
          f"trained->{T} {camera_distance(trained_stats, stats[T]):.4f}, "
          f"{S}<->{T} {camera_distance(stats[S], stats[T]):.4f}")

    # ---- stage 2: systemic probes on the calibrated handheld embedding ---- #
    m_t = with_stats(model, stats[T]).to(dev)
    Z = {part: predict(m_t, loaders[(T, part)], dev, embed=True) for part in ("train", "val", "test")}
    Y = {"train": tgt_train.label_array(), "val": tgt_val.label_array(),
         "test": tgt_test.label_array()}
    probe_info = {}
    print(f"\n=== stage 2: linear probes on the calibrated {T} embedding ===")
    for t in probes:
        k = model.index(t)
        tr, va = ~np.isnan(Y["train"][:, k]), ~np.isnan(Y["val"][:, k])
        ytr, yva = Y["train"][tr, k].astype(int), Y["val"][va, k].astype(int)
        if ytr.sum() < a.min_probe_pos or len(np.unique(ytr)) < 2:
            probe_info[t] = {"fitted": False, "reason": f"{int(ytr.sum())} positive training images "
                                                        f"< {a.min_probe_pos}"}
            print(f"  {t:<22} not fitted: {probe_info[t]['reason']}")
            continue
        fit = fit_probe(Z["train"][tr], ytr, Z["val"][va], yva, a.probe_C, a.seed)
        model.set_probe(t, fit["weight"], fit["bias"])
        probe_info[t] = {"fitted": True, **{k2: v for k2, v in fit.items() if k2 not in ("weight", "bias")}}
        te = ~np.isnan(Y["test"][:, k])
        probe_info[t]["shuffle"] = probe_shuffle(Z["train"][tr], ytr, Z["test"][te],
                                                 Y["test"][te, k].astype(int), fit["C"],
                                                 a.n_shuffle, a.seed)
        print(f"  {t:<22} C={fit['C']:<6g} val AUROC {fit['val_auroc']:.3f} "
              f"(train pos {fit['pos_train']}/{fit['n_train']})")
    del m_t
    fitted = set(a.source_targets) | {t for t, i in probe_info.items() if i["fitted"]}

    # ---- predictions under each protocol, thresholds on home val ---------- #
    preds, thresholds = {}, {"trained": {}, "calibrated": {}}
    for proto in ("trained", "calibrated"):
        for name in (S, T):
            m = model if proto == "trained" else with_stats(model, stats[name]).to(dev)
            for part in ("val", "test"):
                if (name, part) not in loaders:
                    continue
                q = predict(m, loaders[(name, part)], dev)
                if proto == "trained":
                    q[:, nS:] = np.nan                  # probes exist only calibrated
                preds[(proto, name, part)] = q
    for proto in ("trained", "calibrated"):
        for t in targets:
            k = model.index(t)
            home = S if t in a.source_targets else T
            if t not in fitted or (proto == "trained" and t not in a.source_targets):
                continue
            y = (src_val if home == S else tgt_val).label_array()[:, k]
            q = preds[(proto, home, "val")][:, k]
            ok = ~np.isnan(y)
            thresholds[proto][t] = threshold_at_sensitivity(y[ok].astype(int), q[ok], a.target_sens)

    for proto in ("trained", "calibrated"):
        for name, ds in ((S, src_test), (T, tgt_test)):
            q = preds[(proto, name, "test")]
            out = pd.DataFrame({"file": ds.df["file"], "patient": ds.patients})
            for t in targets:
                out[f"p_{t}"] = q[:, model.index(t)]
                out[f"threshold_{t}"] = thresholds[proto].get(t, np.nan)
            out.to_csv(os.path.join(a.out, f"predictions_{name}_{proto}.csv"), index=False)

    # ---- capability gate ---------------------------------------------------- #
    print(f"\n=== capability gate (margin max({a.gate_margin}, comparator CV SD); "
          f"sensitivity floor {a.target_sens - a.sens_tolerance:.2f}) ===")
    def gate_device(name, fit_df, test_df, au_map, scores, shuffle=None):
        """Capability rows (+ quality strata) for one device; ``scores`` maps
        protocol -> [N, K] test probabilities aligned with ``test_df``."""
        rows, strata = [], []
        for t in targets:
            au = au_map[t]
            # Comparator seed fixed across training seeds: the bar is a property of
            # the data and split, so per-seed image-minus-metadata deltas stay paired.
            comp = (metadata_comparator(fit_df, test_df, t, a.gate_feature_sets,
                                        seed=a.comparator_seed, cv_repeats=a.cv_repeats)
                    if au["ok"] else {"reason": "n/a"})
            full = (full_chart_auroc(fit_df, test_df, t, a.comparator_seed) if au["ok"]
                    else float("nan"))
            qonly = (quality_only_auroc(fit_df, test_df, t, a.comparator_seed) if au["ok"]
                     else float("nan"))
            for proto in ("trained", "calibrated"):
                k = model.index(t)
                have = t in fitted and not (proto == "trained" and t not in a.source_targets)
                sc = scores[proto][:, k] if have else np.full(len(test_df), np.nan)
                audit = au if have else {"ok": False, "reason": (
                    "probe exists only under self-calibration" if t in fitted else
                    probe_info.get(t, {}).get("reason", au.get("reason") or "not fitted"))}
                thr = thresholds[proto].get(t, float("nan"))
                rows.append(gate_cell(
                    t, name, proto, test_df, sc, thr, comp,
                    target_sens=a.target_sens, sens_tolerance=a.sens_tolerance,
                    gate_margin=a.gate_margin, min_pos_patients=a.min_test_pos_patients,
                    n_boot=a.n_boot, seed=a.seed, audit=audit, full_chart=full,
                    quality_only=qonly,
                    shuffle=(shuffle or {}).get(t) if proto == "calibrated" else None))
                if have and au["ok"]:
                    strata += [dict(r, device=name, protocol=proto)
                               for r in quality_strata(test_df, t, sc, thr)]
        return rows, strata

    cap_rows, strata_rows = [], []
    probe_shuffles = {t: i["shuffle"] for t, i in probe_info.items() if i.get("fitted")}
    for name, ds in ((S, src_test), (T, tgt_test)):
        fit_df = pd.concat([splits[name]["train"], splits[name]["val"]], ignore_index=True)
        r, st_ = gate_device(name, fit_df, ds.df, audits[name],
                             {p_: preds[(p_, name, "test")] for p_ in ("trained", "calibrated")},
                             shuffle=probe_shuffles if name == T else None)
        cap_rows += r
        strata_rows += st_
    pd.DataFrame(cap_rows).to_csv(os.path.join(a.out, "capability.csv"), index=False)
    print(format_table(cap_rows))
    bad = [(t, i["shuffle"]) for t, i in probe_info.items() if i.get("fitted")
           and i["shuffle"]["n"] and abs(i["shuffle"]["auroc"] - 0.5) > 3 * i["shuffle"]["se"]]
    if bad:
        print(f"[WARNING] shuffled-label probe control far from 0.5 for "
              f"{[t for t, _ in bad]}: check for leakage before trusting those rows")

    # ---- mechanism: is the camera still decodable from the embedding? ------ #
    print(f"\n=== mechanism: camera decodability from the trunk embedding ({S} vs {T} test; "
          f"patient-grouped 5-fold linear read-out AUROC) ===")
    groups = np.concatenate([[f"{S}:{q}" for q in src_test.patients],
                             [f"{T}:{q}" for q in tgt_test.patients]])
    lab = np.r_[np.zeros(len(src_test)), np.ones(len(tgt_test))]
    decode = {}
    for proto in ("trained", "calibrated"):
        ms = model if proto == "trained" else with_stats(model, stats[S]).to(dev)
        mt = model if proto == "trained" else with_stats(model, stats[T]).to(dev)
        X = np.concatenate([predict(ms, loaders[(S, "test")], dev, embed=True),
                            predict(mt, loaders[(T, "test")], dev, embed=True)])
        decode[proto] = decodability(X, lab, groups, a.seed)
        print(f"  {proto:<11} camera AUROC {decode[proto]['auroc']:.3f} "
              f"(fold SD {decode[proto]['auroc_sd']:.3f}, n={decode[proto]['n']})")

    # ---- mechanism: remove the vessels, score again ------------------------ #
    ablation_rows = []
    if a.ablation_frac > 0:
        print(f"\n=== mechanism: vessel ablation ({a.ablation_frac:.0%} of the field of view "
              f"inpainted; control = the same mask rotated 90 degrees; calibrated protocol) ===")
        for name, ds in ((S, src_test), (T, tgt_test)):
            m_cal = with_stats(model, stats[name]).to(dev)
            q = {"intact": preds[("calibrated", name, "test")]}
            for kind in ("vessels", "control"):
                q[kind] = predict(m_cal, loaders[(name, "test")], dev,
                                  transform=lambda x, k=kind: ablate_normalised(x, k, a.ablation_frac))
            del m_cal
            y_all = ds.label_array()
            for t in targets:
                k = model.index(t)
                if t not in fitted or not audits[name][t]["ok"]:
                    continue
                ok = ~np.isnan(y_all[:, k]) & ~np.isnan(q["intact"][:, k])
                y = y_all[ok, k].astype(int)
                if len(np.unique(y)) < 2:
                    continue
                g = ds.patients[ok]
                pv, pc, pi = q["vessels"][ok, k], q["control"][ok, k], q["intact"][ok, k]
                vc = paired_delta_bootstrap(y, pv, pc, g, a.n_boot, a.seed)
                iv = paired_delta_bootstrap(y, pi, pv, g, a.n_boot, a.seed)
                ablation_rows.append({
                    "target": t, "device": name, "n": int(ok.sum()), "pos": int(y.sum()),
                    "auroc_intact": _auc(y, pi), "auroc_vessels": _auc(y, pv), "auroc_control": _auc(y, pc),
                    "vessels_minus_control": vc["delta"], "vmc_lo": vc["delta_lo"], "vmc_hi": vc["delta_hi"],
                    "intact_minus_vessels": iv["delta"]})
        if ablation_rows:
            abl = pd.DataFrame(ablation_rows)
            abl.to_csv(os.path.join(a.out, "anatomy_ablation.csv"), index=False)
            print(abl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print("  vessels_minus_control < 0 with its CI below 0: the target leans on the vasculature "
                  "more than on the same amount of other retina.")

    # ---- calibration-size sweep + device envelopes ------------------------ #
    print(f"\n=== calibration-size sweep on {T} (sizes {a.calib_sizes}, {a.calib_repeats} "
          f"repeats, priors {a.prior_strengths}) ===")
    sweep = []
    y_tt = tgt_test.label_array()

    def score_row(q, n, r, prior, kind):
        row = {"n": n, "repeat": r, "prior_strength": prior, "kind": kind}
        for t in targets:
            k = model.index(t)
            thr = thresholds["calibrated" if kind != "trained" else "trained"].get(t)
            ok = ~np.isnan(y_tt[:, k]) & ~np.isnan(q[:, k])
            if thr is None or ok.sum() == 0 or len(np.unique(y_tt[ok, k])) < 2:
                continue
            yy, qq = y_tt[ok, k].astype(int), q[ok, k]
            flag = qq >= thr
            row[f"auroc_{t}"] = _auc(yy, qq)
            row[f"sens_{t}"] = float(flag[yy == 1].mean())
            row[f"flagged_{t}"] = float(flag.mean())
        return row

    sweep.append(score_row(preds[("trained", T, "test")], 0, 0, 0.0, "trained"))
    sweep.append(score_row(preds[("calibrated", T, "test")], len(tgt_pool), 0, 0.0, "full_pool"))
    rng = np.random.default_rng(a.seed)

    def subset_calibration(ds, n, seed, prior=0.0, idx=None):
        idx = rng.choice(len(ds), size=n, replace=False) if idx is None else idx
        cal, info = self_calibrate(model, calib_loader(Subset(ds, idx), seed), dev,
                                   prior_strength=prior)
        return cal, get_bn_stats(cal), info, idx

    # (a) How many captures: calibrate on N handheld POOL images (never test
    #     images), score the handheld test split at the shipped thresholds. Every
    #     prior is applied from the same unblended estimate ``st``.
    for n in a.calib_sizes:
        if n > len(tgt_pool):
            continue
        for r in range(a.calib_repeats):
            # the camera distance is always of the plain (prior-free) measurement
            cal0, st, info0, idx = subset_calibration(tgt_pool, n, a.seed + r)
            for prior in a.prior_strengths:
                if prior > 0:                        # same captures, prior applied in-forward
                    cal, _, info, _ = subset_calibration(tgt_pool, n, a.seed + r, prior, idx)
                else:
                    cal, info = cal0, info0
                row = score_row(predict(cal, loaders[(T, "test")], dev), n, r, prior, "subset")
                row.update(dist_own=camera_distance(st, stats[T]),
                           dist_other=camera_distance(st, stats[S]), n_used=info.n_images)
                sweep.append(row)
            del cal0, cal
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(a.out, "calibration_sweep.csv"), index=False)
    if len(sweep_df):
        show = [c for c in sweep_df.columns if c.startswith(("auroc_", "sens_"))
                and c.split("_", 1)[1] in a.source_targets]
        print(sweep_df.groupby(["kind", "prior_strength", "n"])[show].agg(["mean", "std"])
              .to_string(float_format=lambda v: f"{v:.3f}"))

    # (b) Envelopes: the 95th percentile of the same-camera distance of an
    #     N-capture calibration drawn from the profile's own pool
    #     (--envelope-repeats draws; unblended statistics, as the phone compares).
    envelope: Dict[str, Dict[int, float]] = {S: {}, T: {}}
    for name, pool in ((S, src_pool), (T, tgt_pool)):
        for n in a.calib_sizes:
            if n > len(pool):
                continue
            d = []
            for r in range(a.envelope_repeats):
                cal, st, _, _ = subset_calibration(pool, n, a.seed + 1000 + r)
                d.append(camera_distance(st, stats[name]))
                del cal
            envelope[name][n] = float(np.percentile(d, 95))

    # (c) Out-of-sample device recognition: N-capture calibrations drawn from each
    #     device's TEST images -- patients no profile, envelope or threshold ever
    #     saw; labels unread. recognised = nearest profile is the right camera AND
    #     inside its envelope; false_accept = matched to and accepted by the OTHER
    #     camera's profile (the phone would apply the wrong capability table).
    recog: Dict[str, Dict[int, float]] = {S: {}, T: {}}
    false_accept: Dict[str, Dict[int, float]] = {S: {}, T: {}}
    for name, ds in ((S, src_test), (T, tgt_test)):
        other = T if name == S else S
        for n in a.calib_sizes:
            if n > len(ds) or n not in envelope[name]:
                continue
            ok, fa = [], []
            for r in range(a.calib_repeats):
                cal, st, _, _ = subset_calibration(ds, n, a.seed + 2000 + r)
                d_own, d_oth = camera_distance(st, stats[name]), camera_distance(st, stats[other])
                ok.append(d_own < d_oth and d_own <= envelope[name][n])
                fa.append(d_oth < d_own and d_oth <= envelope[other].get(n, -np.inf))
                del cal
            recog[name][n], false_accept[name][n] = float(np.mean(ok)), float(np.mean(fa))
    fmt = lambda dd, f: "; ".join(f"{nm}: " + ", ".join(f"{n}:{v:{f}}" for n, v in e.items())
                                  for nm, e in dd.items())
    print(f"\n[envelope] 95th-pct same-camera distance by N ({a.envelope_repeats} pool draws): "
          f"{fmt(envelope, '.3g')}")
    print(f"[envelope] out-of-sample recognition by N (test-patient draws): {fmt(recog, '.2f')}")
    print(f"[envelope] false accept by the other camera's profile: {fmt(false_accept, '.2f')}")

    # ---- profiles, device report, checkpoint -------------------------------- #
    profiles = {nm: DeviceProfile(nm, ds_, stats[nm], len(pool), envelope.get(nm, {}),
                                  [r for r in cap_rows if r["device"] == nm])
                for nm, ds_, pool in ((S, a.source_dataset, src_pool),
                                      (T, a.target_dataset, tgt_pool))}
    # What the phone prints after N FRESH captures (test patients, labels unread),
    # not the profile compared with itself.
    demo_n = max(2, min(a.demo_n, len(tgt_test)))
    cal, demo_stats, _, _ = subset_calibration(tgt_test, demo_n, a.seed + 3000)
    del cal
    report = device_capability(list(profiles.values()), demo_stats, demo_n, targets)
    print(f"\n=== what the phone reports after {demo_n} fresh {T} captures ===")
    print(f"  nearest profile {report['nearest_profile']} (distance "
          f"{report['distances'][report['nearest_profile']]:.3g}, envelope {report['radius']:.3g}, "
          f"inside: {report['inside_envelope']}); shown: {report['shown'] or 'none'}")
    for r in report["targets"]:
        print(f"  {r['target']:<22} {r['status']:<18} {r['reason']}")

    # ---- unfamiliar cameras: was the phone's label-free report right? ------ #
    # A camera with labels the model was never validated on. Its images build no
    # profile, set no threshold and pick no comparator for the model; its labels
    # are used only to score. Two questions, paired on the same test patients:
    #   observed  -- the full gate on this camera (as a validation study would run it)
    #   lookup    -- what the phone would SHOW after N unlabelled captures
    #                (device_capability against the validated profiles)
    # An "unsafe" report = the phone shows a target the camera's own labels do not
    # support; a "missed" one = it hides a target they do.
    unf_rows, lookup_rows, unf_info = [], [], {}
    for spec in a.unfamiliar:
        root_u, ds_u, name_u = parse_unfamiliar(spec)
        if name_u in (S, T):
            raise SystemExit(f"[fatal] --unfamiliar name {name_u!r} collides with a validated device")
        df_u, csv_u = load_table(root_u, ds_u)
        split_u = multitask_split(df_u, a.source_targets[0], seed=a.split_seed)
        au_u = {t: encoding_audit(df_u, t) for t in targets}
        dir_u = load_any(root_u, ds_u)["images_dir"]
        mku = lambda part: MultiTargetDataset(split_u[part], dir_u, targets, "val", a.image_size)
        test_u = mku("test")
        pool_u = torch.utils.data.ConcatDataset([mku("train"), mku("val")])
        print(f"\n=== unfamiliar camera {name_u} ({ds_u} @ {csv_u}): pool {len(pool_u)}, "
              f"test {len(test_u)} images; labelled targets: "
              f"{[t for t in targets if au_u[t]['ok']] or 'none'} ===")
        loader_u = dl(test_u)
        cal_u, info_u = self_calibrate(model, calib_loader(pool_u, a.seed), dev)
        stats_u = get_bn_stats(cal_u)
        q_tr = predict(model, loader_u, dev)
        q_tr[:, nS:] = np.nan
        scores_u = {"trained": q_tr, "calibrated": predict(cal_u, loader_u, dev)}
        del cal_u
        fit_u = pd.concat([split_u["train"], split_u["val"]], ignore_index=True)
        rows_u, strata_u = gate_device(name_u, fit_u, test_u.df, au_u, scores_u)
        unf_rows += rows_u
        strata_rows += strata_u
        observed = {r["target"]: r["status"] for r in rows_u if r["protocol"] == "calibrated"}
        for n in a.calib_sizes:
            if n > len(pool_u):
                continue
            reps = []
            for r in range(a.calib_repeats):
                cal, st, _, _ = subset_calibration(pool_u, n, a.seed + 4000 + r)
                del cal
                reps.append(device_capability(list(profiles.values()), st, n, targets))
            nearest = pd.Series([x["nearest_profile"] for x in reps]).value_counts().to_dict()
            inside = float(np.mean([x["inside_envelope"] for x in reps]))
            for t in targets:
                shown = float(np.mean([t in x["shown"] for x in reps]))
                ok_obs = observed[t] == "SUPPORTED"
                lookup_rows.append({"device": name_u, "n": n, "target": t, "observed": observed[t],
                                    "shown_rate": shown, "inside_rate": inside,
                                    "nearest": json.dumps(nearest),
                                    "unsafe_rate": 0.0 if ok_obs else shown,
                                    "missed_rate": (1.0 - shown) if ok_obs else 0.0})
        unf_info[name_u] = {"dataset": ds_u, "csv": csv_u, "n_pool": len(pool_u),
                            "n_test": len(test_u), "calibration": vars(info_u),
                            "distance_to_profiles": {nm: camera_distance(stats_u, pr.bn_stats)
                                                     for nm, pr in profiles.items()},
                            "observed": observed}
        print(format_table([r for r in rows_u if au_u[r["target"]]["ok"]]))
        lk = pd.DataFrame([r for r in lookup_rows if r["device"] == name_u])
        if len(lk):
            print(f"  camera distance (full pool) to profiles: "
                  + ", ".join(f"{k} {v:.3g}" for k, v in unf_info[name_u]["distance_to_profiles"].items()))
            print("  phone lookup vs this camera's own labels (shown rate over "
                  f"{a.calib_repeats} draws of N captures):")
            print(lk.pivot_table(index="target", columns="n", values="shown_rate")
                  .assign(observed=pd.Series(observed)).to_string(float_format=lambda v: f"{v:.2f}"))
    if unf_rows:
        pd.DataFrame(unf_rows).to_csv(os.path.join(a.out, "capability_unfamiliar.csv"), index=False)
        pd.DataFrame(lookup_rows).to_csv(os.path.join(a.out, "unfamiliar_lookup.csv"), index=False)
    pd.DataFrame(strata_rows).to_csv(os.path.join(a.out, "quality_strata.csv"), index=False)

    profiles_json = {"targets": targets, "source_targets": list(a.source_targets),
                     "probe_targets": list(a.probe_targets), "thresholds": thresholds,
                     "target_sens": a.target_sens, "calib_batch": a.calib_batch,
                     "trained_stats": stats_to_lists(trained_stats),
                     "profiles": {k: v.to_dict() for k, v in profiles.items()}}
    with open(os.path.join(a.out, "profiles.json"), "w") as f:
        json.dump(profiles_json, f, default=float)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {**model.config(), "image_size": a.image_size},
                "thresholds": thresholds, "profiles": profiles_json["profiles"],
                "trained_stats": profiles_json["trained_stats"], "probe_info": probe_info,
                "capability": cap_rows, "args": vars(a), "best_epoch": best_ep,
                "best_val": best}, a.ckpt)
    results = {"args": vars(a), "best_epoch": best_ep, "best_val_mean_auroc": best,
               "targets": targets, "probe_info": probe_info, "audits": audits,
               "thresholds": thresholds, "capability": cap_rows, "envelope": envelope,
               "device_recognition": recog, "device_false_accept": false_accept,
               "device_report": report, "camera_decodability": decode,
               "anatomy_ablation": ablation_rows,
               "unfamiliar": unf_info, "shuffle_source_labels": a.shuffle_source_labels,
               "prior_mode": "in_forward",   # runs without this key blended the prior post hoc
               "calibration": {S: vars(info_s), T: vars(info_t)},
               "camera_distance": {"trained_to_" + S: camera_distance(trained_stats, stats[S]),
                                   "trained_to_" + T: camera_distance(trained_stats, stats[T]),
                                   S + "_to_" + T: camera_distance(stats[S], stats[T])},
               "minutes": (time.time() - t0) / 60,
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] {a.out}/ (results.json, capability.csv, calibration_sweep.csv, profiles.json) "
          f"and {a.ckpt} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
