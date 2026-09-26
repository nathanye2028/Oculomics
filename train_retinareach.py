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

Outputs (``--out``): ``results.json``, ``capability.csv``,
``calibration_sweep.csv``, ``predictions_<device>_<protocol>.csv``,
``split_<device>.csv``, ``profiles.json`` (what ``export_retinareach.py``
ships). ``--ckpt`` holds the weights (as-trained statistics + probes) with the
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
from capability_gate import (GATE_FEATURE_SETS, encoding_audit, format_table,  # noqa: E402
                             full_chart_auroc, gate_cell, label_matrix, metadata_comparator,
                             multitask_split)
from dataset import DeviceAug, LabelSpec, MBRSETDataset  # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metadata_model import FEATURE_SETS, _auc, load_table, threshold_at_sensitivity  # noqa: E402
from retinareach import (DEFAULT_BACKBONE, PROBE_TARGETS, SOURCE_TARGETS,  # noqa: E402
                         DeviceProfile, RetinaReachNet, blend_stats, camera_distance,
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
        present = set(os.listdir(images_dir)) if os.path.isdir(images_dir) else set()
        df = df[df["file"].astype(str).isin(present)].reset_index(drop=True)
        super().__init__(csv=df, images_dir=images_dir, task="retinareach", split=split,
                         image_size=image_size, drop_missing_labels=False,
                         drop_missing_files=False, fov_crop=True, device_aug=device_aug)
        self.df = df
        self.patients = df["patient"].to_numpy()

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
def predict(model: nn.Module, loader, device, embed: bool = False) -> np.ndarray:
    """Sigmoid probabilities [N, K] (or pooled embeddings [N, D])."""
    model.eval()
    out = []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
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
    y_src_train = src_train.labels.numpy()
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
    y_src_val = src_val.labels.numpy()[:, :nS]

    def val_score(m) -> float:
        pv = predict(m, loaders[(S, "val")], dev)[:, :nS]
        aucs = [_auc(y[~np.isnan(y)].astype(int), q[~np.isnan(y)])
                for y, q in zip(y_src_val.T, pv.T) if len(np.unique(y[~np.isnan(y)])) == 2]
        aucs = [x for x in aucs if x == x]
        return float(np.mean(aucs)) if aucs else float("nan")

    best, best_ep, since, best_state = -1.0, -1, 0, None
    print(f"\n=== stage 1: {S} trunk + heads {a.source_targets}, up to {a.epochs} epochs ===")
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb, te = 0.0, 0, time.time()
        for b in train_loader:
            x = b["image"].to(dev, non_blocking=True)
            y = b["label"][:, :nS].to(dev, non_blocking=True)
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
    Z = {part: predict(m_t, loaders[(T, part)], dev, embed=True) for part in ("train", "val")}
    Y = {"train": tgt_train.labels.numpy(), "val": tgt_val.labels.numpy()}
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
            y = (src_val if home == S else tgt_val).labels.numpy()[:, k]
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
    cap_rows = []
    for name, ds in ((S, src_test), (T, tgt_test)):
        fit_df = pd.concat([splits[name]["train"], splits[name]["val"]], ignore_index=True)
        test_df = ds.df
        for t in targets:
            au = audits[name][t]
            # Comparator seed fixed across training seeds: the bar is a property of
            # the data and split, so per-seed image-minus-metadata deltas stay paired.
            comp = (metadata_comparator(fit_df, test_df, t, a.gate_feature_sets,
                                        seed=a.comparator_seed, cv_repeats=a.cv_repeats)
                    if au["ok"] else {"reason": "n/a"})
            full = (full_chart_auroc(fit_df, test_df, t, a.comparator_seed) if au["ok"]
                    else float("nan"))
            for proto in ("trained", "calibrated"):
                k = model.index(t)
                have = t in fitted and not (proto == "trained" and t not in a.source_targets)
                sc = preds[(proto, name, "test")][:, k] if have else np.full(len(test_df), np.nan)
                audit = au if have else {"ok": False, "reason": (
                    "probe exists only under self-calibration" if t in fitted else
                    probe_info.get(t, {}).get("reason", au.get("reason") or "not fitted"))}
                cap_rows.append(gate_cell(
                    t, name, proto, test_df, sc, thresholds[proto].get(t, float("nan")), comp,
                    target_sens=a.target_sens, sens_tolerance=a.sens_tolerance,
                    gate_margin=a.gate_margin, min_pos_patients=a.min_test_pos_patients,
                    n_boot=a.n_boot, seed=a.seed, audit=audit, full_chart=full))
    pd.DataFrame(cap_rows).to_csv(os.path.join(a.out, "capability.csv"), index=False)
    print(format_table(cap_rows))

    # ---- calibration-size sweep + device envelopes ------------------------ #
    print(f"\n=== calibration-size sweep on {T} (sizes {a.calib_sizes}, {a.calib_repeats} "
          f"repeats, priors {a.prior_strengths}) ===")
    sweep = []
    y_tt = tgt_test.labels.numpy()

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

    def subset_calibration(ds, n, seed):
        idx = rng.choice(len(ds), size=n, replace=False)
        cal, info = self_calibrate(model, calib_loader(Subset(ds, idx), seed), dev)
        return cal, get_bn_stats(cal), info

    # (a) How many captures: calibrate on N handheld POOL images (never test
    #     images), score the handheld test split at the shipped thresholds. Every
    #     prior is applied from the same unblended estimate ``st``.
    for n in a.calib_sizes:
        if n > len(tgt_pool):
            continue
        for r in range(a.calib_repeats):
            cal, st, info = subset_calibration(tgt_pool, n, a.seed + r)
            for prior in a.prior_strengths:
                set_bn_stats(cal, st if prior <= 0 else
                             blend_stats(trained_stats, st, info.n_images, prior))
                row = score_row(predict(cal, loaders[(T, "test")], dev), n, r, prior, "subset")
                row.update(dist_own=camera_distance(st, stats[T]),
                           dist_other=camera_distance(st, stats[S]), n_used=info.n_images)
                sweep.append(row)
            del cal
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
                cal, st, _ = subset_calibration(pool, n, a.seed + 1000 + r)
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
                cal, st, _ = subset_calibration(ds, n, a.seed + 2000 + r)
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
    cal, demo_stats, _ = subset_calibration(tgt_test, demo_n, a.seed + 3000)
    del cal
    report = device_capability(list(profiles.values()), demo_stats, demo_n, targets)
    print(f"\n=== what the phone reports after {demo_n} fresh {T} captures ===")
    print(f"  nearest profile {report['nearest_profile']} (distance "
          f"{report['distances'][report['nearest_profile']]:.3g}, envelope {report['radius']:.3g}, "
          f"inside: {report['inside_envelope']}); shown: {report['shown'] or 'none'}")
    for r in report["targets"]:
        print(f"  {r['target']:<22} {r['status']:<18} {r['reason']}")

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
               "device_report": report,
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
