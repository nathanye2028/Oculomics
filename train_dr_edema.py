#!/usr/bin/env python3
"""
train_dr_edema.py
=================
Four-way fundus classifier: **none / DR only / macular edema only / both**.

    python train_dr_edema.py --root <BRSET> --dataset brset --seed 0 \\
        --external-root <mBRSET> --external-dataset mbrset \\
        --out exp_dr_edema/seed0 --ckpt ck_dr_edema/seed0.pt

The network
-----------
One MobileNetV4-Conv-Small trunk (``retinareach.RetinaReachNet``, 1,280-d pooled
embedding) and two sigmoid read-outs, ``DR`` and ``edema``. The four classes are
the four combinations of the two decisions:

    class = [p_DR >= t_DR] + 2 * [p_edema >= t_edema]     0 none, 1 DR only,
                                                          2 edema only, 3 both

Two heads rather than one 4-way softmax because "edema only" is nearly empty
(18 of 4,884 labelled mBRSET images): a softmax would have to learn that class
from a handful of examples, while each sigmoid head sees every positive of its
own finding. A row missing one of the two labels still trains the other head
(masked BCE); only rows with both labels are scored as a four-way example.

``--dr-def any`` (default) calls DR present at ICDR >= 1; ``referable`` at >= 2.

Pipeline
--------
1. Patient-grouped 70/10/20 split, stratified on the four-way class
   (``--split-seed`` 42, decoupled from the training seed).
2. Training: masked BCE, sampler weighted by class frequency ** -``--sampler-power``
   (0.5: rare classes are seen more often without being repeated to death),
   AdamW, warmup-cosine, optional EMA; selection on mean validation AUROC.
3. Thresholds fixed on validation: ``--threshold youden`` (max sensitivity +
   specificity - 1) or ``sens`` (highest threshold reaching ``--target-sens``).
4. Test: per-head AUROC and operating point, four-way confusion matrix,
   accuracy, balanced accuracy, macro-F1. The same on ``--external-root`` (every
   row, thresholds unchanged) when given.

Outputs (``--out``): ``results.json``, ``confusion_<set>.csv``,
``predictions_<set>.csv``, ``split.csv``. ``--ckpt`` holds the weights, config
and thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capability_gate import label_matrix  # noqa: E402
from dataset import DeviceAug  # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metadata_model import _auc, load_table, operating_point, threshold_at_sensitivity  # noqa: E402
from retinareach import DEFAULT_BACKBONE, RetinaReachNet  # noqa: E402
from train_mbrset import ModelEMA, pick_device  # noqa: E402
from train_retinareach import MultiTargetDataset, predict  # noqa: E402

CLASSES = ("none", "dr_only", "edema_only", "both")
DR_TASK = {"any": "dr_binary", "referable": "dr_referable"}


# --------------------------------------------------------------------------- #
# Labels, split, sampler
# --------------------------------------------------------------------------- #
def joint_class(y: np.ndarray) -> np.ndarray:
    """[N, 2] (DR, edema) labels -> four-way class 0..3; -1 where either is missing."""
    c = y[:, 0] + 2 * y[:, 1]
    return np.where(np.isnan(c), -1, c).astype(int)


def joint_split(df: pd.DataFrame, cls: np.ndarray, seed: int = 42, val_frac: float = 0.10,
                test_frac: float = 0.20) -> Dict[str, pd.DataFrame]:
    """``capability_gate.multitask_split`` stratified on the four-way class."""
    from sklearn.model_selection import StratifiedGroupKFold
    if "patient" not in df.columns:
        raise ValueError("joint_split: no 'patient' column; refusing a per-image split")
    df = df.reset_index(drop=True)
    groups = df["patient"].to_numpy()
    k = max(2, int(round(1.0 / test_frac)))
    tv, te = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(df, cls, groups))
    k_v = max(2, int(round(1.0 / (val_frac / (1.0 - test_frac)))))
    tr, va = next(StratifiedGroupKFold(k_v, shuffle=True, random_state=seed)
                  .split(df.iloc[tv], cls[tv], groups[tv]))
    out = {"train": df.iloc[tv].iloc[tr], "val": df.iloc[tv].iloc[va], "test": df.iloc[te]}
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def joint_weights(cls: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Per-row sampler weights, class count ** -power; rows missing a label weigh
    like the most common class."""
    counts = np.bincount(cls[cls >= 0], minlength=len(CLASSES)).astype(float)
    w_cls = np.where(counts > 0, np.maximum(counts, 1.0) ** -power, 0.0)
    w = np.where(cls >= 0, w_cls[np.clip(cls, 0, None)], w_cls[counts.argmax()])
    return torch.as_tensor(w, dtype=torch.double)


# --------------------------------------------------------------------------- #
# Thresholds and metrics
# --------------------------------------------------------------------------- #
def pick_threshold(y: np.ndarray, p: np.ndarray, mode: str, target_sens: float) -> float:
    ok = ~np.isnan(y)
    y, p = y[ok].astype(int), p[ok]
    if len(np.unique(y)) < 2:
        return 0.5
    if mode == "sens":
        return threshold_at_sensitivity(y, p, target_sens)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[1:][np.argmax((tpr - fpr)[1:])])      # thr[0] is the inf sentinel


def four_way(y: np.ndarray, p: np.ndarray, thr: Sequence[float]) -> Dict[str, object]:
    """Per-head and four-way metrics for labels ``y`` [N, 2], probabilities ``p``
    [N, 2] and the two thresholds."""
    heads = {}
    for i, name in enumerate(("dr", "edema")):
        ok = ~np.isnan(y[:, i])
        yi, pi = y[ok, i].astype(int), p[ok, i]
        heads[name] = {"n": int(ok.sum()), "positives": int(yi.sum()),
                       "auroc": _auc(yi, pi) if len(np.unique(yi)) == 2 else float("nan"),
                       **operating_point(yi, pi, float(thr[i]))}
    true = joint_class(y)
    ok = true >= 0
    true = true[ok]
    pred = ((p[ok, 0] >= thr[0]).astype(int) + 2 * (p[ok, 1] >= thr[1]).astype(int))
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (true, pred), 1)
    support, called = cm.sum(1), cm.sum(0)
    recall = np.divide(np.diag(cm), support, out=np.full(4, np.nan), where=support > 0)
    prec = np.divide(np.diag(cm), called, out=np.full(4, np.nan), where=called > 0)
    f1 = np.where(support > 0, 2 * np.diag(cm) / np.maximum(support + called, 1), np.nan)
    return {"heads": heads, "n": int(ok.sum()), "confusion": cm.tolist(),
            "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "balanced_accuracy": float(np.nanmean(recall)), "macro_f1": float(np.nanmean(f1)),
            "per_class": {c: {"support": int(support[k]), "recall": float(recall[k]),
                              "precision": float(prec[k]), "f1": float(f1[k])}
                          for k, c in enumerate(CLASSES)}}


def report(name: str, m: Dict[str, object]) -> str:
    h = m["heads"]
    lines = [f"[{name}] n={m['n']}  accuracy {m['accuracy']:.3f}  balanced accuracy "
             f"{m['balanced_accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}"]
    for k in ("dr", "edema"):
        lines.append(f"    {k:<6} AUROC {h[k]['auroc']:.3f}  sens {h[k]['sensitivity']:.3f}  "
                     f"spec {h[k]['specificity']:.3f}  ({h[k]['positives']}/{h[k]['n']} positive)")
    lines.append("    rows = truth, columns = predicted: " + " ".join(f"{c:>10}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion"]):
        lines.append(f"    {c:>34}: " + " ".join(f"{v:>10d}" for v in row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="training dataset (labels + images)")
    p.add_argument("--dataset", default="brset", choices=["brset", "mbrset"])
    p.add_argument("--external-root", default="", help="a second dataset, scored only")
    p.add_argument("--external-dataset", default="mbrset", choices=["brset", "mbrset"])
    p.add_argument("--dr-def", default="any", choices=list(DR_TASK),
                   help="DR present = ICDR >= 1 (any) or >= 2 (referable)")
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
    p.add_argument("--sampler-power", type=float, default=0.5,
                   help="class weight = count ** -power (0 = no rebalancing, 1 = fully balanced)")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--gpu-aug", dest="gpu_aug", action="store_true", default=None)
    p.add_argument("--no-gpu-aug", dest="gpu_aug", action="store_false")
    p.add_argument("--threshold", default="youden", choices=["youden", "sens"])
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--split-seed", type=int, default=42, help="patient split, fixed across seeds")
    p.add_argument("--out", default="exp_dr_edema/seed0")
    p.add_argument("--ckpt", default="ck_dr_edema/seed0.pt")
    p.add_argument("--log-every", type=int, default=100)
    a = p.parse_args(argv)

    seed_everything(a.seed, deterministic=not a.nondeterministic)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    targets = [DR_TASK[a.dr_def], "edema"]
    t0 = time.time()

    # ---- data ------------------------------------------------------------ #
    from brset_dataset import load_any
    df, csv = load_table(a.root, a.dataset)
    img_dir = load_any(a.root, a.dataset)["images_dir"]
    cls_all = joint_class(label_matrix(df, targets))
    if (cls_all >= 0).sum() == 0:
        raise SystemExit(f"[fatal] no row of {csv} has both a DR grade and an edema label")
    splits = joint_split(df, cls_all, seed=a.split_seed)
    pd.concat([v.assign(split=k)[["file", "patient", "split"]] for k, v in splits.items()]
              ).to_csv(os.path.join(a.out, "split.csv"), index=False)
    print(f"[data] {a.dataset} @ {csv}: {len(df)} rows / {df['patient'].nunique()} patients; "
          f"DR = {targets[0]}")
    for k, v in splits.items():
        c = np.bincount(joint_class(label_matrix(v, targets)) + 1, minlength=5)
        print(f"         {k:<5} {len(v):>6} rows  " + "  ".join(
            f"{n} {x}" for n, x in zip(("unlabelled",) + CLASSES, c)))

    gpu_aug = a.gpu_aug if a.gpu_aug is not None else dev.type == "cuda"
    mk = lambda part, split, d=None, imgs=None: MultiTargetDataset(
        splits[part] if d is None else d, imgs or img_dir, targets, split, a.image_size,
        device_aug=gpu_aug and split == "train")
    train_ds, val_ds, test_ds = mk("train", "train"), mk("val", "val"), mk("test", "val")
    for nm, d in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if len(d) == 0:
            raise SystemExit(f"[fatal] {nm} has no images on disk under {img_dir}")

    g = torch.Generator(); g.manual_seed(a.seed)
    dl = lambda d, **kw: DataLoader(d, batch_size=a.batch_size, num_workers=a.num_workers,
                                    worker_init_fn=seed_worker, pin_memory=dev.type == "cuda", **kw)
    y_train = train_ds.label_array()
    sampler = (WeightedRandomSampler(joint_weights(joint_class(y_train), a.sampler_power),
                                     len(train_ds), replacement=True, generator=g)
               if a.sampler_power > 0 else None)
    train_loader = dl(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True,
                      generator=g, persistent_workers=a.num_workers > 0)
    val_loader, test_loader = dl(val_ds), dl(test_ds)

    # ---- model + training -------------------------------------------------- #
    model = RetinaReachNet(targets, (), backbone=a.backbone, pretrained=not a.no_pretrained,
                           dropout=a.dropout).to(dev)
    print(f"[model] {a.backbone}: {sum(q.numel() for q in model.parameters()) / 1e6:.3f} M params, "
          f"embedding {model.feat_dim}, heads {targets} -> classes {list(CLASSES)}")
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
    y_val = val_ds.label_array()

    def val_score(m) -> float:
        pv = predict(m, val_loader, dev)
        aucs = [_auc(y[~np.isnan(y)].astype(int), q[~np.isnan(y)])
                for y, q in zip(y_val.T, pv.T) if len(np.unique(y[~np.isnan(y)])) == 2]
        aucs = [x for x in aucs if x == x]
        return float(np.mean(aucs)) if aucs else float("nan")

    cuda = dev.type == "cuda"
    best, best_ep, since, best_state = -1.0, -1, 0, None
    print(f"\n=== training, up to {a.epochs} epochs on {dev} ===")
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb, te = 0.0, 0, time.time()
        for b in train_loader:
            x = b["image"].to(dev, non_blocking=cuda)
            y = b["label"].contiguous().to(dev, non_blocking=cuda)
            if dev_aug is not None:
                x = dev_aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
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
    if best_state is None:
        best_state = {k: t.detach().cpu().clone()
                      for k, t in (ema.ema if ema is not None else model).state_dict().items()}
        print("[warn] val AUROC was never defined; keeping the final weights")
    model.load_state_dict(best_state)
    model.eval()
    print(f"[train] selected epoch {best_ep} (val mean AUROC {best:.4f})")

    # ---- thresholds on validation, then score ------------------------------ #
    p_val = predict(model, val_loader, dev)
    thr = [pick_threshold(y_val[:, i], p_val[:, i], a.threshold, a.target_sens) for i in range(2)]
    print(f"[threshold] {a.threshold}: DR {thr[0]:.4f}, edema {thr[1]:.4f} (fixed on validation)")

    sets = {"test": (test_ds, test_loader)}
    if a.external_root:
        if os.path.realpath(a.external_root) == os.path.realpath(a.root):
            raise SystemExit("[fatal] --external-root is the training dataset")
        ext_df, _ = load_table(a.external_root, a.external_dataset)
        ext_ds = mk(None, "val", d=ext_df,
                    imgs=load_any(a.external_root, a.external_dataset)["images_dir"])
        sets["external"] = (ext_ds, dl(ext_ds))
    results = {"config": vars(a), "targets": targets, "classes": list(CLASSES),
               "thresholds": {"dr": thr[0], "edema": thr[1]}, "best_epoch": best_ep,
               "val_mean_auroc": best}
    for name, (ds, loader) in sets.items():
        y, pr = ds.label_array(), predict(model, loader, dev)
        m = four_way(y, pr, thr)
        results[name] = m
        print("\n" + report(name, m))
        pd.DataFrame(m["confusion"], index=[f"true_{c}" for c in CLASSES],
                     columns=[f"pred_{c}" for c in CLASSES]
                     ).to_csv(os.path.join(a.out, f"confusion_{name}.csv"))
        pred = (pr[:, 0] >= thr[0]).astype(int) + 2 * (pr[:, 1] >= thr[1]).astype(int)
        true = joint_class(y)
        pd.DataFrame({"file": ds.df["file"], "patient": ds.patients, "p_dr": pr[:, 0],
                      "p_edema": pr[:, 1], "predicted": [CLASSES[k] for k in pred],
                      "truth": [CLASSES[k] if k >= 0 else "" for k in true]}
                     ).to_csv(os.path.join(a.out, f"predictions_{name}.csv"), index=False)

    results["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    torch.save({"model": model.state_dict(), "config": {**model.config(), "dr_def": a.dr_def,
                                                        "image_size": a.image_size},
                "classes": list(CLASSES), "thresholds": results["thresholds"]}, a.ckpt)
    print(f"\n[done] {results['minutes']:.1f} min -> {a.out}/results.json, {a.ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
