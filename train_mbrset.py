"""
train_mbrset.py
===============
Clinical-label classification on **mBRSET** — 5,164 smartphone (Phelcom Eyer)
fundus images with expert DR/edema grades.

Why this track matters
----------------------
The IDRiD segmentation study is capped at 54 annotated images, so a ~0.03
effect cannot be resolved: seed variance swamps it. mBRSET is ~100x larger and
is the dataset this project targets, so a GCG effect of the same size becomes
measurable here.

Design
------
* **Patient-grouped splits** (70/10/20). Both eyes of a patient never straddle
  splits — mandatory, otherwise the model memorises patients and the test score
  is inflated.
* Metrics: **AUROC** (primary; the project's stated metric), plus accuracy,
  macro-F1, and quadratic-weighted kappa for ordinal DR grade.
* GCG vs control is the *same* backbone with gating toggled (``--no-gcg``),
  exactly as in the segmentation study.

Run:
    python train_mbrset.py --root <mBRSET 1.0 dir> --task dr_referable --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset, stratified_split      # noqa: E402
from model import MBRSETClassifier                        # noqa: E402
from fundus_utils import seed_everything, seed_worker     # noqa: E402
from metrics import quadratic_weighted_kappa, CSVLogger   # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Returns dict with auroc / acc / macro-F1 / kappa on a held-out loader."""
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits_all.append(model(x).cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)

    from sklearn.metrics import roc_auc_score, f1_score
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(y, prob[:, 1]))
        else:                      # macro one-vs-rest over present classes
            present = [c for c in range(num_classes) if (y == c).sum() > 0]
            auroc = float(np.mean([roc_auc_score((y == c).astype(int), prob[:, c])
                                   for c in present]))
    except Exception:              # single-class split
        auroc = float("nan")
    return {
        "auroc": auroc,
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "kappa": quadratic_weighted_kappa(pred, y, num_classes),
        "n": int(len(y)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="mBRSET clinical-label classification.")
    p.add_argument("--root", required=True, help="mBRSET 1.0 dir (images/ + labels_mbrset.csv).")
    p.add_argument("--task", default="dr_referable",
                   choices=["dr_referable", "dr_binary", "dr_grade", "edema",
                            "quality", "artifacts"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--gcg-variant", default="baseline")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--ckpt-dir", default="ck_mbrset")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-json", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    run_name = args.run_name or f"{'nogcg' if args.no_gcg else 'gcg'}_seed{args.seed}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, f"{run_name}.pt")

    csv_path = os.path.join(args.root, "labels_mbrset.csv")
    img_dir = os.path.join(args.root, "images")

    # Patient-grouped, label-stratified 70/10/20 split (no patient leakage).
    splits = stratified_split(csv_path, task=args.task, val_frac=0.10,
                              test_frac=0.20, group_col="patient", seed=args.seed)
    mk = lambda df, sp: MBRSETDataset(csv=df, images_dir=img_dir, task=args.task,
                                      split=sp, image_size=args.image_size,
                                      drop_missing_files=True, fov_crop=True)
    train_ds, val_ds, test_ds = mk(splits["train"], "train"), mk(splits["val"], "val"), mk(splits["test"], "val")
    C = train_ds.num_classes

    print(f"[info] device : {device}")
    print(f"[info] task   : {args.task}  classes={C}  gcg={'off' if args.no_gcg else args.gcg_variant}")
    print(f"[info] splits : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (patient-grouped)")
    print(f"[info] class counts (train): {train_ds.class_counts().tolist()}")

    g = torch.Generator(); g.manual_seed(args.seed)
    sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds),
                                    replacement=True, generator=g)
    dl = lambda ds, **kw: DataLoader(ds, batch_size=args.batch_size,
                                     num_workers=args.num_workers, worker_init_fn=seed_worker,
                                     generator=g, pin_memory=(device.type == "cuda"), **kw)
    train_loader = dl(train_ds, sampler=sampler)
    val_loader, test_loader = dl(val_ds, shuffle=False), dl(test_ds, shuffle=False)

    model = MBRSETClassifier(num_classes=C, pretrained=not args.no_pretrained,
                             use_gcg=not args.no_gcg, gcg_variant=args.gcg_variant).to(device)
    print(f"[info] model  : MobileNetV3-Small, {sum(q.numel() for q in model.parameters())/1e6:.3f}M params")

    crit = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    csv_log = CSVLogger(os.path.join(args.ckpt_dir, f"{run_name}_metrics.csv"))

    best_auroc, best_epoch, since = -1.0, -1, 0
    print(f"\n=== training up to {args.epochs} epochs (val AUROC selects best) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
        vm = evaluate(model, val_loader, device, C)
        tag = ""
        if vm["auroc"] > best_auroc:
            best_auroc, best_epoch, since = vm["auroc"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": vm,
                        "args": vars(args)}, ckpt)
            tag = "  <- best"
        else:
            since += 1
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={run/max(nb,1):.4f}  "
              f"val_AUROC={vm['auroc']:.4f} acc={vm['acc']:.3f} kappa={vm['kappa']:.3f}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(run/max(nb,1), 5),
                     **{f"val_{k}": round(v, 5) for k, v in vm.items() if k != "n"}})
        sched.step()
        if args.patience and since >= args.patience:
            print(f"  early stop: no val AUROC gain for {args.patience} epochs")
            break
    csv_log.close()

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    tm = evaluate(model, test_loader, device, C)
    print(f"\n=== TEST (held-out, best checkpoint @ epoch {best_epoch}) ===")
    print(f"  AUROC={tm['auroc']:.4f}  acc={tm['acc']:.4f}  macroF1={tm['f1']:.4f}  "
          f"kappa={tm['kappa']:.4f}  n={tm['n']}")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump({"task": args.task, "use_gcg": not args.no_gcg,
                       "gcg_variant": None if args.no_gcg else args.gcg_variant,
                       "seed": args.seed, "num_classes": C,
                       "test": tm, "best_val_auroc": best_auroc, "best_epoch": best_epoch,
                       "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                       "args": vars(args)}, f, indent=2)
        print(f"[info] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
