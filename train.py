"""
train.py
========
.. deprecated:: 2026-08
   Superseded by ``train_mbrset.py`` (AUROC selection, patient-grouped
   splits, BRSET->mBRSET transfer). Kept only for reference; nothing in
   the repo invokes it.

Robust training loop for the mBRSET clinical-label **classifier**
(:class:`model.MBRSETClassifier` over :class:`dataset.MBRSETDataset`).

Includes, per spec:
  * **Cross-Entropy loss** (multiclass clinical labels; optional class weighting).
  * **Adam optimizer**.
  * **Per-epoch validation tracking** (loss, accuracy, macro-F1).
  * **Automatic checkpointing** — saves the *best* model weights (by val accuracy)
    to ``~/oculomics_project/weights/`` (plus a rolling ``last.pt``).
  * **File logging** — every epoch's metrics are written to
    ``~/oculomics_project/logs/train_metrics.log`` so you don't have to watch
    the terminal; tail it any time with ``tail -f``.

Data: by default fetches the public Kaggle DR dataset
(``mohamedabdalkader/retinal-disease-detection``) and uses its ``train`` split
for training and ``valid`` for validation, remapped onto the mBRSET schema.
Override with ``--train-csv/--val-csv/--images-dir`` for your own data.

Example
-------
    python train.py --task dr_grade --epochs 50 --batch-size 32
    tail -f ~/oculomics_project/logs/train_metrics.log
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset       # noqa: E402
from model import MBRSETClassifier      # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metrics import quadratic_weighted_kappa, CSVLogger  # noqa: E402

KAGGLE_SLUG = "mohamedabdalkader/retinal-disease-detection"
COLUMN_MAP = {
    "Image name": "file",
    "Retinopathy grade": "final_icdr",
    "Risk of macular edema": "final_edema",
}
DEFAULT_WEIGHTS = os.path.expanduser("~/oculomics_project/weights")
DEFAULT_LOG = os.path.expanduser("~/oculomics_project/logs/train_metrics.log")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(log_path: str) -> logging.Logger:
    """Logger that writes metrics to a file (and echoes to the console)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("oculomics.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)          # <- metrics persisted to disk
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()                # console echo (harmless)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def resolve_data(args):
    """Return (train_df, val_df, images_dir_train, images_dir_val)."""
    if args.train_csv and args.val_csv and args.images_dir:
        tr = pd.read_csv(args.train_csv).rename(columns=COLUMN_MAP)
        va = pd.read_csv(args.val_csv).rename(columns=COLUMN_MAP)
        return tr, va, args.images_dir, args.images_dir
    import kagglehub
    root = kagglehub.dataset_download(KAGGLE_SLUG)
    base = os.path.join(root, "Diabetic Retinopathy")
    tr_dir = os.path.join(base, "train")
    va_dir = os.path.join(base, "valid")
    tr = pd.read_csv(os.path.join(tr_dir, "annotations.csv")).rename(columns=COLUMN_MAP)
    va = pd.read_csv(os.path.join(va_dir, "annotations.csv")).rename(columns=COLUMN_MAP)
    return tr, va, os.path.join(tr_dir, "images"), os.path.join(va_dir, "images")


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    tp = torch.zeros(num_classes); fp = torch.zeros(num_classes); fn = torch.zeros(num_classes)
    all_pred, all_true = [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_pred.append(pred.cpu()); all_true.append(y.cpu())
        for c in range(num_classes):
            pc, yc = (pred == c), (y == c)
            tp[c] += (pc & yc).sum().item()
            fp[c] += (pc & ~yc).sum().item()
            fn[c] += (~pc & yc).sum().item()
    f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).mean().item()
    preds = torch.cat(all_pred).numpy() if all_pred else []
    trues = torch.cat(all_true).numpy() if all_true else []
    # Quadratic-weighted kappa: the clinical standard for ordinal DR grading.
    kappa = quadratic_weighted_kappa(preds, trues, num_classes) if len(preds) else 0.0
    return loss_sum / max(total, 1), correct / max(total, 1), f1, kappa


def main() -> int:
    p = argparse.ArgumentParser(description="Robust classifier training loop (CE + Adam + checkpointing + file logging).")
    p.add_argument("--task", default="dr_grade", help="Classification task in dataset.LABEL_REGISTRY.")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--class-weighted-loss", action="store_true",
                   help="Weight CE by inverse class frequency (in addition to balanced sampling).")
    p.add_argument("--no-sampler", action="store_true",
                   help="Disable WeightedRandomSampler (use plain shuffling).")
    p.add_argument("--weights-dir", default=DEFAULT_WEIGHTS)
    p.add_argument("--log-file", default=DEFAULT_LOG)
    p.add_argument("--max-train-batches", type=int, default=0, help="Cap train batches/epoch (0=all; smoke runs).")
    p.add_argument("--max-val-batches", type=int, default=0, help="Cap val batches (0=all).")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    os.makedirs(args.weights_dir, exist_ok=True)
    logger = setup_logger(args.log_file)
    csv_log = CSVLogger(os.path.join(os.path.dirname(args.log_file), "train_metrics.csv"))
    best_path = os.path.join(args.weights_dir, "best_model.pt")
    last_path = os.path.join(args.weights_dir, "last.pt")

    # --- data ---
    train_df, val_df, tr_images, va_images = resolve_data(args)
    train_ds = MBRSETDataset(csv=train_df, images_dir=tr_images, task=args.task,
                             split="train", image_size=args.image_size, drop_missing_files=True)
    val_ds = MBRSETDataset(csv=val_df, images_dir=va_images, task=args.task,
                           split="val", image_size=args.image_size, drop_missing_files=True)
    num_classes = train_ds.num_classes
    if num_classes is None or train_ds.multilabel:
        raise SystemExit("train.py uses Cross-Entropy for multiclass tasks; "
                         f"task {args.task!r} is regression/multilabel. Pick a classification task.")

    g = torch.Generator(); g.manual_seed(args.seed)
    if args.no_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, worker_init_fn=seed_worker,
                                  generator=g, pin_memory=(device.type == "cuda"))
    else:
        sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds),
                                        replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, worker_init_fn=seed_worker,
                                  generator=g, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            pin_memory=(device.type == "cuda"))

    # --- model / loss / optim ---
    model = MBRSETClassifier.from_dataset(train_ds, pretrained=not args.no_pretrained).to(device)
    ce_weight = train_ds.class_weights().to(device) if args.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(f"start: task={args.task} classes={num_classes} device={device} "
                f"train={len(train_ds)} val={len(val_ds)} epochs={args.epochs} "
                f"batch={args.batch_size} lr={args.lr} sampler={not args.no_sampler} "
                f"class_weighted_loss={args.class_weighted_loss}")
    logger.info(f"weights -> {args.weights_dir}   log -> {args.log_file}")

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for i, batch in enumerate(train_loader, 1):
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if args.max_train_batches and i >= args.max_train_batches:
                break
        train_loss, train_acc = loss_sum / max(total, 1), correct / max(total, 1)

        # cap val batches if requested (smoke runs)
        vloader = val_loader
        if args.max_val_batches:
            from itertools import islice
            vloader = list(islice(val_loader, args.max_val_batches))
        val_loss, val_acc, val_f1, val_kappa = evaluate(model, vloader, criterion, device, num_classes)

        # checkpoint: best by val accuracy
        is_best = val_acc > best_val_acc
        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                "val_loss": val_loss, "val_f1": val_f1, "val_kappa": val_kappa,
                "task": args.task, "num_classes": num_classes, "args": vars(args)}
        torch.save(ckpt, last_path)
        tag = ""
        if is_best:
            best_val_acc = val_acc
            torch.save(ckpt, best_path)
            tag = "  *BEST -> saved best_model.pt"

        logger.info(f"epoch={epoch:03d}/{args.epochs}  "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                    f"val_kappa={val_kappa:.4f}  best_val_acc={best_val_acc:.4f}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(train_loss, 5), "train_acc": round(train_acc, 5),
                     "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 5),
                     "val_f1": round(val_f1, 5), "val_kappa": round(val_kappa, 5),
                     "best_val_acc": round(best_val_acc, 5)})

    csv_log.close()
    logger.info(f"done: best_val_acc={best_val_acc:.4f}  best weights at {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
