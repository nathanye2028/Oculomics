"""
pretrain_rfmid.py
=================
**In-domain encoder pretraining** on RFMiD multi-disease classification.

Why: IDRiD gives only 54 masked images for segmentation. ImageNet transfer
already helped enormously (mean Dice 0.183 -> 0.387), which shows the encoder
initialisation dominates in this low-data regime. RFMiD supplies 1,920 *fundus*
images with disease labels — pretraining on those should beat ImageNet transfer,
because the features are learned on retinas rather than natural images.

Chain:
    pretrain_rfmid.py            -> encoder weights (MobileNetV3-Large)
    train_idrid.py --init-encoder <weights>   -> segmentation fine-tune

The encoder here is exactly ``model_seg.MobileNetV3Encoder``, so its state_dict
drops straight into GCG-U-Net.

Run:
    python pretrain_rfmid.py --epochs 30 --batch-size 32 --out weights/rfmid_encoder.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import MobileNetV3Encoder          # noqa: E402
from rfmid_dataset import RFMiDDataset, download_rfmid  # noqa: E402
from fundus_utils import seed_everything, seed_worker   # noqa: E402
from metrics import CSVLogger                     # noqa: E402


class RFMiDClassifier(nn.Module):
    """MobileNetV3-Large encoder + global-pool + linear multi-label head."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        feat_dim = self.encoder.out_channels[-1]        # 960 (stride-32 stage)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(feat_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep = self.encoder(x)[-1]                      # [B, 960, H/32, W/32]
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        return self.head(pooled)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    """Mean AUROC over labels with both classes present, plus BCE loss."""
    model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight).item() * y.size(0)
        n += y.size(0)
        logits_all.append(logits.cpu()); y_all.append(y.cpu())
    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score(y[:, c], logits[:, c])
                for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
        auc = float(sum(aucs) / max(len(aucs), 1))
    except Exception:  # noqa
        auc = float("nan")
    return loss_sum / max(n, 1), auc


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain the segmentation encoder on RFMiD.")
    p.add_argument("--slug", default=None, help="Kaggle slug (default: try known RFMiD mirrors).")
    p.add_argument("--root", default=None, help="Local RFMiD root (skips download).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-imagenet", action="store_true",
                   help="Start from scratch instead of ImageNet weights.")
    p.add_argument("--out", default="weights/rfmid_encoder.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    root = args.root or download_rfmid(args.slug)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train_ds = RFMiDDataset(root, "train", args.image_size, augment=True, seed=args.seed)
    val_ds = RFMiDDataset(root, "val", args.image_size, augment=False, seed=args.seed)
    print(f"[info] device : {device}")
    print(f"[info] root   : {root}")
    print(f"[info] data   : train={len(train_ds)} val={len(val_ds)}  "
          f"labels={train_ds.num_classes}"
          + (f"  (dropped {train_ds.missing} rows w/o images)" if train_ds.missing else ""))

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              generator=g, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            pin_memory=(device.type == "cuda"))

    model = RFMiDClassifier(train_ds.num_classes, pretrained=not args.no_imagenet).to(device)
    pos_weight = train_ds.pos_weight().to(device)       # heavy class imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    csv_log = CSVLogger(os.path.splitext(args.out)[0] + "_metrics.csv")

    best_auc = -1.0
    print(f"\n=== pretraining {args.epochs} epochs (val AUROC selects best) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
        val_loss, val_auc = evaluate(model, val_loader, device, pos_weight)
        tag = ""
        if val_auc > best_auc:
            best_auc = val_auc
            # Save ONLY the encoder -> drops directly into GCG-U-Net.
            torch.save({"encoder": model.encoder.state_dict(), "epoch": epoch,
                        "val_auc": val_auc, "labels": train_ds.label_cols,
                        "args": vars(args)}, args.out)
            tag = "  <- best (encoder saved)"
        print(f"  epoch {epoch:3d}/{args.epochs}  train_loss={run/max(nb,1):.4f}  "
              f"val_loss={val_loss:.4f}  val_AUROC={val_auc:.4f}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(run/max(nb,1), 5),
                     "val_loss": round(val_loss, 5), "val_auroc": round(val_auc, 5)})
        sched.step()
    csv_log.close()

    print(f"\nbest val AUROC {best_auc:.4f}  ->  encoder at {args.out}")
    print("next:  python train_idrid.py --init-encoder " + args.out +
          " --pretrained --patch-size 512 --eval-tiled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
