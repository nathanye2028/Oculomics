"""
train_idrid.py
==============
Real lesion-segmentation trainer + benchmark on **IDRiD** (multi-label,
per-pixel ground-truth masks from :mod:`idrid_dataset`).

This is the honest benchmark the placeholder pipeline could not be: it trains
on genuine microaneurysm / haemorrhage / hard-exudate / soft-exudate masks and
evaluates **per-lesion Dice on the held-out test set**.

Use it to measure what Context Gating actually buys, apples-to-apples on the
*same* backbone:
    python train_idrid.py --arch gcg_unet              # with GCG
    python train_idrid.py --arch gcg_unet --no-gcg     # GCG ablation (control)
    python train_idrid.py --arch baseline              # vanilla U-Net reference

Loss: per-channel BCE (pos-weighted for the extreme lesion/background imbalance)
+ soft Dice averaged over lesion channels.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idrid_dataset import IDRiDSegDataset, DEFAULT_LESIONS  # noqa: E402

KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seg_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)                       # per-channel over batch + space
    num = 2 * (probs * target).sum(dims) + 1.0
    den = probs.sum(dims) + target.sum(dims) + 1.0
    dice = 1.0 - (num / den).mean()
    return bce + dice


@torch.no_grad()
def accumulate_dice(logits, target, inter, denom):
    """Accumulate per-channel intersection / denominator for dataset-level Dice."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter += 2 * (pred * target).sum(dim=(0, 2, 3))
    denom += pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))


@torch.no_grad()
def evaluate(model, loader, lesions, device):
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    denom = torch.zeros(C, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        accumulate_dice(model(imgs), masks, inter, denom)
    dice = (inter / denom.clamp(min=1e-6)).cpu()
    return {code: dice[i].item() for i, code in enumerate(lesions)}


def build(args, num_classes, device):
    if args.arch == "baseline":
        from unet_baseline import build_baseline
        m = build_baseline(num_classes=num_classes, base=args.base)
        desc = f"baseline U-Net (base={args.base}, no gating)"
    else:
        from model_seg import build_model
        m = build_model(arch="gcg_unet", num_classes=num_classes,
                        pretrained=args.pretrained, use_gcg=not args.no_gcg)
        desc = f"GCG-U-Net (MobileNetV3) {'WITH' if not args.no_gcg else 'NO'} GCG"
    return m.to(device), desc


def main() -> int:
    p = argparse.ArgumentParser(description="IDRiD lesion-segmentation trainer / benchmark.")
    p.add_argument("--root", default=None, help="IDRiD download root (else fetch via kagglehub).")
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS),
                   help="Lesion channels to segment (subset of MA HE EX SE OD).")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-weight", type=float, default=8.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="Cap steps/epoch (0=all); for quick smoke runs.")
    args = p.parse_args()

    device = pick_device()
    root = args.root
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download(KAGGLE_SLUG)

    train_ds = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                               lesions=args.lesions, fov_crop=True, augment=True)
    test_ds = IDRiDSegDataset(root, split="test", image_size=args.image_size,
                              lesions=args.lesions, fov_crop=True, augment=False)
    C = train_ds.num_classes
    print(f"[info] device : {device}")
    print(f"[info] lesions: {args.lesions}  (C={C} channels)")
    print(f"[info] data   : train={len(train_ds)}  test={len(test_ds)}  @ {args.image_size}px")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model, desc = build(args, C, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model  : {desc}, {n_params/1e6:.2f}M params")

    # shape sanity on one batch
    imgs, masks = next(iter(train_loader))
    print(f"\n=== one batch ===\nimage: {tuple(imgs.shape)}  mask: {tuple(masks.shape)} "
          f"(per-lesion pos%: {[round(100*masks[:,i].mean().item(),3) for i in range(C)]})")

    pos_weight = torch.full((C, 1, 1), args.pos_weight, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n=== training {args.epochs} epochs ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for step, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            loss = seg_loss(model(imgs), masks, pos_weight)
            loss.backward(); opt.step()
            running += loss.item(); n += 1
            if args.max_steps and step >= args.max_steps:
                break
        dice = evaluate(model, test_loader, args.lesions, device)
        mean_d = sum(dice.values()) / len(dice)
        per = "  ".join(f"{k}={v:.3f}" for k, v in dice.items())
        print(f"  epoch {epoch}/{args.epochs}  train_loss={running/max(n,1):.4f}  "
              f"test_dice[mean={mean_d:.3f}]  {per}")

    print("\nOK: trained + evaluated on real IDRiD lesion masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
