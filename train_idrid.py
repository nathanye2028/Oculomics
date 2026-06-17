"""
train_idrid.py
==============
Real lesion-segmentation trainer + benchmark on **IDRiD** (multi-label,
per-pixel ground-truth masks from :mod:`idrid_dataset`).

Methodology (proper held-out evaluation)
----------------------------------------
IDRiD ships an official **train** (54) and **test** (27) split. We further
carve a **validation** split out of train (``--val-frac``):

    train  -> optimise weights
    val    -> per-epoch metric used for *model selection* (best checkpoint)
    test   -> scored EXACTLY ONCE at the end, on the best checkpoint

So no metric is ever read off the batch the model just trained on, and the
test set never influences training or checkpoint selection. The best model
(highest val mean-Dice) is saved to ``--ckpt-dir``.

Apples-to-apples GCG comparison
-------------------------------
The control for "does Guided Context Gating help?" is the **same backbone with
gating switched off**:

    python train_idrid.py --arch gcg_unet              # GCG on
    python train_idrid.py --arch gcg_unet --no-gcg     # GCG control (only gating differs)

``--arch baseline`` (a vanilla U-Net, :mod:`unet_baseline`) is a SEPARATE
"standard-architecture" reference point — it differs in backbone, depth and
pretraining, so it is NOT the GCG ablation. Do not read baseline-vs-gcg_unet as
the effect of gating; read gcg_unet on-vs-off for that.

Loss: per-channel BCE (pos-weighted for the extreme lesion/background
imbalance) + soft Dice averaged over lesion channels.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idrid_dataset import IDRiDSegDataset, DEFAULT_LESIONS  # noqa: E402

KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seg_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)                       # per-channel over batch + space
    num = 2 * (probs * target).sum(dims) + 1.0
    den = probs.sum(dims) + target.sum(dims) + 1.0
    dice = 1.0 - (num / den).mean()
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, lesions, device):
    """Dataset-level per-lesion Dice over an entire loader (held-out split)."""
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    denom = torch.zeros(C, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = (torch.sigmoid(model(imgs)) > 0.5).float()
        inter += 2 * (pred * masks).sum(dim=(0, 2, 3))
        denom += pred.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = (inter / denom.clamp(min=1e-6)).cpu()
    return {code: dice[i].item() for i, code in enumerate(lesions)}


def build(args, num_classes, device):
    if args.arch == "baseline":
        from unet_baseline import build_baseline
        m = build_baseline(num_classes=num_classes, base=args.base)
        desc = f"baseline U-Net (base={args.base}) [standard-architecture reference, NOT the GCG control]"
    else:
        from model_seg import build_model
        m = build_model(arch="gcg_unet", num_classes=num_classes,
                        pretrained=args.pretrained, use_gcg=not args.no_gcg)
        tag = "WITH GCG" if not args.no_gcg else "NO GCG (control)"
        desc = f"GCG-U-Net (MobileNetV3) {tag}"
    return m.to(device), desc


def default_run_name(args) -> str:
    if args.arch == "baseline":
        return f"baseline_b{args.base}"
    return "gcg_unet_" + ("gcg" if not args.no_gcg else "nogcg")


def main() -> int:
    p = argparse.ArgumentParser(description="IDRiD lesion-segmentation trainer / benchmark.")
    p.add_argument("--root", default=None, help="IDRiD download root (else fetch via kagglehub).")
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS),
                   help="Lesion channels to segment (subset of MA HE EX SE OD).")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--pos-weight", type=float, default=20.0)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of TRAIN held out for validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--no-gcg", action="store_true", help="Disable GCG (the apples-to-apples control).")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    run_name = args.run_name or default_run_name(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}.pt")

    root = args.root
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download(KAGGLE_SLUG)

    # Two views of the official train split: augmented (for training) and
    # clean (for validation). A seeded permutation splits them by index.
    full_train_aug = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                                     lesions=args.lesions, fov_crop=True, augment=True)
    full_train_clean = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                                       lesions=args.lesions, fov_crop=True, augment=False)
    test_ds = IDRiDSegDataset(root, split="test", image_size=args.image_size,
                              lesions=args.lesions, fov_crop=True, augment=False)

    n = len(full_train_aug)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_clean, val_idx)

    C = full_train_aug.num_classes
    print(f"[info] device : {device}")
    print(f"[info] lesions: {args.lesions}  (C={C} channels)")
    print(f"[info] splits : train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  @ {args.image_size}px")
    print(f"[info] ckpt   : {ckpt_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model, desc = build(args, C, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model  : {desc}, {n_params/1e6:.2f}M params")

    pos_weight = torch.full((C, 1, 1), args.pos_weight, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_epoch = -1
    print(f"\n=== training {args.epochs} epochs (val for model selection) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            loss = seg_loss(model(imgs), masks, pos_weight)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1

        val_dice = evaluate(model, val_loader, args.lesions, device)
        mean_val = sum(val_dice.values()) / len(val_dice)
        per = "  ".join(f"{k}={v:.3f}" for k, v in val_dice.items())
        flag = ""
        if mean_val > best_val:
            best_val, best_epoch = mean_val, epoch
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "val_mean_dice": mean_val,
                "val_dice": val_dice, "lesions": args.lesions, "arch": desc,
                "args": vars(args),
            }, ckpt_path)
            flag = "  <- best (saved)"
        print(f"  epoch {epoch:2d}/{args.epochs}  train_loss={running/max(nb,1):.4f}  "
              f"val_dice[mean={mean_val:.3f}]  {per}{flag}")

    # ---- final, honest evaluation: best checkpoint on the held-out TEST set ----
    print(f"\n[info] best val mean-Dice {best_val:.3f} @ epoch {best_epoch}; "
          f"loading {ckpt_path} for test evaluation")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    test_dice = evaluate(model, test_loader, args.lesions, device)
    mean_test = sum(test_dice.values()) / len(test_dice)
    per = "  ".join(f"{k}={v:.3f}" for k, v in test_dice.items())
    print(f"\n=== TEST (held-out, best checkpoint) ===")
    print(f"  mean Dice = {mean_test:.3f}   {per}")
    print("\nOK: train/val/test separated; best checkpoint saved and scored on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
