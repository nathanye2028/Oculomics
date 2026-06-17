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
    val    -> per-epoch metric for *model selection* (best checkpoint)
    test   -> scored EXACTLY ONCE at the end, on the best checkpoint

No metric is read off the just-trained batch; test never influences training
or selection. Best model (highest val mean-Dice) is saved to ``--ckpt-dir``.

Imbalance & resolution
----------------------
Lesions are <1% of pixels, so the loss is **Focal Tversky** (recall-favouring),
not plain BCE/Dice which collapses to background. With ``--patch-size`` the
model trains on native-resolution lesion-biased crops (microaneurysms survive),
and ``--eval-tiled`` scores the held-out sets by stitching native-resolution
tiles instead of downscaling.

Apples-to-apples GCG comparison
-------------------------------
The control for "does Guided Context Gating help?" is the **same backbone with
gating off**:

    python train_idrid.py --arch gcg_unet              # GCG on
    python train_idrid.py --arch gcg_unet --no-gcg     # GCG control

``--arch baseline`` (vanilla U-Net) is a SEPARATE standard-architecture
reference, NOT the GCG ablation (different backbone/depth/pretraining).
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
from fundus_utils import (  # noqa: E402
    seed_everything, seed_worker, focal_tversky_loss, tversky_loss, tiled_predict,
)

KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loss(name: str):
    if name == "focal_tversky":
        return lambda lg, t: focal_tversky_loss(lg, t, alpha=0.7, beta=0.3, gamma=0.75)
    if name == "tversky":
        return lambda lg, t: tversky_loss(lg, t, alpha=0.7, beta=0.3)
    if name == "dice_bce":
        def _dice_bce(lg, t):
            bce = F.binary_cross_entropy_with_logits(lg, t)
            p = torch.sigmoid(lg); dims = (0, 2, 3)
            num = 2 * (p * t).sum(dims) + 1.0
            den = p.sum(dims) + t.sum(dims) + 1.0
            return bce + (1.0 - num / den).mean()
        return _dice_bce
    raise ValueError(f"unknown loss {name!r}")


def _dice_from_counts(inter, denom, lesions):
    dice = (inter / denom.clamp(min=1e-6)).cpu()
    return {code: dice[i].item() for i, code in enumerate(lesions)}


@torch.no_grad()
def evaluate_whole(model, loader, lesions, device):
    """Per-lesion Dice over a loader (whole-image, resized). Fast; for val selection."""
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    denom = torch.zeros(C, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = (torch.sigmoid(model(imgs)) > 0.5).float()
        inter += 2 * (pred * masks).sum(dim=(0, 2, 3))
        denom += pred.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    return _dice_from_counts(inter, denom, lesions)


@torch.no_grad()
def evaluate_tiled(model, dataset, lesions, device, tile, overlap):
    """Per-lesion Dice at NATIVE resolution via tiled inference. Accurate; for final test."""
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C)
    denom = torch.zeros(C)
    for i in range(len(dataset)):
        img_np, masks = dataset.load_full(i)
        img_t = dataset._to_tensors(img_np, masks)[0]            # native [3,H,W]
        prob = tiled_predict(model, img_t, tile, overlap, device).cpu()
        pred = (prob > 0.5).float()
        target = torch.from_numpy(masks.astype(np.float32))
        inter += 2 * (pred * target).sum(dim=(1, 2))
        denom += pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return _dice_from_counts(inter, denom, lesions)


def build(args, num_classes, device):
    if args.arch == "baseline":
        from unet_baseline import build_baseline
        m = build_baseline(num_classes=num_classes, base=args.base)
        desc = f"baseline U-Net (base={args.base}) [standard-architecture ref, NOT the GCG control]"
    else:
        from model_seg import build_model
        m = build_model(arch="gcg_unet", num_classes=num_classes,
                        pretrained=args.pretrained, use_gcg=not args.no_gcg)
        desc = f"GCG-U-Net (MobileNetV3) {'WITH GCG' if not args.no_gcg else 'NO GCG (control)'}"
    return m.to(device), desc


def default_run_name(args) -> str:
    if args.arch == "baseline":
        return f"baseline_b{args.base}"
    return "gcg_unet_" + ("gcg" if not args.no_gcg else "nogcg")


def main() -> int:
    p = argparse.ArgumentParser(description="IDRiD lesion-segmentation trainer / benchmark.")
    p.add_argument("--root", default=None)
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS))
    p.add_argument("--image-size", type=int, default=512, help="Whole-image (resize) size.")
    p.add_argument("--patch-size", type=int, default=0, help=">0 -> native-res patch training.")
    p.add_argument("--fg-bias", type=float, default=0.7, help="Prob a train patch is lesion-centred.")
    p.add_argument("--eval-tiled", action="store_true", help="Score test at native res via tiling.")
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--loss", default="focal_tversky", choices=["focal_tversky", "tversky", "dice_bce"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    run_name = args.run_name or default_run_name(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}.pt")
    patch = args.patch_size if args.patch_size > 0 else None

    root = args.root
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download(KAGGLE_SLUG)

    # Train view: augmented + (optionally) native-res patches. Val/test: whole image (clean).
    train_full = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                                 lesions=args.lesions, fov_crop=True, patch_size=patch,
                                 fg_bias=args.fg_bias, augment=True, seed=args.seed)
    val_full = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                               lesions=args.lesions, fov_crop=True, patch_size=None,
                               augment=False, seed=args.seed)
    test_ds = IDRiDSegDataset(root, split="test", image_size=args.image_size,
                              lesions=args.lesions, fov_crop=True, patch_size=None,
                              augment=False, seed=args.seed)

    n = len(train_full)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    C = train_full.num_classes
    print(f"[info] device : {device}")
    print(f"[info] lesions: {args.lesions}  (C={C})   loss={args.loss}")
    print(f"[info] splits : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[info] train  : {'patch '+str(patch)+'px native-res' if patch else 'whole-image '+str(args.image_size)+'px'}"
          f"   eval={'tiled native-res' if args.eval_tiled else 'whole-image '+str(args.image_size)+'px'}")
    print(f"[info] ckpt   : {ckpt_path}")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              worker_init_fn=seed_worker, generator=g,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            pin_memory=(device.type == "cuda"))

    model, desc = build(args, C, device)
    print(f"[info] model  : {desc}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    loss_fn = make_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val, best_epoch = -1.0, -1
    print(f"\n=== training {args.epochs} epochs (val for selection) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(imgs), masks)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1

        val_dice = evaluate_whole(model, val_loader, args.lesions, device)
        mean_val = sum(val_dice.values()) / len(val_dice)
        per = "  ".join(f"{k}={v:.3f}" for k, v in val_dice.items())
        flag = ""
        if mean_val > best_val:
            best_val, best_epoch = mean_val, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mean_dice": mean_val,
                        "val_dice": val_dice, "lesions": args.lesions, "arch": desc,
                        "args": vars(args)}, ckpt_path)
            flag = "  <- best (saved)"
        print(f"  epoch {epoch:2d}/{args.epochs}  loss={running/max(nb,1):.4f}  "
              f"val[mean={mean_val:.3f}]  {per}{flag}")

    # ---- final honest eval: best checkpoint on held-out TEST ----
    print(f"\n[info] best val mean-Dice {best_val:.3f} @ epoch {best_epoch}; loading {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    if args.eval_tiled:
        tile = patch or args.image_size
        test_dice = evaluate_tiled(model, test_ds, args.lesions, device, tile, args.tile_overlap)
    else:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, worker_init_fn=seed_worker)
        test_dice = evaluate_whole(model, test_loader, args.lesions, device)
    mean_test = sum(test_dice.values()) / len(test_dice)
    per = "  ".join(f"{k}={v:.3f}" for k, v in test_dice.items())
    print(f"\n=== TEST (held-out, best checkpoint) ===\n  mean Dice = {mean_test:.3f}   {per}")
    print("\nOK: train/val/test separated; focal-tversky loss; best checkpoint scored on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
