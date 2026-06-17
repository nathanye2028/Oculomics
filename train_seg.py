"""
train_seg.py
============
Baseline trainer + smoke test for the 2D U-Net (MobileNetV3 encoder) with
GCG-gated skips (:mod:`model`) on retinal fundus images (:mod:`seg_dataset`).

It establishes the segmentation baseline (image-only, no metadata fusion),
runs a short training loop on a **1/6 subset** of the RFMiD images, and prints
tensor shapes for one batch plus the per-step loss trajectory so you can see
the loop optimises.

Note: RFMiD has no lesion masks, so this trains against the documented
placeholder target in seg_dataset.py — it validates the full architecture +
data + optimisation pipeline end-to-end, not real lesion-segmentation quality.
Swap ``--masks-dir`` to a mask-bearing dataset for genuine training.

Run:
    python train_seg.py                       # auto-detect RFMiD, 1/6 subset
    python train_seg.py --steps 20 --batch-size 4 --no-gcg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import build_model       # noqa: E402
from seg_dataset import RFMiDSegDataset  # noqa: E402
from fundus_utils import seed_everything, seed_worker, focal_tversky_loss  # noqa: E402

KAGGLE_SLUG = "andrewmvd/retinal-disease-classification"
RFMID_TRAIN_REL = os.path.join(
    "Training_Set", "Training_Set", "Training"
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_images_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    import kagglehub
    root = kagglehub.dataset_download(KAGGLE_SLUG)
    return os.path.join(root, RFMID_TRAIN_REL)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Focal Tversky: recall-favouring, robust to the extreme lesion/background
    # imbalance where plain BCE/Dice collapses to all-background.
    return focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=0.75)


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    num = 2 * (pred * target).sum() + 1.0
    den = pred.sum() + target.sum() + 1.0
    return (num / den).item()


def main() -> int:
    p = argparse.ArgumentParser(description="Baseline GCG-U-Net seg trainer / smoke test.")
    p.add_argument("--images-dir", default=None, help="Override RFMiD Training dir.")
    p.add_argument("--masks-dir", default=None, help="Real masks dir (else placeholder).")
    p.add_argument("--fraction", type=float, default=1.0 / 6.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--arch", default="baseline", choices=["baseline", "gcg_unet"],
                   help="'baseline' = plain U-Net benchmark; 'gcg_unet' = MobileNetV3+GCG.")
    p.add_argument("--base", type=int, default=32, help="Baseline U-Net base channel width.")
    p.add_argument("--no-gcg", action="store_true", help="Disable GCG blocks (gcg_unet ablation).")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    images_dir = resolve_images_dir(args.images_dir)
    print(f"[info] device     : {device}")
    print(f"[info] images dir : {images_dir}")

    ds = RFMiDSegDataset(
        images_dir=images_dir, image_size=args.image_size,
        fraction=args.fraction, masks_dir=args.masks_dir,
        mask_mode="fov", augment=True,
    )
    print(f"[info] dataset    : {len(ds)} images (fraction={args.fraction:.4f})"
          f"{' [placeholder masks]' if not args.masks_dir else ''}")

    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, worker_init_fn=seed_worker, generator=g,
    )

    if args.arch == "baseline":
        from unet_baseline import build_baseline
        model = build_baseline(num_classes=1, base=args.base).to(device)
        desc = f"baseline U-Net (base={args.base}, no gating)"
    else:
        model = build_model(
            arch="gcg_unet", num_classes=1,
            pretrained=args.pretrained, use_gcg=not args.no_gcg,
        ).to(device)
        desc = (f"GCG-U-Net (MobileNetV3 enc) "
                f"{'WITH' if not args.no_gcg else 'NO'} GCG")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model      : {desc}, {n_params/1e6:.2f}M params")

    # --- inspect exactly one batch ---
    images, masks = next(iter(loader))
    print("\n=== one batch ===")
    print(f"image : shape={tuple(images.shape)} dtype={images.dtype}")
    print(f"mask  : shape={tuple(masks.shape)} dtype={masks.dtype} "
          f"(pos frac={masks.mean().item():.3f})")
    with torch.no_grad():
        logits = model(images.to(device))
    print(f"logits: shape={tuple(logits.shape)} dtype={logits.dtype}")
    assert logits.shape == (images.shape[0], 1, args.image_size, args.image_size)

    # --- short training loop ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    print(f"\n=== training {args.steps} steps (batch={args.batch_size}) ===")
    it = iter(loader)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            imgs, msks = next(it)
        except StopIteration:
            it = iter(loader); imgs, msks = next(it)
        imgs, msks = imgs.to(device), msks.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            loss = dice_bce_loss(out, msks)
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        print(f"  step {step:2d}/{args.steps}  loss={loss.item():.4f}  "
              f"dice={dice_score(out, msks):.4f}")
    dt = time.time() - t0
    print(f"\nOK: trained {args.steps} steps in {dt:.1f}s "
          f"({dt/args.steps:.2f}s/step on {device.type}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
