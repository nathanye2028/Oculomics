"""
eval_fgadr.py
=============
Evaluate a trained segmentation checkpoint on the **FGADR test split** and say,
per lesion, whether it actually works.

Complements the other checks rather than repeating them:
  * ``overfit_test.py``   — can the model fit 3 fixed IDRiD patches? (correctness)
  * ``eval_fgadr.py``     — how does a checkpoint score on 367 held-out FGADR
                            images? (generalisation)  <- this file
  * ``evaluate_deploy.py``— does INT8 quantization cost accuracy? (deployment)

Why sensitivity and precision, not just Dice
--------------------------------------------
Dice alone cannot distinguish "predicts nothing" from "predicts everything" —
both score near zero, but only the second is fixable by moving the threshold.
For lesions at FGADR's measured prevalence (MA is 0.063% of pixels) that
distinction is the whole diagnosis, so every row reports Dice, IoU, sensitivity
and precision together.

``--tiled`` runs inference at native 1280 via ``fundus_utils.tiled_predict``
instead of downscaling. Slower, but it is the honest number for small lesions:
a global resize to 512 leaves a median microaneurysm just 14 px.

Run:
    python eval_fgadr.py --checkpoint checkpoints/gcg_seed0.pt
    python eval_fgadr.py --checkpoint checkpoints/gcg_seed0.pt --limit 40 --save-overlays out/
    python eval_fgadr.py --checkpoint checkpoints/gcg_seed0.pt --tiled --tile 512
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fgadr_dataset import (                        # noqa: E402
    FGADRSegDataset, IMAGENET_MEAN, IMAGENET_STD,
)
from fundus_utils import tiled_predict             # noqa: E402
from metrics import dice_iou_from_counts           # noqa: E402

# Overlay colours per lesion (RGB), chosen to stay distinguishable when blended.
_COLOURS = {
    "MA": (255, 64, 64), "HE": (255, 160, 32),
    "EX": (64, 224, 96), "SE": (96, 160, 255),
    "IRMA": (224, 96, 224), "NV": (255, 255, 96),
}


def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device, arch: str, use_gcg: bool):
    """Load a checkpoint, preferring the lesion list recorded inside it."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    lesions = ck.get("lesions") if isinstance(ck, dict) else None
    if lesions:
        print(f"[info] lesions from checkpoint: {lesions}")
    else:
        lesions = ["MA", "HE", "EX", "SE"]
        print(f"[warn] checkpoint records no lesion list; assuming {lesions}")

    from model_seg import arch_cfg_from_checkpoint, build_model
    # The checkpoint records which encoder/decoder it was trained with; honour it
    # rather than assuming the defaults, or a separable-decoder run would load as
    # a pile of missing keys and score near chance.
    cfg = arch_cfg_from_checkpoint(ck)
    print(f"[info] arch from checkpoint: encoder={cfg['encoder']} decoder={cfg['decoder']} "
          f"lateral={cfg['lateral_channels']}")
    net = build_model(arch=arch, num_classes=len(lesions), pretrained=False,
                      use_gcg=use_gcg, **cfg)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"[info] checkpoint epoch {ck['epoch']}"
              + (f", recorded val mean dice {ck['val_mean_dice']:.4f}"
                 if "val_mean_dice" in ck else ""))
    return net.eval().to(device), list(lesions)


def save_overlay(img_np: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                 lesions: List[str], path: str) -> None:
    """Write a side-by-side ground-truth vs prediction overlay PNG.

    Left = annotation, right = prediction, so a glance shows whether the model
    is missing lesions or inventing them.
    """
    from PIL import Image

    def blend(base: np.ndarray, masks: np.ndarray) -> np.ndarray:
        out = base.astype(np.float32).copy()
        for c, code in enumerate(lesions):
            m = masks[c].astype(bool)
            if not m.any():
                continue
            colour = np.array(_COLOURS.get(code, (255, 255, 255)), dtype=np.float32)
            out[m] = 0.35 * out[m] + 0.65 * colour
        return out.clip(0, 255).astype(np.uint8)

    pair = np.concatenate([blend(img_np, gt), blend(img_np, pred)], axis=1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(pair).save(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the FGADR test split.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=None, help="FGADR Seg-set dir (auto-detected under data/).")
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images.")
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--tiled", action="store_true",
                   help="Infer at native resolution via tiled_predict (slower, fairer to MA).")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--save-overlays", default=None, metavar="DIR",
                   help="Write side-by-side GT/prediction PNGs here.")
    p.add_argument("--max-overlays", type=int, default=12)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    net, lesions = load_model(args.checkpoint, device, args.arch, not args.no_gcg)

    ds = FGADRSegDataset(args.root, split=args.split, image_size=args.image_size,
                         lesions=lesions, augment=False)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[info] device={device}  split={args.split}  images={n}"
          f"{'  (tiled @ native)' if args.tiled else f'  @ {args.image_size}px'}")

    C = len(lesions)
    inter = torch.zeros(C); psum = torch.zeros(C); tsum = torch.zeros(C)
    empty_gt = torch.zeros(C)          # images where the lesion is truly absent
    t0 = time.time()

    for i in range(n):
        if args.tiled:
            img_np, gt_np = ds.load_full(i)                       # native resolution
            x = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            probs = tiled_predict(net, x, args.tile, args.overlap, device).cpu()
            gt = torch.from_numpy(gt_np.astype(np.float32))
        else:
            x, gt, _ = ds[i]
            with torch.inference_mode():
                probs = torch.sigmoid(net(x[None].to(device)))[0].cpu()
            img_np = None

        pred = (probs > args.thresh).float()
        inter += (pred * gt).sum(dim=(1, 2))
        psum += pred.sum(dim=(1, 2))
        tsum += gt.sum(dim=(1, 2))
        empty_gt += (gt.sum(dim=(1, 2)) == 0).float()

        if args.save_overlays and i < args.max_overlays:
            if img_np is None:
                img_np = ((x * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1) * 255) \
                    .permute(1, 2, 0).numpy()
            save_overlay(np.asarray(img_np, dtype=np.uint8), pred.numpy(), gt.numpy(),
                         lesions, os.path.join(args.save_overlays, f"{ds.files[i]}"))

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"\r  {i+1}/{n} images ({time.time()-t0:.0f}s)", end="", flush=True)
    print()

    dice, iou = dice_iou_from_counts(inter, psum, tsum, lesions)
    sens = (inter / tsum.clamp(min=1e-6))
    prec = (inter / psum.clamp(min=1e-6))

    print(f"\n=== FGADR {args.split} — {n} images @ thresh {args.thresh} ===")
    print(f"{'lesion':<7}{'dice':>8}{'IoU':>8}{'sens':>8}{'prec':>8}{'imgs w/o lesion':>18}")
    print("-" * 57)
    for i, c in enumerate(lesions):
        print(f"{c:<7}{dice[c]:>8.4f}{iou[c]:>8.4f}{sens[i]:>8.4f}{prec[i]:>8.4f}"
              f"{int(empty_gt[i]):>13}/{n:<4}")
    print("-" * 57)
    print(f"{'mean':<7}{np.mean(list(dice.values())):>8.4f}{np.mean(list(iou.values())):>8.4f}"
          f"{sens.mean():>8.4f}{prec.mean():>8.4f}")

    # Interpretation hints — the failure modes Dice alone hides.
    for i, c in enumerate(lesions):
        ratio = (psum[i] / tsum[i]).item() if tsum[i] > 0 else float("inf")
        if tsum[i] == 0:
            print(f"[diag] {c}: no ground-truth pixels in these {n} images — "
                  f"unscoreable here (model emitted {int(psum[i])} px). "
                  "Evaluate on more images or the full split.")
        elif psum[i] == 0:
            print(f"[diag] {c}: predicted ZERO positive pixels — collapsed to background. "
                  "Raise alpha (false-negative weight) or train longer.")
        elif inter[i] == 0:
            print(f"[diag] {c}: predicted {int(psum[i])} px but ZERO overlap with the "
                  f"{int(tsum[i])} px of ground truth — not collapsed, just wrong "
                  "everywhere. Expected from an undertrained model.")
        elif sens[i] > 0.9 and prec[i] < 0.05:
            print(f"[diag] {c}: over-predicting {ratio:.0f}x the true lesion area "
                  f"(sens {sens[i]:.2f}, prec {prec[i]:.3f}) — expected early in "
                  "training; recovers as it sharpens.")
    if args.save_overlays:
        print(f"\n[ok] overlays in {args.save_overlays}/ (left = ground truth, right = prediction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
