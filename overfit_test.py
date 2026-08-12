"""
overfit_test.py
===============
The single strongest correctness check short of a full training run:
**can the model overfit one tiny, fixed batch of real IDRiD data?**

If the data pipeline, loss, optimizer and gradient flow are all correct, the
network MUST drive loss -> ~0 and Dice -> ~1 on a handful of fixed images.
If it can't, something is genuinely broken — this catches:
  * image<->mask misalignment after FOV crop / patching,
  * a loss that can't actually drive learning,
  * dead gradients through the GCG block,
  * label / channel mix-ups.

We sample patches that *contain lesions* at NATIVE resolution (so even
microaneurysms are big enough to learn), freeze that exact batch, and train on
it. We also assert the GCG block's parameters receive non-zero gradients.

Run:
    python overfit_test.py                      # GCG-U-Net, with gating
    python overfit_test.py --no-gcg             # control
    python overfit_test.py --gcg-variant attention
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idrid_dataset import IDRiDSegDataset, DEFAULT_LESIONS  # noqa: E402
from fundus_utils import seed_everything, focal_tversky_loss  # noqa: E402
from model_seg import ENCODER_NAMES  # noqa: E402

KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def per_lesion_dice(logits, target, lesions):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    psum = pred.sum(dim=(0, 2, 3)); tsum = target.sum(dim=(0, 2, 3))
    dice = (2 * inter / (psum + tsum).clamp(min=1e-6)).cpu()
    present = (tsum > 0).cpu()
    return {l: dice[i].item() for i, l in enumerate(lesions)}, present


def build(args, C, device):
    from model_seg import build_model
    if args.arch == "baseline":
        from unet_baseline import build_baseline
        return build_baseline(num_classes=C, base=args.base).to(device), "baseline U-Net"
    gcg_factory = None
    variant = "off" if args.no_gcg else args.gcg_variant
    if not args.no_gcg and args.gcg_variant != "baseline":
        from gcg_blocks import GCG_VARIANTS
        gcg_factory = GCG_VARIANTS[args.gcg_variant]
    lat = args.lateral_channels
    m = build_model(arch="gcg_unet", num_classes=C, pretrained=False,
                    use_gcg=not args.no_gcg, gcg_factory=gcg_factory,
                    encoder=args.encoder, decoder=args.decoder,
                    lateral_channels=None if lat < 0 else lat)
    return m.to(device), f"GCG-U-Net ({args.encoder}/{args.decoder}, gcg={variant})"


def main() -> int:
    p = argparse.ArgumentParser(description="Overfit a fixed IDRiD batch (correctness check).")
    p.add_argument("--root", default=None)
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS))
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    # A new backbone can build cleanly and still fail to LEARN — wrong tap strides,
    # a skip the decoder never receives, or dead gradients through the gate. Unit
    # tests catch shapes; only this catches "trains but goes nowhere". Run it per
    # encoder before committing a 15-run sweep to the queue.
    p.add_argument("--encoder", default="mobilenetv3", choices=list(ENCODER_NAMES),
                   help="Backbone under test. NB: efficientvit_* expose 4 taps "
                        "(strides 4..32), not MobileNet's 5 — worth verifying.")
    p.add_argument("--decoder", default="dense", choices=["dense", "separable"])
    p.add_argument("--lateral-channels", type=int, default=-1,
                   help="-1 = decide from --decoder; 0 = force off; N = project to N.")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--gcg-variant", default="baseline",
                   choices=["baseline", "attention", "cbam", "se", "none"])
    p.add_argument("--dice-thresh", type=float, default=0.80,
                   help="Pass bar: mean Dice over present lesions must exceed this.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    root = args.root
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download(KAGGLE_SLUG)

    # Lesion-biased native-resolution patches -> guaranteed lesion content.
    ds = IDRiDSegDataset(root, split="train", lesions=args.lesions,
                         patch_size=args.patch_size, fg_bias=1.0, augment=False, seed=args.seed)
    C = ds.num_classes

    # Freeze ONE fixed batch in memory (materialize patches once, reuse every step).
    imgs, masks = [], []
    for i in range(args.batch_size):
        x, m = ds[i]
        imgs.append(x); masks.append(m)
    imgs = torch.stack(imgs).to(device)
    masks = torch.stack(masks).to(device)
    pos = masks.sum(dim=(0, 2, 3))
    print(f"[info] device={device}  fixed batch={tuple(imgs.shape)}  patch={args.patch_size}px")
    print(f"[info] lesion pixels in batch: " +
          "  ".join(f"{l}={int(pos[i])}" for i, l in enumerate(args.lesions)))

    model, desc = build(args, C, device)
    print(f"[info] model: {desc}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    init_loss = None
    gcg_grad_ok = None
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = focal_tversky_loss(logits, masks)
        loss.backward()
        # one-time gradient-flow check through the GCG block
        if gcg_grad_ok is None and not args.no_gcg and args.arch == "gcg_unet":
            gnorm = sum(p.grad.abs().sum().item()
                        for n, p in model.named_parameters()
                        if "gcg" in n and p.grad is not None)
            n_gcg = sum(1 for n, _ in model.named_parameters() if "gcg" in n)
            gcg_grad_ok = (n_gcg > 0 and gnorm > 0)
            print(f"[info] GCG params={n_gcg}, grad-flow {'OK' if gcg_grad_ok else 'FAIL'} (|grad|={gnorm:.3e})")
        opt.step()
        if init_loss is None:
            init_loss = loss.item()
        if step % 50 == 0 or step == 1:
            dice, present = per_lesion_dice(logits, masks, args.lesions)
            shown = "  ".join(f"{l}={dice[l]:.3f}{'*' if present[i] else ''}"
                              for i, l in enumerate(args.lesions))
            print(f"  step {step:4d}  loss={loss.item():.4f}  dice[{shown}]")

    # final assessment on present lesions
    dice, present = per_lesion_dice(model(imgs), masks, args.lesions)
    present_vals = [dice[l] for i, l in enumerate(args.lesions) if present[i]]
    mean_present = sum(present_vals) / max(len(present_vals), 1)
    final_loss = focal_tversky_loss(model(imgs), masks).item()

    print("\n" + "=" * 60)
    print(f"init loss {init_loss:.4f} -> final loss {final_loss:.4f}")
    print(f"mean Dice over present lesions ({[l for i,l in enumerate(args.lesions) if present[i]]}) "
          f"= {mean_present:.3f}")
    ok = (final_loss < 0.3) and (mean_present > args.dice_thresh)
    if not args.no_gcg and args.arch == "gcg_unet":
        ok = ok and bool(gcg_grad_ok)
        print(f"GCG gradient flow: {'OK' if gcg_grad_ok else 'FAIL'}")
    print(f"\nRESULT: {'PASS — pipeline can learn the real task' if ok else 'FAIL — investigate (alignment/loss/gradients)'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
