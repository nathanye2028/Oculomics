"""
pretrain_retlesion.py
=====================
Pretrain the full GCG-U-Net on the **coarse** lesion-localization dataset
(``ajithdari/retinal-lesion-segmentation-dataset``: 1,393 usable EyePACS-derived
images, MA + cotton-wool-spot masks), then fine-tune on IDRiD.

Why *pretraining* and not extra training data
---------------------------------------------
We measured the annotation granularity against IDRiD's expert masks:

    blob diameter as % of retina width :  ajithdari 2.30%   IDRiD 0.535%
    objects marked per image           :  ajithdari 8       IDRiD 38
    image positive fraction            :  ajithdari 0.78%   IDRiD 0.099%

So these "microaneurysms" are ~4x wider (≈18x the area) than real MAs and only
~8 are marked per image — they are **coarse region markers**, not pixel-precise
lesion outlines. Mixing them into IDRiD training would optimise for fat blobs
and be scored against 18-px dots, contaminating the comparison.

As a *pretraining* task the coarse labels are still useful: "lesion-ish region
vs background" teaches relevant features, and the head is re-learned on IDRiD.
This is the same logic as vessel pretraining, but with an advantage — the output
is already 4-channel (MA/HE/EX/SE), so **every** tensor transfers to the IDRiD
model, including the head.

Only MA and SE are annotated here; HE/EX are unknown, so per-sample ``valid``
vectors keep the loss from treating them as negatives.

Run:
    python pretrain_retlesion.py --epochs 20 --out weights/retlesion_unet.pt
    python run_experiment.py ... --init-weights weights/retlesion_unet.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import build_model                                  # noqa: E402
from retlesion_dataset import RetLesionDataset, download_retlesion  # noqa: E402
from idrid_dataset import DEFAULT_LESIONS                          # noqa: E402
from fundus_utils import seed_everything, seed_worker, focal_tversky_loss  # noqa: E402
from metrics import dice_iou_from_counts, CSVLogger                # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, lesions, device):
    """Per-lesion Dice, counting only channels that are annotated per sample."""
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    psum = torch.zeros(C, device=device)
    tsum = torch.zeros(C, device=device)
    for x, m, v in loader:
        x, m, v = x.to(device), m.to(device), v.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        w = v[:, :, None, None]                    # mask out unannotated channels
        inter += (pred * m * w).sum(dim=(0, 2, 3))
        psum += (pred * w).sum(dim=(0, 2, 3))
        tsum += (m * w).sum(dim=(0, 2, 3))
    dice, _ = dice_iou_from_counts(inter, psum, tsum, lesions)
    return dice


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain GCG-U-Net on coarse lesion localization.")
    p.add_argument("--root", default=None)
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS))
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--no-scale-match", action="store_true",
                   help="Disable IDRiD scale matching (default: match).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--no-imagenet", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--out", default="weights/retlesion_unet.pt")
    args = p.parse_args()

    seed_everything(args.seed, deterministic=not args.nondeterministic)
    device = pick_device()
    root = args.root or download_retlesion()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    full = RetLesionDataset(root, lesions=args.lesions, patch_size=args.patch_size,
                            scale_match=not args.no_scale_match, drop_empty=True,
                            augment=True, seed=args.seed)
    # The clean (val-transform) view shares full's item list instead of
    # re-running the ~1,900-mask non-empty scan a second time; only the
    # per-sample transform flags differ.
    import copy
    clean = copy.copy(full)
    clean.augment = False
    clean.fg_bias = 0.0
    n = len(full)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_ds = Subset(clean, perm[:n_val].tolist())
    train_ds = Subset(full, perm[n_val:].tolist())
    C = full.num_classes

    print(f"[info] device : {device}")
    print(f"[info] data   : train={len(train_ds)} val={len(val_ds)}  "
          f"annotated={full.counts()}  (dropped {full.n_dropped} empty-mask images)")
    print(f"[info] scale  : {'matched to IDRiD' if not args.no_scale_match else 'native 896px'}"
          f"  patch={args.patch_size}")

    g = torch.Generator(); g.manual_seed(args.seed)
    dl = lambda ds, **kw: DataLoader(ds, batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, generator=g,
                                     pin_memory=(device.type == "cuda"), **kw)
    train_loader, val_loader = dl(train_ds, shuffle=True, drop_last=True), dl(val_ds, shuffle=False)

    # 4-channel output -> the ENTIRE network (head included) transfers to IDRiD.
    model = build_model(arch="gcg_unet", num_classes=C, pretrained=not args.no_imagenet,
                        use_gcg=not args.no_gcg).to(device)
    print(f"[info] model  : GCG-U-Net, C={C}, "
          f"{sum(q.numel() for q in model.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    csv_log = CSVLogger(os.path.splitext(args.out)[0] + "_metrics.csv")

    best, best_ep, since = -1.0, -1, 0
    print(f"\n=== pretraining up to {args.epochs} epochs (val Dice on annotated channels) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for step, (x, m, v) in enumerate(train_loader, 1):
            x, m, v = x.to(device), m.to(device), v.to(device)
            opt.zero_grad(set_to_none=True)
            loss = focal_tversky_loss(model(x), m, alpha=0.7, beta=0.3, gamma=0.75, valid=v)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
            if args.max_steps and step >= args.max_steps:
                break
        d = evaluate(model, val_loader, args.lesions, device)
        annotated = [c for c in args.lesions if full.counts().get(c, 0) > 0]
        mean_d = float(np.mean([d[c] for c in annotated])) if annotated else 0.0
        tag = ""
        if mean_d > best:
            best, best_ep, since = mean_d, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": d,
                        "val_mean": mean_d, "lesions": args.lesions,
                        "args": vars(args)}, args.out)
            tag = "  <- best (saved)"
        else:
            since += 1
        per = "  ".join(f"{c}={d[c]:.3f}" for c in annotated)
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={run/max(nb,1):.4f}  "
              f"val[mean={mean_d:.3f}]  {per}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(run/max(nb,1), 5),
                     "val_mean_dice": round(mean_d, 5),
                     **{f"val_dice_{c}": round(d[c], 5) for c in annotated}})
        sched.step()
        if args.patience and since >= args.patience:
            print(f"  early stop: no gain for {args.patience} epochs (best {best:.3f})")
            break
    csv_log.close()

    print(f"\nbest val mean-Dice {best:.3f} @ epoch {best_ep}  ->  {args.out}")
    print("NOTE: this Dice is against COARSE labels; it is not comparable to IDRiD Dice.")
    print(f"next:  python run_experiment.py --seeds 0 1 2 --epochs 200 --patience 0 "
          f"--patch-size 512 --eval-tiled --init-weights {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
