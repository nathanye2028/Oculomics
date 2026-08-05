"""
pretrain_vessel.py
==================
Pretrain the **entire GCG-U-Net** (encoder *and* decoder) on retinal vessel
segmentation, then transfer it to IDRiD lesion segmentation.

Why this beats classification pretraining
-----------------------------------------
``pretrain_encoder.py`` (DDR/RFMiD) can only initialise the encoder — the whole
decoder, the skip fusions and the GCG blocks still start random. Vessel
segmentation is the *same kind of task* as lesion segmentation (dense, binary,
thin structures), so here **every layer except the final 1x1 head** is
pretrained on real data:

    vessels : 600 train images, ~7% positive pixels, 2048x2048
    lesions :  43 train images, <1% positive pixels

Only the output head differs (1 channel -> 4 lesion channels), so the transfer
is a partial state_dict load that keeps every shape-compatible tensor.

Run:
    python pretrain_vessel.py --epochs 30 --out weights/vessel_unet.pt
    python run_experiment.py ... --init-weights weights/vessel_unet.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import build_model                        # noqa: E402
from vessel_dataset import VesselSegDataset, download_vessels  # noqa: E402
from fundus_utils import seed_everything, seed_worker    # noqa: E402
from metrics import CSVLogger                            # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_compatible(model, state_dict, verbose=True):
    """Load every tensor whose name AND shape match; report the rest.

    Used to move vessel-pretrained weights into the 4-class lesion model: the
    final head (1 vs 4 output channels) is skipped, everything else transfers.
    """
    own = model.state_dict()
    ok, skipped = [], []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            ok.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    if verbose:
        print(f"[transfer] loaded {len(ok)}/{len(own)} tensors; "
              f"skipped {len(skipped)} (shape/name mismatch, e.g. output head)")
        if skipped:
            print(f"[transfer] skipped keys: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")
    return len(ok), len(skipped)


def dice_bce(logits, target):
    """Vessels are ~7% of pixels — balanced enough that plain Dice+BCE works."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    num = 2 * (p * target).sum(dims) + 1.0
    den = p.sum(dims) + target.sum(dims) + 1.0
    return bce + (1.0 - num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = tot_p = tot_t = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        inter += (pred * y).sum().item()
        tot_p += pred.sum().item()
        tot_t += y.sum().item()
    return 2 * inter / max(tot_p + tot_t, 1e-6)


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain GCG-U-Net on vessel segmentation.")
    p.add_argument("--root", default=None)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=0, help=">0 -> native-res patches.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8,
                   help="Early stop after N epochs with no test-Dice gain (0 = off).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--nondeterministic", action="store_true",
                   help="Allow non-deterministic cuDNN kernels (faster, not reproducible).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--no-imagenet", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--out", default="weights/vessel_unet.pt")
    args = p.parse_args()

    seed_everything(args.seed, deterministic=not args.nondeterministic)
    device = pick_device()
    root = args.root or download_vessels()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    patch = args.patch_size if args.patch_size > 0 else None

    train_ds = VesselSegDataset(root, "train", image_size=args.image_size,
                                patch_size=patch, augment=True, seed=args.seed)
    test_ds = VesselSegDataset(root, "test", image_size=args.image_size,
                               patch_size=None, augment=False, seed=args.seed)
    print(f"[info] device : {device}")
    print(f"[info] data   : train={len(train_ds)} test={len(test_ds)}  "
          f"{'patch '+str(patch) if patch else 'resize '+str(args.image_size)}px")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              generator=g, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, worker_init_fn=seed_worker,
                             pin_memory=(device.type == "cuda"))

    # Same architecture as the lesion model, 1 output channel (vessel vs background).
    model = build_model(arch="gcg_unet", num_classes=1, pretrained=not args.no_imagenet,
                        use_gcg=not args.no_gcg).to(device)
    print(f"[info] model  : GCG-U-Net vessels, "
          f"{sum(q.numel() for q in model.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    csv_log = CSVLogger(os.path.splitext(args.out)[0] + "_metrics.csv")

    best, since_best = -1.0, 0
    print(f"\n=== pretraining up to {args.epochs} epochs "
          f"(test Dice selects best; patience={args.patience or 'off'}) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = dice_bce(model(x), y)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
            if args.max_steps and step >= args.max_steps:
                break
        dice = evaluate(model, test_loader, device)
        tag = ""
        if dice > best:
            best, since_best = dice, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "vessel_dice": dice, "args": vars(args)}, args.out)
            tag = "  <- best (saved)"
        else:
            since_best += 1
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={run/max(nb,1):.4f}  "
              f"vessel_Dice={dice:.4f}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(run/max(nb,1), 5),
                     "vessel_dice": round(dice, 5)})
        sched.step()
        if args.patience and since_best >= args.patience:
            print(f"  early stop: no gain for {args.patience} epochs (best {best:.4f})")
            break
    csv_log.close()

    print(f"\nbest vessel Dice {best:.4f}  ->  {args.out}")
    print(f"next:  python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 "
          f"--eval-tiled --init-weights {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
