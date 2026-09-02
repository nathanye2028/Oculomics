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

Scaling to large batches / multiple GPUs
----------------------------------------
Effective batch = ``batch_size * accum_steps * num_gpus``.

Accumulation is exactly gradient-equivalent to one large batch only for
per-pixel losses (BCE / focal BCE, which are means over pixels). The Tversky
terms pool tp/fp/fn over the whole batch *before* dividing, so accumulating
``accum_steps`` micro-batches is a mean of per-micro-batch Tversky losses, not
the Tversky of the union — close, but not identical, and noisier for the
sparsest lesion (MA) where a micro-batch can hold zero positives.

  * One GPU, big effective batch via accumulation + fp16:
        python train_idrid.py --amp --batch-size 8 --accum-steps 4
  * Multi-GPU (DistributedDataParallel) via torchrun — batch shards across GPUs:
        torchrun --nproc_per_node=4 train_idrid.py --amp \
            --batch-size 16 --patch-size 512 --epochs 100 --eval-tiled

Under torchrun the script auto-detects RANK/WORLD_SIZE, wraps the model in DDP
(NCCL on CUDA), shards train/val/test with DistributedSampler, uses no_sync
during accumulation, all-reduces metrics, and checkpoints from rank 0.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idrid_dataset import IDRiDSegDataset, DEFAULT_LESIONS, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from fundus_utils import (  # noqa: E402
    seed_everything, seed_worker, focal_tversky_loss, tversky_loss, tiled_predict, pick_device,
)
from fgadr_dataset import DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC  # noqa: E402
from metrics import dice_iou_from_counts, sens_prec_from_counts, mean_present, CSVLogger  # noqa: E402

KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"

# Per-source salt added to --seed before it reaches each loader's make_rng.
# make_rng keys on (seed, index), so without this IDRiD image k and FGADR image
# k would draw the IDENTICAL flip / rotation / jitter / crop-offset stream
# every epoch — correlated augmentation across sources that a mixed batch has
# no way to notice. IDRiD stays at offset 0 so IDRiD-only runs reproduce
# pre-salt results bit-for-bit. Offsets are far apart so no plausible --seed
# range makes two sources collide.
SOURCE_SEED_OFFSET = {"idrid": 0, "eophtha": 10_007, "fgadr": 20_011, "retlesion": 30_013}


def git_commit() -> "str | None":
    """Short hash of the checked-out commit, or None (not a repo / no git).

    Best-effort; recorded in results so a number can be traced to code."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:                       # noqa: BLE001 - never fail a run over this
        return None


# --------------------------------------------------------------------------- #
# Distributed (multi-GPU) helpers — active only under torchrun.
# --------------------------------------------------------------------------- #
def ddp_info():
    """(rank, world_size, local_rank) from torchrun env; (0,1,0) if single-process."""
    return (int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(*tensors):
    if is_dist():
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


def make_loss(name: str):
    """Return loss(logits, target, valid=None).

    ``valid`` [B,C] marks which lesion channels are annotated for each sample,
    so partially-annotated sources (e-ophtha labels EX only) don't have their
    unlabelled channels scored as all-negative.
    """
    if name == "focal_tversky":
        return lambda lg, t, valid=None: focal_tversky_loss(
            lg, t, alpha=0.7, beta=0.3, gamma=0.75, valid=valid)
    if name == "tversky":
        return lambda lg, t, valid=None: tversky_loss(lg, t, alpha=0.7, beta=0.3, valid=valid)
    if name == "lesion_seg":
        # Focal Tversky + focal BCE (losses.py): region overlap that survives
        # extreme imbalance, plus a per-pixel gradient that behaves early in
        # training before any overlap exists.
        from losses import lesion_seg_loss
        return lambda lg, t, valid=None: lesion_seg_loss(lg, t, valid=valid)
    if name == "dice_bce":
        def _dice_bce(lg, t, valid=None):
            p = torch.sigmoid(lg); dims = (0, 2, 3)
            if valid is not None:
                # Mask PROBABILITIES, not logits: a zeroed logit is sigmoid=0.5,
                # which pads the Dice denominator of the *annotated* samples in
                # the batch and adds a constant log(2)/pixel to BCE.
                w = valid.to(lg.dtype)[:, :, None, None]
                bce_map = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
                bce = (bce_map * w).sum() / w.expand_as(bce_map).sum().clamp(min=1.0)
                p, t = p * w, t * w
            else:
                bce = F.binary_cross_entropy_with_logits(lg, t)
            num = 2 * (p * t).sum(dims) + 1.0
            den = p.sum(dims) + t.sum(dims) + 1.0
            return bce + (1.0 - num / den).mean()
        return _dice_bce
    raise ValueError(f"unknown loss {name!r}")


class _WithValid(torch.utils.data.Dataset):
    """Adapt a 2-tuple (img, mask) dataset to the 3-tuple (img, mask, valid) API."""

    def __init__(self, ds, num_classes: int):
        self.ds = ds
        self.valid = torch.ones(num_classes)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, m = self.ds[i]
        return x, m, self.valid.clone()


class _NoValid(torch.utils.data.Dataset):
    """Inverse of :class:`_WithValid` — drop ``valid`` for the 2-tuple eval API.

    Evaluation sets are fully annotated by construction, so the vector carries
    no information there; ``evaluate_whole`` expects (img, mask).
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, m = self.ds[i][:2]
        return x, m

    def load_full(self, i):                    # keep tiled eval working
        return self.ds.load_full(i)


def shard_for_eval(ds):
    """Non-padded per-rank eval shard.

    ``DistributedSampler`` pads the index list so every rank gets an equal share,
    which double-counts samples in the all-reduced Dice counts (on IDRiD's ~11
    val images with 4 GPUs that is a ~9% distortion of the selection metric).
    Counts-based eval tolerates uneven shards — there is no gradient sync to
    keep in lockstep — so a plain strided subset is both correct and simpler.
    """
    if not is_dist():
        return ds
    return Subset(ds, list(range(dist.get_rank(), len(ds), dist.get_world_size())))


@torch.no_grad()
def evaluate_whole(model, loader, lesions, device):
    """Per-lesion Dice + IoU + sensitivity/precision over a loader (whole-image,
    resized). Fast; for val selection.

    Under DDP the loader must be built over :func:`shard_for_eval` (never a
    padded ``DistributedSampler``); counts are summed across ranks so every rank
    computes the same global metrics (and agrees on the best epoch).
    Returns (dice_dict, iou_dict, sens_dict, prec_dict).
    """
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    psum = torch.zeros(C, device=device)
    tsum = torch.zeros(C, device=device)
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pred = (torch.sigmoid(model(imgs)) > 0.5).float()
        inter += (pred * masks).sum(dim=(0, 2, 3))
        psum += pred.sum(dim=(0, 2, 3))
        tsum += masks.sum(dim=(0, 2, 3))
    all_reduce_sum(inter, psum, tsum)
    return (*dice_iou_from_counts(inter, psum, tsum, lesions),
            *sens_prec_from_counts(inter, psum, tsum, lesions))


def _load_full_of(ds, i):
    """``load_full`` through any nesting of Subsets (eval shards, val splits)."""
    if isinstance(ds, Subset):
        return _load_full_of(ds.dataset, ds.indices[i])
    return ds.load_full(i)


class _NativeFull(torch.utils.data.Dataset):
    """Yields native-resolution (uint8 image [3,H,W], uint8 masks [C,H,W]).

    Returning uint8 (not normalised float) halves the worker->main transfer;
    normalisation happens once on-device in :func:`evaluate_tiled`.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img_np, masks = _load_full_of(self.ds, i)[:2]            # [H,W,3], [C,H,W] uint8
        img = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1)))
        return img, torch.from_numpy(np.ascontiguousarray(masks))


def _take_one(batch):              # module-level collate (picklable for spawn workers)
    return batch[0]


@torch.inference_mode()
def evaluate_tiled(model, dataset, lesions, device, tile, overlap, tile_batch=16, num_workers=2):
    """Per-lesion Dice at NATIVE resolution via tiled inference. Accurate; for final test.

    Efficient at scale: image/mask decode runs in parallel DataLoader workers
    (overlapping GPU compute), tiles are batched through the model, and Dice
    counts accumulate on-device (no per-image CPU sync).
    """
    model.eval()
    C = len(lesions)
    inter = torch.zeros(C, device=device)
    psum = torch.zeros(C, device=device)
    tsum = torch.zeros(C, device=device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    # Under DDP, shard images across ranks (each GPU tiles a subset) then sum
    # counts. Non-padded shard: see shard_for_eval.
    nf = _NativeFull(shard_for_eval(dataset))
    loader = DataLoader(nf, batch_size=1, num_workers=num_workers,
                        collate_fn=_take_one, worker_init_fn=seed_worker,
                        pin_memory=(device.type == "cuda"))
    for img_u8, masks_u8 in loader:
        img_u8 = img_u8.to(device, non_blocking=True)
        fg_map = img_u8.amax(dim=0) > 12               # field-of-view pixels (skip black tiles)
        img = (img_u8.float().div_(255.0) - mean) / std
        prob = tiled_predict(model, img, tile, overlap, device,
                             tile_batch=tile_batch, fg_map=fg_map, n_classes=C)
        pred = (prob > 0.5).float()
        target = masks_u8.to(device, non_blocking=True).float()
        inter += (pred * target).sum(dim=(1, 2))
        psum += pred.sum(dim=(1, 2))
        tsum += target.sum(dim=(1, 2))
    all_reduce_sum(inter, psum, tsum)
    return (*dice_iou_from_counts(inter, psum, tsum, lesions),
            *sens_prec_from_counts(inter, psum, tsum, lesions))


def build(args, num_classes, device):
    if args.arch == "baseline":
        from unet_baseline import build_baseline
        m = build_baseline(num_classes=num_classes, base=args.base)
        desc = f"baseline U-Net (base={args.base}) [standard-architecture ref, NOT the GCG control]"
    else:
        from model_seg import build_model
        gcg_factory = None
        variant = "off" if args.no_gcg else (args.gcg_variant or "baseline")
        if not args.no_gcg and args.gcg_variant and args.gcg_variant != "baseline":
            from gcg_blocks import GCG_VARIANTS
            gcg_factory = GCG_VARIANTS[args.gcg_variant]
        imagenet = not getattr(args, "no_pretrained", False)
        # -1 is the "let the decoder style decide" sentinel; 0 means explicitly off.
        lat = getattr(args, "lateral_channels", -1)
        m = build_model(arch="gcg_unet", num_classes=num_classes,
                        pretrained=imagenet, use_gcg=not args.no_gcg,
                        gcg_factory=gcg_factory,
                        encoder=args.encoder, decoder=args.decoder,
                        lateral_channels=None if lat < 0 else lat)
        desc = f"GCG-U-Net ({args.encoder}/{args.decoder}-decoder) gcg={variant}"
        desc += " [imagenet]" if imagenet else " [SCRATCH — no pretraining]"
        # In-domain init: overwrite the ImageNet encoder with one pretrained on
        # RFMiD fundus images (see pretrain_encoder.py). Fundus features transfer
        # better than natural-image features to a 54-image segmentation task.
        if getattr(args, "init_encoder", None):
            ck = torch.load(args.init_encoder, map_location="cpu")
            m.encoder.load_state_dict(ck["encoder"] if "encoder" in ck else ck)
            desc += f" [encoder<-{os.path.basename(args.init_encoder)}]"
        # Full-network transfer (e.g. vessel-pretrained GCG-U-Net): loads encoder,
        # decoder, skips and GCG blocks; only the output head (1 vs C channels)
        # is shape-incompatible and stays randomly initialised.
        if getattr(args, "init_weights", None):
            from pretrain_vessel import load_compatible
            ck = torch.load(args.init_weights, map_location="cpu")
            n_ok, _ = load_compatible(m, ck.get("model", ck))
            desc += f" [weights<-{os.path.basename(args.init_weights)}:{n_ok}]"
    return m.to(device), desc


def default_run_name(args) -> str:
    if args.arch == "baseline":
        return f"baseline_b{args.base}"
    return "gcg_unet_" + ("gcg" if not args.no_gcg else "nogcg")


def main() -> int:
    p = argparse.ArgumentParser(description="IDRiD lesion-segmentation trainer / benchmark.")
    p.add_argument("--root", default=None)
    p.add_argument("--datasets", nargs="+", default=["idrid"],
                   choices=["idrid", "eophtha", "retlesion", "fgadr"],
                   help="Training sources. IDRiD is always the val/test set. "
                        "'eophtha' adds 47 EX-only images; 'retlesion' adds ~1.4k "
                        "MA + ~0.5k cotton-wool-spot annotated images; 'fgadr' adds "
                        "1290 fully-annotated (MA/HE/EX/SE) train images, selects on "
                        "its 185-image val split and is scored on its own 367-image "
                        "test split in addition to IDRiD's (fixed split_seed=42).")
    p.add_argument("--eophtha-root", default=None)
    p.add_argument("--retlesion-root", default=None)
    p.add_argument("--fgadr-root", default=None,
                   help="FGADR Seg-set dir; auto-detected under data/ when omitted.")
    p.add_argument("--val-source", default="auto", choices=["auto", "idrid", "fgadr"],
                   help="Which held-out set selects the best checkpoint. 'auto' uses "
                        "FGADR's val split when FGADR is training data (matched "
                        "distribution, ~184 imgs), else IDRiD's ~11. Selecting on 11 "
                        "out-of-distribution images is noisy enough to pick a worse "
                        "checkpoint than a better one.")
    p.add_argument("--lesions", nargs="+", default=list(DEFAULT_LESIONS))
    p.add_argument("--image-size", type=int, default=512, help="Whole-image (resize) size.")
    p.add_argument("--patch-size", type=int, default=0, help=">0 -> native-res patch training.")
    p.add_argument("--fg-bias", type=float, default=0.7, help="Prob a train patch is lesion-centred.")
    p.add_argument("--eval-tiled", action="store_true", help="Score test at native res via tiling.")
    p.add_argument("--eval-tiled-val", action="store_true",
                   help="Also select the best checkpoint on TILED native-res val. Without "
                        "this, selection runs on whole-image resize, where NEAREST-"
                        "downsampled masks erase most 1-2px microaneurysms — the selected "
                        "epoch can be materially non-optimal for exactly the lesion the "
                        "final tiled test reports. Costs one tiled pass over val per epoch.")
    p.add_argument("--tile-overlap", type=int, default=0,
                   help="Tile overlap. 0 = fewest tiles / fastest (measured 1.3x vs 64); "
                        ">0 smooths seams at the cost of extra forwards.")
    p.add_argument("--tile-batch", type=int, default=8, help="Tiles per forward pass (CUDA win; modest on MPS).")
    p.add_argument("--batch-size", type=int, default=2, help="PHYSICAL micro-batch held in memory.")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation: effective batch = batch-size * accum-steps. "
                        "Gradient-equivalent to one large batch ONLY for per-pixel losses "
                        "(BCE terms); the pooled Tversky term is averaged over micro-batches "
                        "instead of computed on their union, so accum=4 x batch=2 is not "
                        "numerically the same run as batch=8.")
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision (fp16) -> less memory, faster matmuls.")
    p.add_argument("--epochs", type=int, default=120,
                   help="Max epochs (cosine schedule spans this; shorter = faster anneal).")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stop after N epochs with no val improvement (0 = off).")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--loss", default="focal_tversky",
                   choices=["focal_tversky", "tversky", "dice_bce", "lesion_seg"],
                   help="'lesion_seg' = focal Tversky + focal BCE (losses.py); the BCE "
                        "term gives a usable per-pixel gradient before any region "
                        "overlap exists, which matters most for MA.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--nondeterministic", action="store_true",
                   help="Allow non-deterministic cuDNN kernels (faster, not reproducible).")
    p.add_argument("--allow-tf32", action="store_true",
                   help="Re-enable TF32 convs on Ampere+ (faster, 10-bit mantissa). OFF by "
                        "default: measured on the overfit test, TF32 drove mobilenetv4_m's "
                        "MA Dice 0.974 -> 0.000 while HE/EX were unaffected, because MA "
                        "carries the smallest gradients in the batch. Use only to reproduce "
                        "a pre-fix run or to measure the speed/accuracy trade deliberately.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-images", action="store_true",
                   help="Cache decoded FOV-cropped IDRiD images in RAM (masks bit-packed). "
                        "Removes the full 4288x2848 decode from every patch fetch — the "
                        "dominant per-item cost. ~1.6GB per DataLoader worker; use on the "
                        "GPU box, not a small laptop.")
    p.add_argument("--arch", default="gcg_unet", choices=["baseline", "gcg_unet"])
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--gcg-variant", default="baseline",
                   choices=["baseline", "attention", "cbam", "se", "none"],
                   help="Which GCG block to inject (see gcg_blocks.py). 'baseline' = built-in.")
    # --- architecture / mobile cost knobs ----------------------------------- #
    # Both default to the original architecture so every result recorded before
    # 2026-08 stays reproducible. NB: measured Core ML latency does NOT track MAC
    # count on Apple silicon — see the model_seg.py module docstring before
    # choosing any of these for efficiency reasons.
    from model_seg import ENCODER_NAMES                          # noqa: E402
    p.add_argument("--encoder", default="mobilenetv3", choices=list(ENCODER_NAMES),
                   help="Backbone. 'mobilenetv3' (default) is the encoder all prior "
                        "results used. 'mobilenetv4_*' / 'efficientvit_*' need timm. "
                        "Measured ANE latency: mobilenetv4_m ~neutral (but 1.8x slower "
                        "on CPU), efficientvit_b1 ~1.4x SLOWER. Pick these for accuracy, "
                        "not for speed.")
    p.add_argument("--decoder", default="dense", choices=["dense", "separable"],
                   help="Decoder fuse convs. 'separable' cuts 14.09 -> 3.26 GMAC (4.3x) "
                        "and halves params, but measured 9.2 vs 9.0 ms on the ANE and "
                        "46.6 vs 46.5 ms on CPU: NO latency win. Bandwidth-bound, not "
                        "compute-bound. Keep 'dense' unless ablating.")
    p.add_argument("--lateral-channels", type=int, default=-1,
                   help="Width of the 1x1 projection on the deepest encoder feature. "
                        "-1 = pick by --decoder (off for dense, 256 for separable); "
                        "0 = force off. '--decoder dense --lateral-channels 256' is the "
                        "one cost change that measured faster on BOTH paths (1.07x ANE, "
                        "1.15x CPU) and drops 6.84M -> 5.38M params.")
    # Pretraining is ON by default: ImageNet transfer alone moved mean Dice
    # 0.183 -> 0.387 on IDRiD (pretrain_encoder.py), a larger effect than any
    # architecture change measured here. Scratch training must be asked for.
    p.add_argument("--pretrained", action="store_true",
                   help="Deprecated no-op — ImageNet init is now the default. "
                        "Kept so existing scripts keep working.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train the encoder from scratch (ablation only). Expect "
                        "roughly half the Dice of an ImageNet-initialised run.")
    p.add_argument("--init-encoder", default=None,
                   help="Encoder-only weights from pretrain_encoder.py (DDR/RFMiD).")
    p.add_argument("--init-weights", default=None,
                   help="FULL-network weights from pretrain_vessel.py (encoder+decoder+GCG).")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-json", default=None,
                   help="If set, rank 0 writes test metrics + config as JSON here.")
    args = p.parse_args()

    # Fail here, not after 120 epochs. Tiled eval slides a `tile x tile` window
    # at NATIVE resolution; a model trained on whole-image resizes has only ever
    # seen the retina downscaled to --image-size, so scoring it on native-res
    # crops of --image-size pixels evaluates a model on inputs at ~7x the scale
    # it learned (IDRiD is ~3400px across the FOV). The number it produces
    # describes nothing that was trained. Train with --patch-size to use it.
    if (args.eval_tiled or args.eval_tiled_val) and args.patch_size <= 0:
        raise SystemExit(
            "[fatal] --eval-tiled / --eval-tiled-val require --patch-size > 0. "
            "Tiled evaluation stitches native-resolution tiles the size of the "
            "training patch; a whole-image-trained model (patch_size=0) has never "
            "seen native-resolution input, so tiling it scores the wrong model. "
            "Either drop --eval-tiled (whole-image eval at --image-size, the mode "
            "this model was trained in) or train with --patch-size.")

    # --- distributed init (no-op unless launched with torchrun) ---
    rank, world, local_rank = ddp_info()
    ddp = world > 1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    is_main = (rank == 0)

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    seed_everything(args.seed, deterministic=not args.nondeterministic,
                    allow_tf32=args.allow_tf32)                                     # same on every rank -> identical init & data split
    device = pick_device(local_rank=local_rank, ddp=ddp)
    run_name = args.run_name or default_run_name(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}.pt")
    patch = args.patch_size if args.patch_size > 0 else None

    # Download once on rank 0; other ranks wait then read the shared cache.
    root = args.root
    if root is None:
        import kagglehub
        if ddp and not is_main:
            dist.barrier()
        root = kagglehub.dataset_download(KAGGLE_SLUG)
        if ddp and is_main:
            dist.barrier()

    # Train view: augmented + (optionally) native-res patches. Val/test: whole image (clean).
    train_full = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                                 lesions=args.lesions, fov_crop=True, patch_size=patch,
                                 fg_bias=args.fg_bias, augment=True, seed=args.seed,
                                 cache_images=args.cache_images)
    val_full = IDRiDSegDataset(root, split="train", image_size=args.image_size,
                               lesions=args.lesions, fov_crop=True, patch_size=None,
                               augment=False, seed=args.seed,
                               cache_images=args.cache_images)
    test_ds = IDRiDSegDataset(root, split="test", image_size=args.image_size,
                              lesions=args.lesions, fov_crop=True, patch_size=None,
                              augment=False, seed=args.seed)

    n = len(train_full)
    perm = np.random.default_rng(args.seed).permutation(n)   # identical split on all ranks
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    C = train_full.num_classes

    # Extra training sources. Val/test stay IDRiD-only on purpose: the held-out
    # metric must remain comparable across runs (and e-ophtha has no official
    # split). Training always yields (img, mask, valid).
    train_ds = _WithValid(train_ds, C)
    extra_counts = {}
    if "eophtha" in args.datasets:
        from multi_seg_dataset import MultiLesionSegDataset
        eo_root = args.eophtha_root
        if eo_root is None:
            import kagglehub
            from multi_seg_dataset import EOPHTHA_SLUG
            eo_root = kagglehub.dataset_download(EOPHTHA_SLUG)
        eo = MultiLesionSegDataset(idrid_root=None, eophtha_root=eo_root, split="train",
                                   image_size=args.image_size, lesions=args.lesions,
                                   patch_size=patch, fg_bias=args.fg_bias,
                                   augment=True,
                                   seed=args.seed + SOURCE_SEED_OFFSET["eophtha"])
        if len(eo):
            train_ds = torch.utils.data.ConcatDataset([train_ds, eo])
            extra_counts["eophtha(EX-only)"] = len(eo)
    # FGADR: 1290 fully-annotated train images (~30x IDRiD's 43). It dominates the
    # mixture, so it also gets scored on its own 367-image held-out split below —
    # an IDRiD-only metric over 27 images would be too noisy to attribute changes to.
    fgadr_test_ds = None
    if "fgadr" in args.datasets:
        from fgadr_dataset import FGADRSegDataset
        fg_seed = args.seed + SOURCE_SEED_OFFSET["fgadr"]     # see SOURCE_SEED_OFFSET
        fg = FGADRSegDataset(args.fgadr_root, split="train", image_size=args.image_size,
                             lesions=args.lesions, fov_crop=True, patch_size=patch,
                             fg_bias=args.fg_bias, augment=True, seed=fg_seed)
        if len(fg):
            train_ds = torch.utils.data.ConcatDataset([train_ds, fg])
            extra_counts["fgadr"] = len(fg)
        fgadr_test_ds = _NoValid(FGADRSegDataset(
            args.fgadr_root, split="test", image_size=args.image_size,
            lesions=args.lesions, fov_crop=True, patch_size=None,
            augment=False, seed=fg_seed))

        # Model selection: prefer FGADR's val split. IDRiD's ~11 held-out images
        # cannot reliably rank checkpoints once training is ~97% FGADR, and a bad
        # selection silently ships a worse model than the run actually reached.
        if args.val_source in ("auto", "fgadr"):
            val_ds = _NoValid(FGADRSegDataset(
                args.fgadr_root, split="val", image_size=args.image_size,
                lesions=args.lesions, fov_crop=True, patch_size=None,
                augment=False, seed=fg_seed))
            val_source = "fgadr"
        else:
            val_source = "idrid"
    else:
        if args.val_source == "fgadr":
            raise SystemExit("--val-source fgadr requires 'fgadr' in --datasets")
        val_source = "idrid"
    if "retlesion" in args.datasets:
        if patch is None:
            log("[warn] retlesion in whole-image mode: IDRiD/FGADR contribute full-"
                "retina resizes while retlesion contributes ~15%-of-retina crops — "
                "the same lesion at incompatible scales. Prefer --patch-size with "
                "this source.")
        from retlesion_dataset import RetLesionDataset, download_retlesion
        rl_root = args.retlesion_root or download_retlesion()
        rl = RetLesionDataset(rl_root, lesions=args.lesions,
                              patch_size=patch or args.image_size,
                              scale_match=True, drop_empty=True,
                              fg_bias=args.fg_bias, augment=True,
                              seed=args.seed + SOURCE_SEED_OFFSET["retlesion"])
        if len(rl):
            train_ds = torch.utils.data.ConcatDataset([train_ds, rl])
            extra_counts[f"retlesion{rl.counts()}"] = len(rl)
    log(f"[info] dist   : {'DDP world='+str(world)+' backend='+dist.get_backend() if ddp else 'single-process'}")
    log(f"[info] device : {device}")
    log(f"[info] lesions: {args.lesions}  (C={C})   loss={args.loss}")
    log(f"[info] splits : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        + (f"   (+{extra_counts})" if extra_counts else "")
        + (f"   [val={val_source}; test=IDRiD+FGADR]" if fgadr_test_ds is not None
           else "   [val/test = IDRiD only]"))
    log(f"[info] train  : {'patch '+str(patch)+'px native-res' if patch else 'whole-image '+str(args.image_size)+'px'}"
        f"   eval={'tiled native-res' if args.eval_tiled else 'whole-image '+str(args.image_size)+'px'}")
    log(f"[info] ckpt   : {ckpt_path}")

    # Everything a reader needs to reproduce or audit this run that vars(args)
    # does not already say: the FGADR partition it was scored on, the eval
    # geometry, the loader regime (persistent workers change make_rng's draw
    # sequence), and the code revision. Written into both the checkpoint and
    # the results JSON so neither can be orphaned from its provenance.
    run_meta = {
        "git_commit": git_commit(),
        "fgadr_split_seed": DEFAULT_SPLIT_SEED,
        "fgadr_val_frac": DEFAULT_VAL_FRAC, "fgadr_test_frac": DEFAULT_TEST_FRAC,
        "val_frac": args.val_frac, "val_source": val_source,
        "eval_tiled": args.eval_tiled, "eval_tiled_val": args.eval_tiled_val,
        "tile_overlap": args.tile_overlap, "tile_batch": args.tile_batch,
        "accum_steps": max(1, args.accum_steps), "fg_bias": args.fg_bias,
        "num_workers": args.num_workers, "persistent_workers": args.num_workers > 0,
        "source_seed_offsets": {k: v for k, v in SOURCE_SEED_OFFSET.items() if k in args.datasets},
        "world_size": world,
    }

    # Under DDP, shard the data across ranks so each GPU sees a different subset.
    # Training uses DistributedSampler (padding is harmless there); eval uses
    # shard_for_eval, because sampler padding double-counts into Dice counts.
    g = torch.Generator(); g.manual_seed(args.seed)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True,
                              worker_init_fn=seed_worker, generator=g,
                              persistent_workers=(args.num_workers > 0),
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(shard_for_eval(val_ds), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            persistent_workers=(args.num_workers > 0),
                            pin_memory=(device.type == "cuda"))

    model, desc = build(args, C, device)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    module = model.module if ddp else model        # unwrapped, for eval/checkpoint
    log(f"[info] model  : {desc}, {sum(p.numel() for p in module.parameters())/1e6:.2f}M params")

    loss_fn = make_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Cosine LR schedule (optional linear warmup) — stepped once per epoch.
    sched = None
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if args.warmup_epochs > 0:
            warm = LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs)
            cos = CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup_epochs))
            sched = SequentialLR(opt, [warm, cos], milestones=[args.warmup_epochs])
        else:
            sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # --- large-effective-batch training ---
    # Effective batch = batch_size * accum_steps * world_size. Gradient
    # accumulation fits a small micro-batch in memory; DDP shards across GPUs.
    accum = max(1, args.accum_steps)
    eff_batch = args.batch_size * accum * world
    amp_on = args.amp and device.type in ("cuda", "mps")
    # MPS has no GradScaler, so fp16 there would backprop UNSCALED — exactly the
    # small-gradient underflow that collapses MA (see --allow-tf32 help). bf16
    # keeps fp32's exponent range and needs no loss scaling.
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_on and device.type == "cuda"))
    log(f"[info] batch  : physical={args.batch_size} x accum={accum} x gpus={world} "
        f"-> effective={eff_batch}   amp={'on(' + str(amp_dtype).split('.')[-1] + ')' if amp_on else 'off'}")

    def optimizer_step():
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    csv_log = CSVLogger(os.path.join(args.ckpt_dir, f"{run_name}_metrics.csv")) if is_main else None
    best_val, best_epoch, since_best = -1.0, -1, 0
    log(f"\n=== training up to {args.epochs} epochs "
        f"(val selects best; early stop patience={args.patience or 'off'}) ===")
    for epoch in range(1, args.epochs + 1):
        if ddp:
            train_sampler.set_epoch(epoch)          # reshuffle shards each epoch
        model.train()
        running, nb = 0.0, 0
        steps = len(train_loader)
        opt.zero_grad(set_to_none=True)
        for i, (imgs, masks, valid) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            is_step = ((i + 1) % accum == 0) or (i + 1 == steps)
            # During accumulation, skip DDP's gradient all-reduce except on the
            # step that actually updates -> avoids redundant cross-GPU traffic.
            sync_ctx = model.no_sync() if (ddp and not is_step) else nullcontext()
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                    loss = loss_fn(model(imgs), masks, valid) / accum
                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            nb += 1
            # On-device accumulation: .item() here would force a GPU sync every
            # micro-batch, serialising CPU and GPU.
            running = running + loss.detach() * accum
            if is_step:
                optimizer_step()
        running = float(running)

        if args.eval_tiled_val:
            val_dice, val_iou, val_sens, val_prec = evaluate_tiled(
                module, val_ds, args.lesions, device, patch,   # patch > 0 enforced at parse
                args.tile_overlap, tile_batch=args.tile_batch,
                num_workers=args.num_workers)                  # all-reduced inside
        else:
            val_dice, val_iou, val_sens, val_prec = evaluate_whole(
                module, val_loader, args.lesions, device)      # all-reduced inside
        mean_val = mean_present(val_dice)
        # Dice alone cannot distinguish "predicts nothing" from "predicts
        # everything"; sensitivity next to it diagnoses collapse (losses.py).
        per = "  ".join(f"{k}={v:.3f}/s{val_sens[k]:.2f}" for k, v in val_dice.items())
        flag = ""
        if mean_val > best_val:                     # identical on all ranks (reduced metric)
            best_val, best_epoch, since_best = mean_val, epoch, 0
            if is_main:                             # only rank 0 writes the checkpoint
                torch.save({"model": module.state_dict(), "epoch": epoch, "val_mean_dice": mean_val,
                            "val_dice": val_dice, "val_iou": val_iou, "lesions": args.lesions,
                            "arch": desc, "args": vars(args), **run_meta}, ckpt_path)
            flag = "  <- best (saved)"
        else:
            since_best += 1
        train_loss = running / max(nb, 1)
        log(f"  epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
            f"val[mean={mean_val:.3f}]  {per}{flag}")
        if csv_log is not None:
            row = {"epoch": epoch, "train_loss": round(train_loss, 5),
                   "val_mean_dice": round(mean_val, 5),
                   "lr": round(opt.param_groups[0]["lr"], 6)}
            row.update({f"val_dice_{k}": round(v, 5) for k, v in val_dice.items()})
            row.update({f"val_sens_{k}": round(v, 5) for k, v in val_sens.items()})
            row.update({f"val_prec_{k}": round(v, 5) for k, v in val_prec.items()})
            csv_log.log(row)
        if sched is not None:
            sched.step()
        if args.patience and since_best >= args.patience:
            log(f"  early stop: no val improvement for {args.patience} epochs "
                f"(best {best_val:.3f} @ epoch {best_epoch})")
            break

    # ---- final honest eval: best checkpoint on held-out TEST ----
    if ddp:
        dist.barrier()                              # ensure rank 0 finished writing the ckpt
    log(f"\n[info] best val mean-Dice {best_val:.3f} @ epoch {best_epoch}; loading {ckpt_path}")
    module.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    def score(ds):
        """Score a held-out set with the configured eval mode (tiled or whole)."""
        if args.eval_tiled:
            # patch is guaranteed non-None here: --eval-tiled without
            # --patch-size is rejected at argument parsing (see the SystemExit).
            return evaluate_tiled(module, ds, args.lesions, device, patch, args.tile_overlap,
                                  tile_batch=args.tile_batch, num_workers=args.num_workers)
        loader = DataLoader(shard_for_eval(ds), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker)
        return evaluate_whole(module, loader, args.lesions, device)

    def report(name, dice, iou, sens, prec):
        md, mi = mean_present(dice), mean_present(iou)
        log(f"\n=== TEST — {name} (best checkpoint) ===")
        log(f"  Dice [mean={md:.3f}]   " + "  ".join(f"{k}={v:.3f}" for k, v in dice.items()))
        log(f"  IoU  [mean={mi:.3f}]   " + "  ".join(f"{k}={v:.3f}" for k, v in iou.items()))
        log("  Sens/Prec              " + "  ".join(f"{k}={sens[k]:.3f}/{prec[k]:.3f}"
                                                    for k in dice))
        return md, mi

    test_dice, test_iou, test_sens, test_prec = score(test_ds)
    mean_test, mean_iou = report(f"IDRiD ({len(test_ds)} imgs)", test_dice, test_iou,
                                 test_sens, test_prec)

    # FGADR's own split: matched-distribution and large enough for stable
    # per-lesion numbers. IDRiD's 27 images then read as an out-of-distribution
    # generalisation check. They answer different questions — report both.
    fgadr_dice = fgadr_iou = fgadr_sens = fgadr_prec = None
    if fgadr_test_ds is not None:
        fgadr_dice, fgadr_iou, fgadr_sens, fgadr_prec = score(fgadr_test_ds)
        report(f"FGADR ({len(fgadr_test_ds)} imgs)", fgadr_dice, fgadr_iou,
               fgadr_sens, fgadr_prec)
        log("\n[note] IDRiD is the out-of-distribution check here (27 images, noisy); "
            "FGADR is the matched-distribution number.")

    log("\nOK: train/val/test separated; focal-tversky loss; best checkpoint scored on test.")

    if csv_log is not None:
        csv_log.close()
    if is_main and args.results_json:
        import json
        # Create the parent dir: this write is the very last thing the run does,
        # so a missing directory would discard a full training run's results.
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump({
                "arch": args.arch,
                "use_gcg": (args.arch == "gcg_unet" and not args.no_gcg),
                "gcg_variant": (args.gcg_variant if args.arch == "gcg_unet" and not args.no_gcg else None),
                # Architecture identity, so a results file is self-describing and
                # an arch sweep can group runs without re-deriving it from flags.
                "encoder": args.encoder, "decoder": args.decoder,
                "lateral_channels": args.lateral_channels,
                "params_m": round(sum(p.numel() for p in module.parameters()) / 1e6, 4),
                "seed": args.seed, "epochs": args.epochs,
                "image_size": args.image_size, "patch_size": patch,
                "lr": args.lr, "loss": args.loss, "lesions": args.lesions,
                "datasets": args.datasets,
                "allow_tf32": args.allow_tf32, "amp": args.amp,
                "batch_size": args.batch_size,
                "effective_batch": args.batch_size * max(1, args.accum_steps) * world,
                "pretrained": not getattr(args, "no_pretrained", False),
                "init_encoder": args.init_encoder, "init_weights": args.init_weights,
                **run_meta,                       # split seed, eval geometry, loader regime, commit
                "test_dice": test_dice, "test_mean": mean_test,
                "test_iou": test_iou, "test_iou_mean": mean_iou,
                "test_sens": test_sens, "test_prec": test_prec,
                "fgadr_test_dice": fgadr_dice,
                "fgadr_test_mean": mean_present(fgadr_dice) if fgadr_dice else None,
                "fgadr_test_iou": fgadr_iou,
                "fgadr_test_sens": fgadr_sens, "fgadr_test_prec": fgadr_prec,
                "best_val": best_val, "best_epoch": best_epoch,
            }, f, indent=2)
        log(f"[info] wrote results -> {args.results_json}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
