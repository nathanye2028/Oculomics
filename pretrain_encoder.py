"""
pretrain_encoder.py
===================
**In-domain encoder pretraining** on a combined fundus corpus
(DDR grading + RFMiD multi-disease), producing an encoder that drops straight
into GCG-U-Net for IDRiD lesion segmentation.

Why this is the highest-value lever
-----------------------------------
Segmentation has only ~101 masked images. Initialisation therefore dominates:
ImageNet transfer alone moved mean Dice 0.183 -> 0.387. This script replaces
"features learned on natural images" with "features learned on ~14,400
retinas":

    DDR    12,521 images, ICDR grade 0-4      (multiclass, CE)
    RFMiD   1,920 images, 46 disease labels   (multi-label, BCE + pos_weight)

The two corpora have *different label spaces*, so this is proper **multi-task
pretraining**: one shared encoder, one lightweight head per corpus. A batch may
mix sources; the encoder runs once and each source's rows are routed to their
own head, with the losses summed.

Chain:
    python pretrain_encoder.py --datasets ddr rfmid --epochs 20 \
        --out weights/fundus_encoder.pt
    python run_experiment.py ... --init-encoder weights/fundus_encoder.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import MobileNetV3Encoder                 # noqa: E402
from fundus_utils import seed_everything, seed_worker    # noqa: E402
from metrics import CSVLogger                            # noqa: E402

DDR, RFMID = "ddr", "rfmid"


class _Tagged(Dataset):
    """Wrap a corpus so each item carries its source id: (x, y, source)."""

    def __init__(self, ds: Dataset, source: int, multilabel: bool):
        self.ds, self.source, self.multilabel = ds, source, multilabel

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        y = y.float() if self.multilabel else torch.as_tensor(y).long()
        return x, y, self.source


def _collate(batch):
    """Keep per-source label shapes separate (they differ across corpora)."""
    xs = torch.stack([b[0] for b in batch])
    srcs = torch.tensor([b[2] for b in batch])
    ys = [b[1] for b in batch]
    return xs, ys, srcs


class MultiCorpusPretrainer(nn.Module):
    """Shared MobileNetV3 encoder + one classification head per corpus."""

    def __init__(self, head_dims: Dict[int, int], pretrained: bool = True):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        feat = self.encoder.out_channels[-1]              # 960
        self.heads = nn.ModuleDict(
            {str(k): nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, d))
             for k, d in head_dims.items()})

    def features(self, x: torch.Tensor) -> torch.Tensor:
        deep = self.encoder(x)[-1]
        return F.adaptive_avg_pool2d(deep, 1).flatten(1)

    def forward(self, x: torch.Tensor, source: int) -> torch.Tensor:
        return self.heads[str(source)](self.features(x))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_corpora(args):
    """Return (concat_dataset, head_dims, meta) for the requested corpora."""
    parts, head_dims, meta = [], {}, {}
    if DDR in args.datasets:
        from ddr_dataset import DDRGradingDataset, download_ddr
        root = args.ddr_root or download_ddr()
        ds = DDRGradingDataset(root, image_size=args.image_size, augment=True,
                               seed=args.seed, limit=args.limit)
        parts.append(_Tagged(ds, 0, multilabel=False))
        head_dims[0] = ds.num_classes
        meta["ddr"] = len(ds)
    if RFMID in args.datasets:
        from rfmid_dataset import RFMiDDataset, download_rfmid
        root = args.rfmid_root or download_rfmid(args.rfmid_slug)
        ds = RFMiDDataset(root, "train", image_size=args.image_size,
                          augment=True, seed=args.seed)
        parts.append(_Tagged(ds, 1, multilabel=True))
        head_dims[1] = ds.num_classes
        meta["rfmid"] = len(ds)
        meta["_rfmid_posw"] = ds.pos_weight()
    if not parts:
        raise SystemExit("No corpora selected (--datasets ddr rfmid)")
    return ConcatDataset(parts), head_dims, meta


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain the segmentation encoder on fundus corpora.")
    p.add_argument("--datasets", nargs="+", default=[DDR, RFMID], choices=[DDR, RFMID])
    p.add_argument("--ddr-root", default=None)
    p.add_argument("--rfmid-root", default=None)
    p.add_argument("--rfmid-slug", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Cap DDR images (debug).")
    p.add_argument("--max-steps", type=int, default=0, help="Cap steps/epoch (smoke runs).")
    p.add_argument("--no-imagenet", action="store_true")
    p.add_argument("--out", default="weights/fundus_encoder.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    corpus, head_dims, meta = build_corpora(args)
    posw = meta.get("_rfmid_posw")
    if posw is not None:
        posw = posw.to(device)
    print(f"[info] device : {device}")
    print(f"[info] corpora: " + "  ".join(f"{k}={v}" for k, v in meta.items() if not k.startswith("_")))
    print(f"[info] total  : {len(corpus)} images   heads={ {k: v for k, v in head_dims.items()} }")

    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(corpus, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, worker_init_fn=seed_worker,
                        generator=g, drop_last=True, collate_fn=_collate,
                        pin_memory=(device.type == "cuda"))

    model = MultiCorpusPretrainer(head_dims, pretrained=not args.no_imagenet).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    csv_log = CSVLogger(os.path.splitext(args.out)[0] + "_metrics.csv")

    print(f"\n=== pretraining {args.epochs} epochs ===")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for step, (x, ys, srcs) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            feats = model.features(x)                     # encoder runs ONCE per batch
            loss = feats.new_zeros(())
            for s in srcs.unique():                       # route rows to their head
                m = (srcs == s)
                logits = model.heads[str(int(s))](feats[m])
                y = torch.stack([ys[i] for i in m.nonzero(as_tuple=True)[0].tolist()]).to(device)
                if int(s) == 0:                           # DDR: multiclass
                    loss = loss + F.cross_entropy(logits, y)
                else:                                     # RFMiD: multi-label
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=posw)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
            if args.max_steps and step >= args.max_steps:
                break
        avg = run / max(nb, 1)
        tag = ""
        if avg < best:
            best = avg
            torch.save({"encoder": model.encoder.state_dict(), "epoch": epoch,
                        "train_loss": avg, "corpora": {k: v for k, v in meta.items()
                                                       if not k.startswith("_")},
                        "args": vars(args)}, args.out)
            tag = "  <- best (encoder saved)"
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}{tag}")
        csv_log.log({"epoch": epoch, "train_loss": round(avg, 5),
                     "lr": round(opt.param_groups[0]["lr"], 6)})
        sched.step()
    csv_log.close()

    print(f"\nbest loss {best:.4f}  ->  encoder at {args.out}")
    print(f"next:  python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 "
          f"--eval-tiled --pretrained --init-encoder {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
