#!/usr/bin/env python3
"""
score_external.py
=================
Score a finished ``train_mbrset.py`` checkpoint on ANOTHER dataset, zero-shot
and (optionally) after label-free AdaBN -- without retraining anything.

``train_mbrset.py`` takes exactly one ``--external-test-root``; the public
glaucoma sets give several (REFUGE, PAPILA, ODIR-5K), and a model trained once
should be scored on all of them. This writes one JSON per (checkpoint, dataset)
in the same shape as the trainer's ``external`` / ``external_bnadapt`` blocks so
``summarize_external.py`` can pair treatment-vs-control per external set.

    python score_external.py --ckpt ck_glaucoma/kd_seed0.pt --root <PAPILA> --dataset papila \\
        --bn-adapt --out exp_glaucoma/kd_seed0_on_papila.json

Refuses to score a checkpoint on the directory it was trained on (that is a
training-set number, not an external one). Uses the zero-shot ``model`` weights;
AdaBN statistics are re-estimated on THIS dataset's images, transductively.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brset_dataset import DATASETS, load_any                          # noqa: E402
from dataset import MBRSETDataset                                     # noqa: E402
from fundus_utils import seed_worker                                  # noqa: E402
from model import MBRSETClassifier                                    # noqa: E402
from train_mbrset import adapt_bn, backbone_kwargs_for, evaluate, pick_device  # noqa: E402


def _head_out_dim(state_dict) -> int:
    ws = [v for k, v in state_dict.items() if k.startswith("head.") and k.endswith(".weight") and v.ndim == 2]
    if not ws:
        raise SystemExit("[fatal] checkpoint has no head weights")
    return int(ws[-1].shape[0])


def score_checkpoint(ckpt: str, root: str, dataset: str, task: str = None, bn_adapt: bool = False,
                     bn_adapt_batches: int = 0, batch_size: int = 32, num_workers: int = 0,
                     image_ext: str = ".jpg", device: torch.device = None, seed: int = 0,
                     **loader_kw) -> dict:
    ck = torch.load(ckpt, map_location="cpu")
    a = ck.get("args", {}) or {}
    task = task or a.get("task")
    if task is None:
        raise SystemExit("[fatal] --task not given and the checkpoint records none")
    if a.get("task") and a["task"] != task:
        raise SystemExit(f"[fatal] checkpoint was trained for task {a['task']!r}, asked to score {task!r}")
    if a.get("root") and os.path.realpath(a["root"]) == os.path.realpath(root):
        raise SystemExit(f"[fatal] {root} is the checkpoint's own training root; that is not an "
                         f"external number.")
    device = device or pick_device()
    src = load_any(root, dataset, image_ext=image_ext, **loader_kw)
    size = int(a.get("image_size", 224))
    ds = MBRSETDataset(csv=src["df"], images_dir=src["images_dir"], task=task, split="val",
                       image_size=size, drop_missing_files=True, fov_crop=True)
    if len(ds) == 0:
        raise SystemExit(f"[fatal] 0 usable images for task {task!r} in {dataset} @ {root}")
    C = ds.num_classes
    if C != _head_out_dim(ck["model"]):
        raise SystemExit(f"[fatal] {dataset} task {task!r} has {C} classes; the checkpoint head has "
                         f"{_head_out_dim(ck['model'])}.")
    bk = a.get("backbone", "mobilenetv3_small")
    use_gcg = ck.get("use_gcg")
    if use_gcg is None:
        use_gcg = (not a.get("no_gcg", False)) and not bk.startswith("timm:")
    model = MBRSETClassifier(num_classes=C, pretrained=False, use_gcg=use_gcg,
                             gcg_variant=a.get("gcg_variant", "baseline"), backbone=bk,
                             backbone_kwargs=backbone_kwargs_for(bk, size))
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        worker_init_fn=seed_worker, pin_memory=(device.type == "cuda"))
    em = evaluate(model, loader, device, C)
    out = {"ckpt": os.path.abspath(ckpt), "task": task, "dataset": dataset, "root": os.path.abspath(root),
           "train_dataset": a.get("dataset"), "train_root": a.get("root"), "seed": a.get("seed"),
           "backbone": bk, "image_size": size, "n_external": len(ds),
           "external_pos": int((ds.labels == 1).sum()) if C == 2 else None,
           "external": em, "external_bnadapt": None, "bn_layers": None, "bn_adapt_transductive": None}
    if bn_adapt:
        bn_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                               num_workers=num_workers, worker_init_fn=seed_worker)
        adapted, n_bn = adapt_bn(model, bn_loader, device, max_batches=bn_adapt_batches)
        out["bn_layers"] = n_bn
        if n_bn:
            out["external_bnadapt"] = evaluate(adapted, loader, device, C)
            out["bn_adapt_transductive"] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Score a checkpoint on an external dataset.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--task", default=None, help="Default: the checkpoint's task.")
    p.add_argument("--out", required=True, help="Results JSON to write.")
    p.add_argument("--bn-adapt", action="store_true")
    p.add_argument("--bn-adapt-batches", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--papila-suspect", default="exclude", choices=["exclude", "positive", "negative"])
    p.add_argument("--airogs-release", default=None, help="AIROGS-light: release-* dir (default release-crop).")
    args = p.parse_args()
    kw = {"papila_suspect": args.papila_suspect} if args.dataset == "papila" else {}
    if args.dataset == "airogs" and args.airogs_release:
        kw["airogs_release"] = args.airogs_release
    r = score_checkpoint(args.ckpt, args.root, args.dataset, task=args.task, bn_adapt=args.bn_adapt,
                         bn_adapt_batches=args.bn_adapt_batches, batch_size=args.batch_size,
                         num_workers=args.num_workers, image_ext=args.image_ext, **kw)
    e, b = r["external"], r.get("external_bnadapt")
    print(f"[{args.dataset}] {os.path.basename(args.ckpt)}: zero-shot AUROC={e['auroc']:.4f} n={e['n']}"
          + (f"   AdaBN AUROC={b['auroc']:.4f} ({r['bn_layers']} BN layers)" if b else
             ("   (no BN layers; AdaBN is a no-op)" if args.bn_adapt else "")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"[info] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
