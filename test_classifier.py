"""
test_classifier.py
==================
End-to-end smoke test for the MobileNetV3-Small clinical-label classifier
(:mod:`model`) wired to the mBRSET data pipeline (:mod:`dataset`).

Pulls one batch of real fundus images through ``MBRSETDataset``, runs the
classifier, prints tensor shapes, and takes a few optimisation steps to prove
the forward/backward path works. Uses the public Kaggle DR dataset
(``mohamedabdalkader/retinal-disease-detection``) remapped onto the mBRSET
clinical-label schema.

Run:
    python test_classifier.py                 # dr_grade (5-class)
    python test_classifier.py --task dr_referable --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset      # noqa: E402
from model import MBRSETClassifier     # noqa: E402

KAGGLE_SLUG = "mohamedabdalkader/retinal-disease-detection"
COLUMN_MAP = {
    "Image name": "file",
    "Retinopathy grade": "final_icdr",
    "Risk of macular edema": "final_edema",
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(description="MobileNetV3-Small clinical classifier smoke test.")
    p.add_argument("--task", default="dr_grade")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pretrained", action="store_true", default=True)
    args = p.parse_args()

    device = pick_device()
    import kagglehub
    root = kagglehub.dataset_download(KAGGLE_SLUG)
    split = os.path.join(root, "Diabetic Retinopathy", "train")
    df = pd.read_csv(os.path.join(split, "annotations.csv")).rename(columns=COLUMN_MAP)

    ds = MBRSETDataset(
        csv=df, images_dir=os.path.join(split, "images"),
        task=args.task, split="train", image_size=args.image_size,
        drop_missing_files=True,
    )
    print(f"[info] device : {device}")
    print(f"[info] dataset: {ds}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = MBRSETClassifier.from_dataset(ds, pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model  : MobileNetV3-Small classifier, {n_params/1e6:.2f}M params, "
          f"out_classes={model.num_classes}, multilabel={model.multilabel}")

    batch = next(iter(loader))
    imgs, labels = batch["image"].to(device), batch["label"].to(device)
    logits = model(imgs)
    print("\n=== one batch ===")
    print(f"image : shape={tuple(imgs.shape)} dtype={imgs.dtype}")
    print(f"label : shape={tuple(labels.shape)} dtype={labels.dtype} values={labels.tolist()}")
    print(f"logits: shape={tuple(logits.shape)} dtype={logits.dtype}")
    assert logits.shape[0] == imgs.shape[0]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    print(f"\n=== {args.steps} optimisation steps ===")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = F.cross_entropy(out, labels)
        loss.backward(); opt.step()
        acc = (out.argmax(1) == labels).float().mean().item()
        print(f"  step {step}: loss={loss.item():.4f}  train-acc={acc:.3f}")

    print("\nOK: classifier predicts clinical labels through the mBRSET pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
