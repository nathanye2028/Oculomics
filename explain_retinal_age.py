#!/usr/bin/env python3
"""
explain_retinal_age.py
======================
What does the retinal age clock look at? For a strip of retinas across the age
range this writes, per image, a **Grad-CAM** overlay (which regions push the
predicted age up) and, when the checkpoint carries a Guided Context Gating
block (``--gcg`` on the MobileNetV3-Small trunk), the **gate's spatial map**
(which stride-16 regions the deep context lets through), captured with the
project's ``record_gcg_gates`` / ``collect_gcg_gates`` API. Output: one grid PNG
(rows = images by age, columns = image | Grad-CAM | gate), per-image overlays,
and a CSV of where the attention mass sits (central vs peripheral retina).

Grad-CAM for a regression: the "class score" is the predicted age, so the map
highlights evidence for *older*. It is taken on the deepest spatial feature map
of the trunk (the last 4-D activation before pooling), for both the MobileNetV3
and timm (MobileNetV4) paths, without touching model code: forward hooks.

    python explain_retinal_age.py --checkpoint ck_retinal_age/student_seed0.pt \\
        --root <BRSET> --predictions exp_retinal_age/predictions_pooled.csv --out exp_retinal_age/explain
    python explain_retinal_age.py --checkpoint ck_retinal_age/gcg_seed0.pt --root <BRSET> \\
        --predictions exp_retinal_age/predictions_pooled.csv --out exp_retinal_age/explain_gcg --per-bin 1

Rendering uses PIL only (no matplotlib on every box).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset                                          # noqa: E402
from train_retinal_age import AGE_BIN_LABELS, age_bin, load_age_model    # noqa: E402


# --------------------------------------------------------------------------- #
# Attention maps
# --------------------------------------------------------------------------- #
def grad_cam(model, x: torch.Tensor) -> Tuple[np.ndarray, float, str]:
    """Grad-CAM of the predicted age on the deepest spatial activation of the trunk.
    Returns (map in [0,1] at the activation's resolution, predicted age in years, sign):
    sign "pos" is the standard ReLU map (evidence for *older*); when that map is
    identically zero (every region pulls the age down, e.g. a young retina or an
    untrained model) the absolute map is used instead and sign is "abs"."""
    acts: List[torch.Tensor] = []

    def hook(_m, _i, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[-1] > 1 and out.shape[-2] > 1:
            acts.append(out)

    handles = [m.register_forward_hook(hook) for m in model.net.modules() if len(list(m.children())) == 0]
    try:
        model.eval()
        tta = model.tta
        model.tta = False
        years = model(x)
        model.tta = tta
    finally:
        for h in handles:
            h.remove()
    if not acts:
        raise RuntimeError("no 4-D activation found in the trunk; cannot compute Grad-CAM")
    act = acts[-1]
    grad = torch.autograd.grad(years.sum(), act, retain_graph=False)[0]
    w = grad.mean(dim=(2, 3), keepdim=True)                       # [1,C,1,1]
    raw = (w * act).sum(dim=1)[0].detach()                        # [h,w]
    cam, sign = torch.relu(raw), "pos"
    if float(cam.max()) <= 0:
        cam, sign = raw.abs(), "abs"
    cam = cam - cam.min()
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.cpu().numpy().astype(np.float32), float(years.detach().item()), sign


@torch.no_grad()
def gate_map(model, x: torch.Tensor) -> Optional[np.ndarray]:
    """Spatial map of the GCG gate (mean over gates that expose one), in [0,1]; None if no gate."""
    if getattr(model.net, "gcg", None) is None:
        return None
    try:
        from model_seg import record_gcg_gates, collect_gcg_gates
    except ImportError:
        return None
    tta = model.tta
    model.tta = False
    with record_gcg_gates(model.net):
        model(x)
    model.tta = tta
    maps = []
    for _name, g in collect_gcg_gates(model.net).items():
        sp = g.get("spatial")
        if sp is not None:
            maps.append(sp[0, 0].float().cpu().numpy())
    if not maps:
        return None
    m = np.mean(maps, axis=0).astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi > lo else m * 0


# --------------------------------------------------------------------------- #
# Rendering (PIL only)
# --------------------------------------------------------------------------- #
def colormap(m: np.ndarray) -> np.ndarray:
    """[h,w] in [0,1] -> [h,w,3] uint8, a blue-cyan-yellow-red ramp."""
    m = np.clip(m, 0, 1)
    r = np.clip(1.5 * m - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(2.0 * m - 1.0) * 1.2 + 0.2, 0, 1)
    b = np.clip(1.0 - 1.5 * m, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(img: Image.Image, m: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = Image.fromarray(colormap(m)).resize(img.size, Image.BILINEAR)
    return Image.blend(img.convert("RGB"), heat, alpha)


def _font(size: int):
    for cand in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/Library/Fonts/Arial.ttf"):
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                pass
    return ImageFont.load_default()


def central_fraction(m: np.ndarray, frac: float = 0.5) -> float:
    """Share of the map's mass inside the central ``frac`` of each axis (0.25 = uniform).
    Computed on a 64x64 resample so tiny maps (a 2x2 activation) are not quantised."""
    r = np.asarray(Image.fromarray(np.clip(m, 0, 1).astype(np.float32)).resize((64, 64), Image.BILINEAR))
    h, w = r.shape
    y0, y1 = int(h * (1 - frac) / 2), int(h * (1 + frac) / 2)
    x0, x1 = int(w * (1 - frac) / 2), int(w * (1 + frac) / 2)
    tot = float(r.sum())
    return float(r[y0:y1, x0:x1].sum() / tot) if tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Image selection
# --------------------------------------------------------------------------- #
def pick_rows(pred: pd.DataFrame, dataset: str, per_bin: int, pick: str, seed: int,
              condition: Optional[str] = None) -> pd.DataFrame:
    d = pred.copy()
    if "condition" in d.columns:
        conds = list(pd.unique(d["condition"]))
        c = condition or ("student" if "student" in conds else conds[0])
        d = d[d["condition"] == c]
    d = d[(d["dataset"].astype(str) == dataset) & d["age"].notna() & d["pred_age"].notna()].copy()
    if d.empty:
        raise SystemExit(f"[fatal] no rows for dataset {dataset!r} in the predictions table")
    d["bin"] = age_bin(d["age"].to_numpy())
    d["abs_gap"] = (d["pred_age"] - d["age"]).abs()
    rng = np.random.default_rng(seed)
    out = []
    for b in range(len(AGE_BIN_LABELS)):
        f = d[d["bin"] == b]
        if f.empty:
            continue
        if pick == "typical":
            f = f.sort_values("abs_gap")
        elif pick == "worst":
            f = f.sort_values("abs_gap", ascending=False)
        else:
            f = f.iloc[rng.permutation(len(f))]
        out.append(f.head(per_bin))
    return pd.concat(out).sort_values("age").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Grad-CAM (+ GCG gate maps) for a retinal age clock.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True, help="Dataset root holding the images (BRSET or mBRSET).")
    p.add_argument("--dataset", default="brset", choices=["brset", "mbrset"])
    p.add_argument("--predictions", default=None,
                   help="predictions CSV (per run or pooled) to pick images from by age bin.")
    p.add_argument("--condition", default=None, help="Pooled CSV: which condition's rows (default student).")
    p.add_argument("--files", nargs="*", default=None, help="Explicit image files instead of --predictions.")
    p.add_argument("--per-bin", type=int, default=1)
    p.add_argument("--pick", default="typical", choices=["typical", "random", "worst"],
                   help="Within each age bin: smallest |gap| (typical), random, or largest |gap|.")
    p.add_argument("--bn-stats", default="source", choices=["source", "adapted"])
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="explain")
    args = p.parse_args()

    from brset_dataset import load_any                                   # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load_age_model(args.checkpoint, device, bn_stats=args.bn_stats)
    a = ck.get("args", {})
    size = int(a.get("image_size", 384))
    src = load_any(args.root, args.dataset, image_ext=args.image_ext)
    df = src["df"]
    if "age" not in df.columns:
        raise SystemExit("[fatal] the dataset CSV has no age column")

    if args.files:
        rows = df[df["file"].astype(str).isin(args.files)].copy()
        rows["pred_age"] = np.nan
    elif args.predictions:
        rows = pick_rows(pd.read_csv(args.predictions), args.dataset, args.per_bin, args.pick, args.seed,
                         args.condition)
        rows = rows.merge(df[["file"]], on="file", how="inner")
    else:
        raise SystemExit("[fatal] pass --predictions <csv> or --files ...")
    rows = rows.dropna(subset=["age"]).reset_index(drop=True)
    if rows.empty:
        raise SystemExit("[fatal] no images selected")
    ds = MBRSETDataset(csv=rows, images_dir=src["images_dir"], task="age", split="val",
                       image_size=size, drop_missing_files=True, fov_crop=True)
    if len(ds) == 0:
        raise SystemExit(f"[fatal] none of the selected files exist under {src['images_dir']}")
    os.makedirs(args.out, exist_ok=True)
    has_gate = getattr(model.net, "gcg", None) is not None
    font, small = _font(22), _font(16)
    tile = 320
    cols = 3 if has_gate else 2
    top = 34                                                     # header strip for the column labels
    grid = Image.new("RGB", (cols * tile + (cols + 1) * 12, top + len(ds) * (tile + 44) + 12), "white")
    d0 = ImageDraw.Draw(grid)
    for j, lab in enumerate(["image", "Grad-CAM (evidence for older)",
                             "GCG gate (what the deep context lets through)"][:cols]):
        d0.text((12 + j * (tile + 12), 8), lab, fill=(108, 117, 125), font=small)
    stats = []
    for i in range(len(ds)):
        sample = ds[i]
        x = sample["image"][None].to(device)
        age = float(sample["label"])
        cam, pred, sign = grad_cam(model, x)
        gm = gate_map(model, x) if has_gate else None
        img = ds._load_image(i).convert("RGB").resize((tile, tile), Image.BILINEAR)
        panels = [img, overlay(img, cam)] + ([overlay(img, gm)] if gm is not None else [])
        y = top + 12 + i * (tile + 44)
        for j, pn in enumerate(panels):
            grid.paste(pn, (12 + j * (tile + 12), y))
        d = ImageDraw.Draw(grid)
        d.text((12, y + tile + 4), f"{ds.files[i]}   age {age:.0f}   predicted {pred:.1f}   gap {pred - age:+.1f} y",
               fill=(43, 45, 66), font=font)
        overlay(img, cam).save(os.path.join(args.out, f"{os.path.splitext(ds.files[i])[0]}_cam.png"))
        if gm is not None:
            overlay(img, gm).save(os.path.join(args.out, f"{os.path.splitext(ds.files[i])[0]}_gate.png"))
        stats.append({"file": ds.files[i], "age": age, "pred_age": round(pred, 2), "gap": round(pred - age, 2),
                      "cam_sign": sign, "cam_central_half": round(central_fraction(cam), 3),
                      "gate_central_half": round(central_fraction(gm), 3) if gm is not None else "",
                      "cam_hw": f"{cam.shape[0]}x{cam.shape[1]}"})
    grid_path = os.path.join(args.out, "explain_grid.png")
    grid.save(grid_path)
    with open(os.path.join(args.out, "explain_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys())); w.writeheader(); w.writerows(stats)
    print(f"[info] {len(stats)} images; backbone {a.get('backbone')}, gcg={'on' if has_gate else 'off'}")
    print(f"[info] wrote {grid_path} and explain_stats.csv (+ per-image overlays)")
    cf = np.array([s["cam_central_half"] for s in stats], dtype=float)
    print(f"[info] Grad-CAM mass in the central half of the image: mean {np.nanmean(cf):.2f} "
          f"(0.25 would be uniform; higher = disc / macula, lower = periphery)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
