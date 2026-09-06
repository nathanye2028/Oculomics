#!/usr/bin/env python3
"""
save_gcg_maps.py — save the attention maps the GCG gates produce.

Every Guided Context Gating block computes a **spatial** map ([1,h,w], "where
the network attends" on that skip level) and/or a **channel** vector ([C],
"which channels survive"). This script runs a trained checkpoint over images,
captures those maps at every gated level, and writes, per image:

  <out-dir>/<stem>.npz         spatial_<k> (float16, at the gate's own
                               resolution), channel_<k> (float32), plus
                               gate_names / strides / image_hw / lesions and,
                               with --save-pred, the prediction (uint8 0-255)
  <out-dir>/<stem>.png         a panel: image | ground truth (if any) |
                               prediction | one heat-map overlay per gate
  <out-dir>/gate_stats.csv     per image x gate: mean / std / min / max of the
                               spatial gate, the fraction above 0.5, channel
                               gate stats, and — when lesion masks are
                               available — mean attention ON lesion pixels vs
                               OFF them (the "does GCG look at lesions?" number)
  <out-dir>/run.json           what was run, with the gate names / strides

Works for both models that carry a GCG gate:
  * the segmentation U-Net (train_idrid.py checkpoints) — 5 gates for the
    MobileNetV3 encoder: decoders.0-3 (stride 16 -> 2) and up_full (stride 1);
  * the MobileNetV3-Small classifier with --gcg (train_mbrset.py) — 1 gate on
    the stride-16 feature. (timm-backbone students have no gate.)

Examples
--------
    # IDRiD test images at the checkpoint's training size, with ground truth
    python save_gcg_maps.py --checkpoint checkpoints/gcg_seed1.pt --dataset idrid --split test --limit 8

    # native resolution via tiling (the gates are stitched like the prediction)
    python save_gcg_maps.py --checkpoint checkpoints/fgadr_gcg.pt --dataset fgadr --tiled --limit 6

    # any folder of fundus images (no ground truth), seg or cls checkpoint
    python save_gcg_maps.py --checkpoint ck/gcg_seed0.pt --images <mBRSET>/images --limit 16

A checkpoint trained with --no-gcg has nothing to save and the script says so.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundus_utils import _tile_starts, crop_to_fov, pick_device  # noqa: E402
from idrid_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402  (same constants every loader uses)
from model_seg import collect_gcg_gates, gcg_gate_modules, record_gcg_gates  # noqa: E402

FOV_TOL = 12          # same "retina = max(R,G,B) > 12" rule as the datasets
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Lesion overlay colours — shared with eval_fgadr.py's overlays so the two
# kinds of picture read the same way side by side.
try:
    from eval_fgadr import _COLOURS as LESION_COLOURS
except Exception:                                    # pragma: no cover
    LESION_COLOURS = {"MA": (255, 0, 0), "HE": (255, 140, 0), "EX": (255, 255, 0),
                      "SE": (0, 255, 255), "OD": (255, 0, 255)}

# inferno-like colour map, 8 stops, so the script needs no matplotlib
_CMAP_STOPS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def detect_kind(ck: dict) -> str:
    """'cls' for a train_mbrset.py checkpoint, 'seg' for train_idrid.py."""
    a = ck.get("args", {}) if isinstance(ck, dict) else {}
    if "task" in a or "backbone" in a:
        return "cls"
    return "seg"


def load_seg(path: str, device: torch.device):
    from eval_fgadr import load_model
    net, lesions = load_model(path, device, "gcg_unet", None)
    return net, list(lesions)


def load_cls(path: str, ck: dict, device: torch.device):
    from export_coreml import gcg_from_args
    from model import MBRSETClassifier
    from train_mbrset import backbone_kwargs_for
    a = ck.get("args", {})
    backbone = a.get("backbone", "mobilenetv3_small")
    size = int(a.get("image_size", 224))
    use_gcg = gcg_from_args(a)
    variant = a.get("gcg_variant") or "baseline"
    state = ck["model"]
    # infer the head width from the checkpoint rather than trusting a task name
    n_out = [v for k, v in state.items() if k.startswith("head.") and v.ndim == 2][-1].shape[0]
    net = MBRSETClassifier(num_classes=n_out, pretrained=False, use_gcg=use_gcg,
                           gcg_variant=variant, backbone=backbone,
                           backbone_kwargs=backbone_kwargs_for(backbone, size))
    net.load_state_dict(state)
    print(f"[info] classifier: backbone={backbone} size={size} use_gcg={use_gcg}"
          f"{' variant=' + variant if use_gcg else ''} classes={n_out} "
          f"task={a.get('task', '?')}")
    return net.eval().to(device), size


# --------------------------------------------------------------------------- #
# Inference with gate capture
# --------------------------------------------------------------------------- #
def _gates_to_numpy(gates, input_hw: Tuple[int, int]) -> List[dict]:
    """OrderedDict name -> tensors  ->  list of {name, spatial[h,w], channel[C], stride}."""
    out = []
    for name, g in gates.items():
        sp = g.get("spatial")
        ch = g.get("channel")
        entry = {"name": name, "spatial": None, "channel": None, "stride": None}
        if sp is not None:
            sp = sp[0, 0].float().cpu().numpy()
            entry["spatial"] = sp
            entry["stride"] = int(round(input_hw[0] / sp.shape[0]))
        if ch is not None:
            entry["channel"] = ch[0].float().cpu().numpy()
        out.append(entry)
    return out


@torch.inference_mode()
def infer_whole(net, x: torch.Tensor, device: torch.device, kind: str):
    """x: [3,S,S] normalised. Returns (probs, gates). probs: [C,S,S] (seg) or [K] (cls)."""
    with record_gcg_gates(net):
        out = net(x[None].to(device))
        gates = collect_gcg_gates(net)
    if kind == "seg":
        probs = torch.sigmoid(out)[0].float().cpu()
    else:
        probs = torch.softmax(out, dim=-1)[0].float().cpu()
    return probs, _gates_to_numpy(gates, tuple(x.shape[-2:]))


@torch.inference_mode()
def infer_tiled(net, image: torch.Tensor, tile: int, overlap: int, device: torch.device,
                fg_map: Optional[torch.Tensor] = None, tile_batch: int = 8):
    """Native-resolution inference; the gates are stitched exactly like the
    prediction (tile-averaged, FOV-skipped) and returned at each gate's own
    stride so files stay small. image: [3,H,W] normalised."""
    net.eval()
    _, H, W = image.shape
    tile = min(tile, H, W)
    coords = [(y, x) for y in _tile_starts(H, tile, overlap) for x in _tile_starts(W, tile, overlap)]
    if fg_map is not None:
        flags = torch.stack([fg_map[y:y + tile, x:x + tile].any() for (y, x) in coords]).cpu().tolist()
        coords = [c for c, f in zip(coords, flags) if f]
    image = image.to(device)
    weight = torch.zeros(1, H, W, device=device)
    prob_sum = None
    sp_sum: Dict[str, torch.Tensor] = {}
    ch_sum: Dict[str, torch.Tensor] = {}
    strides: Dict[str, int] = {}
    n_tiles = 0
    with record_gcg_gates(net):
        for i in range(0, len(coords), tile_batch):
            chunk = coords[i:i + tile_batch]
            batch = torch.stack([image[:, y:y + tile, x:x + tile] for (y, x) in chunk], dim=0)
            probs = torch.sigmoid(net(batch)).float()
            gates = collect_gcg_gates(net)
            if prob_sum is None:
                prob_sum = torch.zeros(probs.shape[1], H, W, device=device)
            for j, (y, x) in enumerate(chunk):
                prob_sum[:, y:y + tile, x:x + tile] += probs[j]
                weight[:, y:y + tile, x:x + tile] += 1
            for name, g in gates.items():
                if g.get("spatial") is not None:
                    sp = g["spatial"].float()
                    strides.setdefault(name, int(round(tile / sp.shape[-1])))
                    # nearest: a gate pixel stands for an s x s block, and the
                    # area-pool below then returns exactly the gate's own values
                    # wherever the tile grid aligns with the stride grid.
                    up = F.interpolate(sp, size=(tile, tile), mode="nearest")
                    acc = sp_sum.setdefault(name, torch.zeros(1, H, W, device=device))
                    for j, (y, x) in enumerate(chunk):
                        acc[:, y:y + tile, x:x + tile] += up[j]
                if g.get("channel") is not None:
                    ch = g["channel"].float().sum(0)
                    ch_sum[name] = ch_sum[name] + ch if name in ch_sum else ch
            n_tiles += len(chunk)
    if prob_sum is None:                            # whole image was background
        raise RuntimeError("no field-of-view tiles: is this a fundus image?")
    w = weight.clamp(min=1e-6)
    probs = (prob_sum / w).cpu()
    out = []
    for name, _ in gcg_gate_modules(net):
        entry = {"name": name, "spatial": None, "channel": None, "stride": None}
        if name in sp_sum:
            s = strides[name]
            full = (sp_sum[name] / w).cpu()                           # [1,H,W]
            # on CPU: MPS adaptive pooling refuses non-divisible sizes, and a
            # native-res fundus is rarely a multiple of the stride
            small = F.interpolate(full[None], size=(max(1, H // s), max(1, W // s)),
                                  mode="area")[0, 0]
            entry["spatial"] = small.cpu().numpy()
            entry["stride"] = s
        if name in ch_sum:
            entry["channel"] = (ch_sum[name] / max(n_tiles, 1)).cpu().numpy()
        out.append(entry)
    return probs, out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def colormap(v: np.ndarray) -> np.ndarray:
    """[h,w] in 0..1 -> uint8 [h,w,3] (inferno-like)."""
    v = np.clip(np.nan_to_num(v.astype(np.float32)), 0, 1)
    xs = np.linspace(0, 1, len(_CMAP_STOPS))
    return np.stack([np.interp(v, xs, _CMAP_STOPS[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


def _resize_map(m: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(np.clip(m, 0, 1).astype(np.float32))      # float32 -> mode "F"
    return np.asarray(im.resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.float32)


def stretch(gate: np.ndarray) -> np.ndarray:
    """Min-max stretch a map to 0..1 for DISPLAY only (the files keep raw values).
    A barely-trained gate sits at ~0.5 everywhere; stretching shows its
    relative structure instead of one flat colour."""
    lo, hi = float(np.nanmin(gate)), float(np.nanmax(gate))
    return (gate - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(gate)


def overlay_heat(img: np.ndarray, gate: np.ndarray, alpha: float = 0.55,
                 do_stretch: bool = False) -> np.ndarray:
    """Blend a gate map (any resolution) over an RGB uint8 image."""
    if do_stretch:
        gate = stretch(gate)
    heat = colormap(_resize_map(gate, img.shape[:2])).astype(np.float32)
    return ((1 - alpha) * img.astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)


def overlay_masks(img: np.ndarray, masks: np.ndarray, lesions: Sequence[str]) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for c, code in enumerate(lesions):
        m = masks[c].astype(bool)
        if m.any():
            colour = np.array(LESION_COLOURS.get(code, (255, 255, 255)), dtype=np.float32)
            out[m] = 0.35 * out[m] + 0.65 * colour
    return out.clip(0, 255).astype(np.uint8)


def _fit(img: np.ndarray, max_side: int) -> Image.Image:
    im = Image.fromarray(img)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    return im


def render_panel(tiles: List[Tuple[str, np.ndarray]], max_side: int = 512, cols: int = 4) -> Image.Image:
    """Grid of labelled tiles; every tile is scaled so its longer side is ``max_side``."""
    ims = [(label, _fit(a, max_side)) for label, a in tiles]
    tw = max(im.width for _, im in ims)
    th = max(im.height for _, im in ims)
    bar, pad = 22, 6
    cols = max(1, min(cols, len(ims)))
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + bar + pad) + pad), (28, 31, 51))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for k, (label, im) in enumerate(ims):
        r, c = divmod(k, cols)
        x0 = pad + c * (tw + pad)
        y0 = pad + r * (th + bar + pad)
        draw.text((x0 + 4, y0 + 5), label, fill=(255, 255, 255), font=font)
        canvas.paste(im, (x0, y0 + bar))
    return canvas


# --------------------------------------------------------------------------- #
# Statistics + files
# --------------------------------------------------------------------------- #
def gate_stats_rows(stem: str, gates: List[dict], img: np.ndarray,
                    masks: Optional[np.ndarray], lesions: Sequence[str]) -> List[dict]:
    rows = []
    fov_full = img.max(axis=2) > FOV_TOL
    for k, g in enumerate(gates):
        row = {"image": stem, "gate": g["name"], "level": k, "stride": g["stride"],
               "channels": None if g["channel"] is None else int(g["channel"].size)}
        sp = g["spatial"]
        if sp is not None:
            hw = sp.shape
            fov = _resize_map(fov_full.astype(np.float32), hw) > 0.5
            vals = sp[fov] if fov.any() else sp.ravel()
            row.update(sp_mean=float(vals.mean()), sp_std=float(vals.std()),
                       sp_min=float(vals.min()), sp_max=float(vals.max()),
                       sp_frac_gt_0p5=float((vals > 0.5).mean()))
            if masks is not None:
                union = np.zeros(hw, dtype=bool)
                for c, code in enumerate(lesions):
                    m = _resize_map(masks[c].astype(np.float32), hw) > 0
                    union |= m
                    on = sp[m & fov]
                    off = sp[~m & fov]
                    row[f"on_{code}"] = float(on.mean()) if on.size else float("nan")
                    row[f"off_{code}"] = float(off.mean()) if off.size else float("nan")
                    row[f"ratio_{code}"] = (float(on.mean() / max(off.mean(), 1e-6))
                                            if on.size and off.size else float("nan"))
                on = sp[union & fov]; off = sp[~union & fov]
                row["on_any_lesion"] = float(on.mean()) if on.size else float("nan")
                row["off_any_lesion"] = float(off.mean()) if off.size else float("nan")
                row["ratio_any_lesion"] = (float(on.mean() / max(off.mean(), 1e-6))
                                           if on.size and off.size else float("nan"))
        ch = g["channel"]
        if ch is not None:
            row.update(ch_mean=float(ch.mean()), ch_min=float(ch.min()), ch_max=float(ch.max()),
                       ch_frac_gt_0p5=float((ch > 0.5).mean()))
        rows.append(row)
    return rows


def write_npz(path: str, gates: List[dict], image_hw: Tuple[int, int], lesions: Sequence[str],
              probs: Optional[np.ndarray] = None) -> None:
    arrays = {"gate_names": np.array([g["name"] for g in gates]),
              "strides": np.array([g["stride"] if g["stride"] is not None else -1 for g in gates]),
              "image_hw": np.array(image_hw), "lesions": np.array(list(lesions))}
    for k, g in enumerate(gates):
        if g["spatial"] is not None:
            arrays[f"spatial_{k}"] = g["spatial"].astype(np.float16)
        if g["channel"] is not None:
            arrays[f"channel_{k}"] = g["channel"].astype(np.float32)
    if probs is not None:
        arrays["pred"] = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
    np.savez_compressed(path, **arrays)


def gate_label(g: dict, kind: str, stretched: bool = False) -> str:
    bits = [g["name"].replace(".gcg", "")]
    if g["stride"] is not None:
        bits.append(f"stride {g['stride']}")
    if g["channel"] is not None:
        bits.append(f"{g['channel'].size} ch")
    if g["spatial"] is not None:
        bits.append(f"mean {float(np.nanmean(g['spatial'])):.2f}")
    return "GCG " + " · ".join(bits) + (" (stretched)" if stretched else "")


def save_image_outputs(out_dir: str, stem: str, kind: str, img: np.ndarray,
                       masks: Optional[np.ndarray], lesions: Sequence[str],
                       probs: np.ndarray, gates: List[dict], thresh: float,
                       alpha: float, png_size: int, write_png: bool, write_npz_file: bool,
                       save_pred: bool, separate_pngs: bool, class_names: Optional[Sequence[str]] = None,
                       do_stretch: bool = False) -> None:
    if write_npz_file:
        write_npz(os.path.join(out_dir, f"{stem}.npz"), gates, img.shape[:2], lesions,
                  probs if save_pred else None)
    if not write_png:
        return
    tiles: List[Tuple[str, np.ndarray]] = [("image", img)]
    if kind == "seg":
        if masks is not None:
            tiles.append(("ground truth", overlay_masks(img, masks, lesions)))
        pred = (probs > thresh).astype(np.uint8)
        if pred.shape[-2:] != img.shape[:2]:
            pred = np.stack([_resize_map(p.astype(np.float32), img.shape[:2]) > 0.5 for p in pred]).astype(np.uint8)
        tiles.append((f"prediction @ {thresh:g}", overlay_masks(img, pred, lesions)))
    else:
        top = int(np.argmax(probs))
        name = class_names[top] if class_names and top < len(class_names) else f"class {top}"
        tiles[0] = (f"image · p({name}) = {float(probs[top]):.3f}", img)
    for g in gates:
        if g["spatial"] is not None:
            tiles.append((gate_label(g, kind, do_stretch), overlay_heat(img, g["spatial"], alpha, do_stretch)))
    render_panel(tiles, max_side=png_size).save(os.path.join(out_dir, f"{stem}.png"))
    if separate_pngs:
        for k, g in enumerate(gates):
            if g["spatial"] is not None:
                Image.fromarray(overlay_heat(img, g["spatial"], alpha, do_stretch)).save(
                    os.path.join(out_dir, f"{stem}_gate{k}_stride{g['stride']}.png"))
                raw = stretch(g["spatial"]) if do_stretch else g["spatial"]
                Image.fromarray(colormap(raw)).resize(img.shape[1::-1], Image.NEAREST).save(
                    os.path.join(out_dir, f"{stem}_gate{k}_stride{g['stride']}_raw.png"))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def iter_image_dir(path: str, limit: Optional[int]):
    files = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise SystemExit(f"[fatal] no images ({', '.join(IMG_EXTS)}) in {path}")
    for f in files[:limit]:
        img = np.asarray(Image.open(f).convert("RGB"))
        img, _ = crop_to_fov(img, tol=FOV_TOL)
        yield os.path.splitext(os.path.basename(f))[0], img, None


def iter_dataset(name: str, root: Optional[str], split: str, image_size: int,
                 lesions: Sequence[str], limit: Optional[int]):
    if name == "idrid":
        from idrid_dataset import IDRiDSegDataset
        if root is None:
            import kagglehub
            from train_idrid import KAGGLE_SLUG
            root = kagglehub.dataset_download(KAGGLE_SLUG)
        ds = IDRiDSegDataset(root, split=split, image_size=image_size, lesions=lesions)
    elif name == "fgadr":
        from fgadr_dataset import FGADRSegDataset
        ds = FGADRSegDataset(root, split=split, image_size=image_size, lesions=lesions, augment=False)
    else:
        raise SystemExit(f"[fatal] unknown dataset {name!r}")
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        img, masks = ds.load_full(i)
        yield os.path.splitext(ds.files[i])[0], img, masks


def _seg_whole_inputs(img: np.ndarray, masks: Optional[np.ndarray], size: int):
    img_r = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    masks_r = None
    if masks is not None:
        masks_r = np.stack([np.asarray(Image.fromarray(m).resize((size, size), Image.NEAREST))
                            for m in masks]).astype(np.uint8)
    x = torch.from_numpy(img_r.astype(np.float32).transpose(2, 0, 1) / 255.0)
    return img_r, masks_r, (x - IMAGENET_MEAN) / IMAGENET_STD


def _cls_inputs(img: np.ndarray, size: int):
    from dataset import build_transforms
    x = build_transforms(split="val", image_size=size)(Image.fromarray(img))
    img_r = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    return img_r, x


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Save the attention maps the GCG gates produce.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="auto", choices=["auto", "seg", "cls"],
                   help="'auto' reads it from the checkpoint (train_idrid vs train_mbrset).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", metavar="DIR", help="Folder of fundus images (no ground truth).")
    src.add_argument("--dataset", choices=["idrid", "fgadr"], help="Lesion dataset with masks.")
    p.add_argument("--root", default=None, help="Dataset root (IDRiD: kagglehub download if omitted; "
                                                "FGADR: auto-detected under data/).")
    p.add_argument("--split", default="test", help="idrid: train|test; fgadr: train|val|test|all.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images.")
    p.add_argument("--image-size", type=int, default=None,
                   help="Whole-image inference size (default: the checkpoint's; 512 for a "
                        "patch-trained seg model).")
    p.add_argument("--tiled", action="store_true", help="Seg only: native resolution via tiles.")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--thresh", type=float, default=0.5, help="Seg prediction threshold for the overlay.")
    p.add_argument("--out-dir", default="out/gcg_maps")
    p.add_argument("--alpha", type=float, default=0.55, help="Heat-map opacity in the overlays.")
    p.add_argument("--png-size", type=int, default=512, help="Longer side of each panel tile.")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-npz", action="store_true")
    p.add_argument("--save-pred", action="store_true", help="Also store the prediction in the .npz.")
    p.add_argument("--separate-pngs", action="store_true",
                   help="Also write one overlay PNG (and one raw map PNG) per gate.")
    p.add_argument("--stretch", action="store_true",
                   help="Min-max stretch each map for the PNGs (files keep raw values). Use it "
                        "when a gate sits near one value everywhere and the overlay is flat.")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = detect_kind(ck) if args.model == "auto" else args.model
    ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}

    class_names = None
    if kind == "seg":
        net, lesions = load_seg(args.checkpoint, device)
        size = args.image_size or (int(ck_args.get("image_size") or 512)
                                   if not ck_args.get("patch_size") else 512)
    else:
        if args.tiled or args.dataset:
            raise SystemExit("[fatal] --tiled / --dataset apply to segmentation checkpoints; "
                             "give the classifier a folder via --images.")
        net, ck_size = load_cls(args.checkpoint, ck, device)
        size = args.image_size or ck_size
        lesions = []
        task = ck_args.get("task")
        if task:
            try:
                from dataset import LABEL_REGISTRY
                class_names = list(getattr(LABEL_REGISTRY[task], "class_names", None) or [])
            except Exception:
                class_names = None

    gate_mods = gcg_gate_modules(net)
    if not gate_mods:
        raise SystemExit("[fatal] this checkpoint has no GCG gates (trained with --no-gcg, or a "
                         "timm backbone) — there are no attention maps to save.")
    print(f"[info] {kind} model on {device}; {len(gate_mods)} GCG gate(s): "
          f"{[n for n, _ in gate_mods]}")
    print(f"[info] inference: {'tiled @ native, tile ' + str(args.tile) if args.tiled else f'whole image @ {size}px'}")

    os.makedirs(args.out_dir, exist_ok=True)
    items = (iter_image_dir(args.images, args.limit) if args.images
             else iter_dataset(args.dataset, args.root, args.split, size, lesions, args.limit))

    rows: List[dict] = []
    gate_meta = None
    t0 = time.time()
    n_done = 0
    for stem, img, masks in items:
        if kind == "cls":
            img_r, x = _cls_inputs(img, size)
            probs, gates = infer_whole(net, x, device, "cls")
            masks_r = None
        elif args.tiled:
            x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            fg = torch.from_numpy(img.max(axis=2) > FOV_TOL).to(device)
            probs, gates = infer_tiled(net, x, args.tile, args.overlap, device, fg, args.tile_batch)
            img_r, masks_r = img, masks
        else:
            img_r, masks_r, x = _seg_whole_inputs(img, masks, size)
            probs, gates = infer_whole(net, x, device, "seg")
        probs_np = probs.numpy()
        save_image_outputs(args.out_dir, stem, kind, img_r, masks_r, lesions, probs_np, gates,
                           args.thresh, args.alpha, args.png_size, not args.no_png, not args.no_npz,
                           args.save_pred, args.separate_pngs, class_names, args.stretch)
        rows.extend(gate_stats_rows(stem, gates, img_r, masks_r, lesions))
        if gate_meta is None:
            gate_meta = [{"name": g["name"], "stride": g["stride"],
                          "spatial_hw": None if g["spatial"] is None else list(g["spatial"].shape),
                          "channels": None if g["channel"] is None else int(g["channel"].size)}
                         for g in gates]
        n_done += 1
        print(f"\r  {n_done} image(s) ({time.time() - t0:.0f}s) — {stem}", end="", flush=True)
    print()

    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, "gate_stats.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(args.out_dir, "run.json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "kind": kind,
                   "inference": "tiled" if args.tiled else f"whole@{size}",
                   "images": n_done, "lesions": lesions, "gates": gate_meta,
                   "args": vars(args)}, f, indent=2)

    # Summary: where does GCG look? Mean attention on vs off lesions, per gate.
    if rows:
        print(f"\n=== GCG gates over {n_done} image(s) ===")
        print(f"{'gate':<22}{'stride':>7}{'sp mean':>9}{'sp>0.5':>8}{'ch mean':>9}"
              + (f"{'on lesion':>11}{'off lesion':>11}{'ratio':>7}" if "on_any_lesion" in rows[0] else ""))
        for k, g in enumerate(gate_meta or []):
            rs = [r for r in rows if r["level"] == k]
            def m(key):
                v = [r[key] for r in rs if key in r and r[key] == r[key]]
                return float(np.mean(v)) if v else float("nan")
            line = (f"{g['name'].replace('.gcg', ''):<22}{str(g['stride'] or '-'):>7}"
                    f"{m('sp_mean'):>9.3f}{m('sp_frac_gt_0p5'):>8.2f}{m('ch_mean'):>9.3f}")
            if "on_any_lesion" in rows[0]:
                line += f"{m('on_any_lesion'):>11.3f}{m('off_any_lesion'):>11.3f}{m('ratio_any_lesion'):>7.2f}"
            print(line)
        if "on_any_lesion" in rows[0]:
            print("ratio > 1 = the gate passes more of the skip feature on lesion pixels than off them.")
    print(f"[ok] {n_done} image(s) -> {args.out_dir}/  ({'' if args.no_npz else '<stem>.npz, '}"
          f"{'' if args.no_png else '<stem>.png, '}gate_stats.csv, run.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
