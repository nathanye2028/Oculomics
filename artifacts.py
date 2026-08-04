"""
artifacts.py
============
Phase 1 — lightweight, **GPU-free** artifact-reduction pipeline for mobile
fundus capture (ISEF RQ1: "actively detect and suppress mobile-specific imaging
artifacts — illumination glare, poor focus — in real-time").

Two jobs:

1. **Quality assessment** (`assess_quality`) — per-image, fast, interpretable:
     * focus / sharpness   — variance of the Laplacian within the field of view
     * glare               — fraction of specular (bright + desaturated) pixels
     * illumination uniformity — spread of the low-frequency luminance
   -> a 0-1 quality score + a `gradable` flag with reasons, so low-quality
   smartphone captures can be triaged before they ever reach the model.

2. **Localized normalization** (`preprocess`) — standardize the visual input:
     * FOV crop (drop the black border)
     * glare suppression (smooth-fill specular highlights)
     * illumination normalization (Ben-Graham: subtract local average colour)

Dependency-light on purpose (numpy + Pillow + a tiny torch conv) so it stays
mobile-portable. Thresholds default to sensible values but SHOULD be calibrated
against mBRSET's ``final_quality`` / ``final_artifacts`` labels once available.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from fundus_utils import circular_fov_mask, crop_to_fov

# 3x3 Laplacian for the focus measure.
_LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def focus_score(rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Sharpness = variance of the Laplacian (higher = sharper). Computed on the
    grayscale image within ``mask`` (the field of view), so the black border
    doesn't dominate. Scaled to 0-255 for interpretable magnitudes."""
    gray = (rgb.astype(np.float32).mean(axis=2))[None, None]
    lap = F.conv2d(torch.from_numpy(gray), _LAP, padding=1)[0, 0].numpy()
    if mask is not None:
        lap = lap[mask]
    return float(lap.var())


def detect_glare(rgb: np.ndarray, mask: Optional[np.ndarray] = None,
                 v_thresh: float = 0.85, s_thresh: float = 0.25
                 ) -> Tuple[np.ndarray, float]:
    """Specular glare = bright (high value) AND desaturated (low saturation).
    Returns (glare_mask[H,W] bool, glare_fraction within the FOV)."""
    arr = rgb.astype(np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.clip(mx, 1e-6, None), 0.0)
    glare = (mx > v_thresh) & (sat < s_thresh)
    if mask is not None:
        glare = glare & mask
        denom = int(mask.sum())
    else:
        denom = glare.size
    return glare, float(glare.sum() / max(denom, 1))


def illumination_uniformity(rgb: np.ndarray, mask: Optional[np.ndarray] = None,
                            sigma: float = 25.0) -> float:
    """Spread (std/mean) of the low-frequency luminance within the FOV.
    Lower = more uniform illumination; higher = vignetting / uneven flash."""
    img = Image.fromarray(rgb).convert("L").filter(ImageFilter.GaussianBlur(radius=sigma))
    lum = np.asarray(img, dtype=np.float32)
    vals = lum[mask] if mask is not None else lum.ravel()
    m = vals.mean()
    return float(vals.std() / m) if m > 1e-6 else 0.0


def _fov_mask_for(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FOV-crop the image and return (cropped_rgb, circular_fov_bool_mask)."""
    cropped, _ = crop_to_fov(rgb)
    h, w = cropped.shape[:2]
    return cropped, circular_fov_mask(h, w)


def assess_quality(rgb: np.ndarray, min_focus: float = 13.16,
                   max_glare: float = 0.0263) -> Dict[str, object]:
    """Per-image quality report + gradable decision.

    Defaults are **calibrated** (Youden's J) against expert ``final_quality``
    labels on 5,164 mBRSET images via :mod:`validate_artifacts`, replacing the
    original hand-picked guesses.

    Measured discrimination on that set is modest (AUROC 0.62 focus, 0.60
    composite, 0.49 glare), i.e. these heuristics only weakly predict expert
    gradability -- see the learned quality gate (``train_mbrset.py --task
    quality``) for the stronger alternative.
    """
    cropped, fov = _fov_mask_for(rgb)
    focus = focus_score(cropped, fov)
    _, glare_frac = detect_glare(cropped, fov)
    illum = illumination_uniformity(cropped, fov)

    reasons = []
    if focus < min_focus:
        reasons.append("blurry")
    if glare_frac > max_glare:
        reasons.append("glare")
    if illum > 0.6:
        reasons.append("uneven_illumination")
    gradable = len(reasons) == 0

    # Interpretable 0-1 score (heuristic; calibrate later).
    f_term = min(focus / (2 * min_focus), 1.0)
    g_term = max(0.0, 1.0 - glare_frac / max(max_glare, 1e-6))
    u_term = max(0.0, 1.0 - illum)
    score = float(0.5 * f_term + 0.3 * g_term + 0.2 * u_term)
    return {"focus": focus, "glare_frac": glare_frac, "illumination_spread": illum,
            "quality_score": score, "gradable": gradable, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Normalization / suppression
# --------------------------------------------------------------------------- #
def suppress_glare(rgb: np.ndarray, glare_mask: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Smooth-fill specular highlights with a heavily blurred copy (light inpaint)."""
    blur = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=sigma)),
                      dtype=np.float32)
    out = np.where(glare_mask[..., None], blur, rgb.astype(np.float32))
    return out.astype(np.uint8)


def normalize_illumination(rgb: np.ndarray, sigma: float = 10.0,
                           alpha: float = 4.0, beta: float = 128.0) -> np.ndarray:
    """Ben-Graham normalization: subtract the local average colour to flatten
    uneven illumination and boost lesion contrast. Standard fundus preprocessing."""
    blur = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=sigma)),
                      dtype=np.float32)
    out = alpha * (rgb.astype(np.float32) - blur) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(rgb: np.ndarray, do_fov: bool = True, do_glare: bool = True,
               do_illum: bool = True) -> np.ndarray:
    """Full artifact-reduction chain: FOV crop -> glare suppression -> illumination
    normalization. Returns a uint8 RGB image ready for the model."""
    if do_fov:
        rgb, _ = crop_to_fov(rgb)
    if do_glare:
        h, w = rgb.shape[:2]
        gmask, _ = detect_glare(rgb, circular_fov_mask(h, w))
        rgb = suppress_glare(rgb, gmask)
    if do_illum:
        rgb = normalize_illumination(rgb)
    return rgb


# --------------------------------------------------------------------------- #
# CLI demo: score a folder of fundus images + save before/after samples
# --------------------------------------------------------------------------- #
def _default_idrid_dir() -> Optional[str]:
    try:
        import kagglehub
        from idrid_dataset import resolve_seg_base
        base = resolve_seg_base(kagglehub.dataset_download(
            "aaryapatel98/indian-diabetic-retinopathy-image-dataset"))
        return os.path.join(base, "1. Original Images", "a. Training Set")
    except Exception:
        return None


def main() -> int:
    import argparse
    import csv

    p = argparse.ArgumentParser(description="Artifact-reduction demo: quality scores + before/after.")
    p.add_argument("--images-dir", default=None, help="Folder of fundus images (default: IDRiD train).")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out-dir", default="artifact_samples")
    args = p.parse_args()

    images_dir = args.images_dir or _default_idrid_dir()
    if not images_dir or not os.path.isdir(images_dir):
        print("No images dir (pass --images-dir).")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")))[:args.limit]

    rows = []
    print(f"{'file':<22}{'focus':>9}{'glare':>9}{'illum':>9}{'score':>8}  gradable")
    for f in files:
        rgb = np.asarray(Image.open(os.path.join(images_dir, f)).convert("RGB"))
        q = assess_quality(rgb)
        print(f"{f:<22}{q['focus']:>9.2f}{q['glare_frac']:>9.4f}"
              f"{q['illumination_spread']:>9.3f}{q['quality_score']:>8.2f}  "
              f"{q['gradable']} {q['reasons']}")
        rows.append({"file": f, **{k: q[k] for k in
                     ("focus", "glare_frac", "illumination_spread", "quality_score", "gradable")},
                     "reasons": "|".join(q["reasons"])})
        # before/after sample (downscaled side-by-side)
        before = Image.fromarray(rgb)
        after = Image.fromarray(preprocess(rgb))
        bw = before.resize((256, 256)); aw = after.resize((256, 256))
        canvas = Image.new("RGB", (512, 256))
        canvas.paste(bw, (0, 0)); canvas.paste(aw, (256, 0))
        canvas.save(os.path.join(args.out_dir, f"{os.path.splitext(f)[0]}_before_after.png"))

    with open(os.path.join(args.out_dir, "quality.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} before/after samples + quality.csv to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
