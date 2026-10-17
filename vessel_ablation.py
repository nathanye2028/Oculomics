"""
vessel_ablation.py
==================
Anatomy ablation for RetinaReach (plan 5.7): remove the retinal vasculature from
the image and score again. The oculomics literature says systemic targets
(hypertension above all) are read from the microvasculature; if a target's
AUROC survives vessel removal, the model is not reading the signal the
literature describes.

Vessels are found without training data by the multi-scale line detector
(Nguyen, Bhuiyan, Park & Ramamohanarao, "An effective retinal blood vessel
segmentation method using multi-scale line detection", Pattern Recognition 2013):
on the inverted green channel, the response at line length L is the largest mean
intensity along a line of length L through the pixel (12 orientations) minus the
mean over a W x W window; the standardised responses for L = 1, 3, ..., W and the
standardised intensity are averaged. W = 15 on DRIVE (565 px wide) and is
scaled with the image size here. A fixed fraction of the field of view is kept
as vessel (``frac``), so every image loses the same area.

Removal = inpainting from the surrounding retina: a normalised Gaussian
convolution over the non-vessel pixels fills each vessel pixel with its local
background colour.

The fixed area is deliberate (every image loses the same amount of pixels), so
on a hazy image with few visible vessels the quota is filled with texture: the
paired vessels - control contrast, not the vessel mask alone, carries the result.

The control removes the SAME mask rotated by 90 degrees (square model inputs),
kept inside the field of view: the same amount of thin, branching structure, but
mostly on background retina. The contrast that matters is vessels - control on
the same images (paired), not vessels - intact: inpainting anything costs a
little.

Caveat: dark, elongated lesions (flame haemorrhages) respond to a line detector
too, so a DR drop under vessel removal is partly lesion removal. The ablation is
read mainly for the systemic targets.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
FOV_TOL = 12.0 / 255.0


def window_for(size: int) -> int:
    """W = 15 at DRIVE's 565 px, scaled to the model input, kept odd and >= 5."""
    w = int(round(15 * size / 565))
    w = max(5, w)
    return w if w % 2 else w + 1


@lru_cache(maxsize=8)
def line_kernels(W: int, n_angles: int = 12) -> Dict[int, torch.Tensor]:
    """{L: [n_angles, 1, W, W]} kernels averaging along a centred line of length L."""
    c = W // 2
    out = {}
    for L in range(1, W + 1, 2):
        ks = torch.zeros(n_angles, 1, W, W)
        for a in range(n_angles):
            th = math.pi * a / n_angles
            for t in np.linspace(-(L - 1) / 2, (L - 1) / 2, L):
                x = int(round(c + t * math.cos(th)))
                y = int(round(c - t * math.sin(th)))
                ks[a, 0, y, x] += 1.0
            ks[a, 0] /= ks[a, 0].sum()
        out[L] = ks
    return out


def fov_mask(x01: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] bool: max(R,G,B) > 12/255 (fundus_utils.fov_bbox's rule, per pixel)."""
    return x01.max(dim=1, keepdim=True).values > FOV_TOL


def _standardise(r: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    mf = m.float()
    n = mf.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mu = (r * mf).sum(dim=(2, 3), keepdim=True) / n
    sd = (((r - mu) ** 2) * mf).sum(dim=(2, 3), keepdim=True).div(n).sqrt().clamp(min=1e-6)
    return (r - mu) / sd


@torch.no_grad()
def vessel_response(x01: torch.Tensor, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combined multi-scale line response [B,1,H,W] and the FOV mask."""
    fov = fov_mask(x01)
    inv = 1.0 - x01[:, 1:2]                                   # inverted green: vessels bright
    mf = fov.float()
    mean_in = (inv * mf).sum(dim=(2, 3), keepdim=True) / mf.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    inv = torch.where(fov, inv, mean_in)                      # no bright ring outside the FOV
    avg = F.avg_pool2d(inv, W, stride=1, padding=W // 2, count_include_pad=False)
    acc = _standardise(inv, fov)
    ks = line_kernels(W)
    for L, k in ks.items():
        line = F.conv2d(F.pad(inv, [W // 2] * 4, mode="replicate"), k.to(inv))
        acc = acc + _standardise(line.max(dim=1, keepdim=True).values - avg, fov)
    return acc / (len(ks) + 1), fov


@torch.no_grad()
def vessel_mask(x01: torch.Tensor, frac: float = 0.12, W: int = 0) -> torch.Tensor:
    """[B,1,H,W] bool: the top ``frac`` of the line response inside the FOV
    (eroded by W/2 so the FOV edge is never called a vessel)."""
    W = W or window_for(x01.shape[-1])
    r, fov = vessel_response(x01, W)
    # erode with the outside of the image counted as outside the FOV (max_pool's
    # own padding is -inf, which would leave the frame edge uneroded)
    inner = -F.max_pool2d(-F.pad(fov.float(), [W // 2] * 4, value=0.0), W, stride=1) > 0.5
    out = torch.zeros_like(fov)
    for i in range(r.shape[0]):
        v = r[i][inner[i]]
        if v.numel() == 0:
            continue
        thr = torch.quantile(v.float(), 1.0 - frac) if v.numel() < 16_000_000 else v.kthvalue(
            int((1.0 - frac) * v.numel())).values
        out[i] = (r[i] >= thr) & inner[i]
    return out


@torch.no_grad()
def inpaint(x01: torch.Tensor, mask: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """Fill ``mask`` pixels with the Gaussian-weighted mean of nearby unmasked FOV
    pixels (normalised convolution); everything else is unchanged."""
    sigma = sigma or max(2.0, window_for(x01.shape[-1]) / 2.0)
    r = int(math.ceil(3 * sigma))
    g = torch.exp(-torch.arange(-r, r + 1, dtype=x01.dtype, device=x01.device) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    keep = (fov_mask(x01) & ~mask).to(x01.dtype)

    def blur(t):
        c = t.shape[1]
        t = F.conv2d(F.pad(t, [r, r, 0, 0], mode="replicate"), g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        return F.conv2d(F.pad(t, [0, 0, r, r], mode="replicate"), g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)

    bg = blur(x01 * keep) / blur(keep).clamp(min=1e-6)
    return torch.where(mask.expand_as(x01), bg, x01)


def control_mask(mask: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """The vessel mask rotated by 90 degrees, kept inside the FOV."""
    return torch.rot90(mask, 1, dims=(2, 3)) & fov


@torch.no_grad()
def ablate_normalised(x: torch.Tensor, kind: str, frac: float = 0.12) -> torch.Tensor:
    """ImageNet-normalised batch in -> the same batch with vessels (``kind="vessels"``)
    or the rotated control mask (``kind="control"``) inpainted, normalised again."""
    mean, std = _MEAN.to(x), _STD.to(x)
    x01 = (x * std + mean).clamp(0, 1)
    m = vessel_mask(x01, frac)
    if kind == "control":
        m = control_mask(m, fov_mask(x01))
    elif kind != "vessels":
        raise ValueError(f"unknown ablation {kind!r}")
    return (inpaint(x01, m) - mean) / std


def mask_stats(x01: torch.Tensor, frac: float = 0.12) -> Dict[str, float]:
    """Area of the vessel and control masks as fractions of the FOV, and their overlap."""
    m = vessel_mask(x01, frac)
    fov = fov_mask(x01)
    c = control_mask(m, fov)
    f = fov.float().sum()
    return {"vessel_frac": float(m.float().sum() / f), "control_frac": float(c.float().sum() / f),
            "overlap_of_control": float((m & c).float().sum() / c.float().sum().clamp(min=1))}
