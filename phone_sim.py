#!/usr/bin/env python3
"""
phone_sim.py
============
Re-render a tabletop fundus photograph (BRSET: Canon CR-2, Nikon NF5050) as if the
handheld phone-based camera of mBRSET had taken it, so the tabletop model can be
trained on the kind of photo it will be given. A stochastic imaging chain in the
model's own view (after the loader's FOV crop and resize), every parameter drawn
from measurements of real, unlabelled handheld photos.

    # 1. fit: measure the handheld camera (and the tabletop one, for the report)
    python phone_sim.py fit --target-root $M --source-root $B --source-camera 'Canon CR' \\
        --out ck_phone_sim/profile.json
    # (or from two folders of photos: --target-dir DIR --source-dir DIR)
    # 2. look at it: tabletop -> simulated -> real handheld, plus a per-stage strip
    python phone_sim.py render --profile ck_phone_sim/profile.json --source-dir fundus_samples/brset \\
        --target-dir fundus_samples/mbrset --out exp_phone_sim/render
    # 3. train with it (each tabletop training photo re-rendered with probability p)
    python train_retinareach.py ... --phone-sim ck_phone_sim/profile.json --phone-sim-p 0.5

What the two cameras' photos differ in, in the model's view, and the stage that
carries each difference (measured on the 16 + 16 photos in ``fundus_samples/``;
``fit`` re-measures it on the full train split):

``field``   The loader crops each photo to its FOV bounding box and squashes it to
            a square. Canon frames truncate the FOV circle, so a Canon retina
            arrives squashed ~20 % horizontally (aspect 0.80, the crop's own
            height/width); the phone's FOV is an exact digital circle (aspect
            1.00, hard edge). The phone's field is also narrower: its disc-fovea
            distance is ~1.15x the tabletop's in FOV radii (``fit`` measures it;
            disc size reads higher because phone discs bloom), and its disc sits
            nearer the centre (0.45 vs 0.58 of the radius). So: un-squash (the
            FOV ellipse fitted per image from its side boundaries), zoom about the
            FOV centre -- at least enough for the aperture to clear a truncated
            FOV's flat top and bottom (~1.25 for Canon frames, found by checking
            coverage) -- shift toward the detected disc until it lands where the
            drawn handheld photo has its disc (capped so the aperture never
            leaves the tabletop FOV), a small rotation, and the phone's aperture.
            Training passes a supersampled view (``supersample`` x the model
            size) so the zoom does not cost detail the phone's own photo has.
``light``   Low-order illumination shape (log intensity on 1, r^2, r^4, x, y,
            fitted inside 0.85 of the radius with the disc left out) transferred
            from the drawn handheld photo -- the phone's light is tilted (brighter
            toward the bottom) and its rim reddens: one luminance gain plus a
            per-channel radial term that may only darken; the photo's dark rim
            bands; and smooth random blotches at the measured scale (1/e length
            0.17 FOV diameters, about 1.7 disc diameters -- finer ones read as
            haemorrhages) up to the photo's own blotchiness. The change is scaled
            back until no more retina sits in shadow than in the drawn photo.
``colour``  Green left alone; red and blue take on green's structure, the
            strongest camera fingerprint (the phone's B and R carry G's vessel
            pattern: band correlation 0.94 / 0.85 vs 0.54 / 0.61): each becomes
            (1-w) C + w mu_C (G / mu_G)^kappa, w the smallest weight that reaches the
            drawn photo's correlation; then ratio-preserving gains to the drawn
            photo's centre-zone means. All on the image divided by its own
            low-order illumination field, multiplied back after, so the light
            pattern survives. Measured on planted lesions in the samples: lesion
            contrast x0.96-0.97 (real phones show ~x1.1). Rejected, each on the
            samples: whole-image moment matching -- a 3x3 optimal-transport map
            (``colour_mode: mk``) closed more of the statistical gap but made
            lesions 1.4-1.7x more conspicuous than on the tabletop photo, i.e. the
            simulated phone would show DR more clearly than a real one; a
            per-channel map (``affine``) 2x; per-channel histogram matching
            painted haemorrhage-like red blotches and purple foveal rings (the
            tabletop blue channel is nearly flat and stretching it amplifies
            texture).
``bloom``   The phone's auto-exposure blows the optic disc out to near-white;
            the detected disc region (and only it) is pushed toward the drawn
            photo's disc colour, by local brightness so vessels on the disc stay
            darker than its rim. A luminance-wide highlight curve, tried first,
            whitened exudates and rim reflections into lesion-like specks.
``detail``  The phone is NOT blurrier in the model's view (vessel edges 2.34 vs
            2.29 px at 384); its extra high-frequency power is grain. So blur
            only, at most 0.8 px, and only when a tabletop photo's contrast-
            normalised mid-band detail (sigma 1 - 2.5 px: vessel edges, lesions)
            exceeds the drawn photo's. No unsharp mask: it rings microaneurysms
            and turns grain into microaneurysm-like specks.
``noise``   Grain to the drawn photo's flat-region level (0.86 / 255 at 384 on the
            samples), nearly white at the model's scale and mostly luma; its
            spatial and cross-channel correlation are calibrated by simulation in
            ``fit``. Tapered off on clipped and black pixels, where the phone has
            none (grain on a blown-out disc would read as fine new vessels). Then,
            as the phone's demosaic and colour matrix act after its sensor, R and B
            fine structure (grain included) is pulled toward G's up to the drawn
            photo's correlation; G and the low frequencies are not touched.
``rim``     The drawn photo's bluish-white rim haze, per sector and channel,
            relative to its own illumination model (stray light, added). A drawn
            photo's horizontal light and rim are mirrored when its disc sits on
            the other side from the tabletop photo's.

Every stage takes a whole row of the **bank**: one row of measurements per real
handheld photo, so a simulated photo gets the colour, illumination, sharpness,
noise and rim of ONE real photo together (darker phone photos are also noisier
and softer). The bank holds statistics only, never pixels. ``fit`` uses the
handheld photos' pixels only, no label, unless ``--select icdr0``.

What it does not do: invent or remove lesions on purpose. Every stage is either
geometric, a smooth low-order or large-scale field, a pixel-wise colour map that
leaves the green channel's ratios alone, the disc region, a capped blur or grain. The zoom can still crop a peripheral
lesion out of the narrower field (the phone's own field has the same limit), and
blur can soften microaneurysms the phone also softens; ``tests/test_phone_sim.py``
checks that planted lesions keep their sign and contrast, and that a lesion-free
fundus gains no lesion-like spots. The realism and label-preservation checks on
real data are ``style_transfer_xfer.py check --methods phone_sim``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_transfer import Stylizer, denorm, renorm  # noqa: E402
from vessel_ablation import FOV_TOL, fov_mask  # noqa: E402

FORMAT = "phone_sim/1"
STAGES = ("field", "light", "colour", "bloom", "detail", "noise", "rim")
N_SECTORS = 16
# The detail ladder: (a, r) with a > 0 = Gaussian blur of sigma a px, a = 0 = unchanged.
# Blur only, capped at 0.8 px at 384: in the model's view the handheld is not blurrier
# (vessel edges 2.34 vs 2.29 px) and its extra high-frequency power is grain, so no
# unsharp mask (it rings microaneurysms and turns grain into microaneurysm-like specks).
DETAIL_OPS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0), (0.3, 0.0), (0.45, 0.0), (0.6, 0.0), (0.8, 0.0))
DEFAULT_PARAMS: Dict[str, object] = {
    "zoom": 1.15, "zoom_sd": 0.04, "rot_sd_deg": 4.0, "jitter": 0.03, "rim_width": 0.0023,
    "source_aspect": 0.80, "src_illum": None, "shrink": 0.5, "disc_dist": 0.45,
    "noise_corr": 0.55, "noise_rho": 0.94, "light_lim": [0.25, 2.5], "colour_common": [0.6, 1.6],
    "colour_spread": 1.15, "colour_mode": "coupled", "kappa": [0.4, 1.4], "fine_chroma": False, "supersample": 1.5, "disc_prom": 6.0, "blotch_len": 0.7, "blotch_max": 0.3, "blotch_gain": 1.0,
}


# --------------------------------------------------------------------------- #
# Batched image operations (per-image parameters, any device)
# --------------------------------------------------------------------------- #
def frame_coords(B: int, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates in [-1, 1] (``grid_sample``'s align_corners=False)."""
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2 - 1
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.expand(B, H, W), yy.expand(B, H, W)


def gaussian_kernels(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """[B, 2r+1] normalised 1-D Gaussians; sigma <= 1e-3 gives a delta."""
    t = torch.arange(-radius, radius + 1, device=sigma.device, dtype=torch.float32)
    s = sigma.float().clamp_min(1e-3).view(-1, 1)
    k = torch.exp(-0.5 * (t.view(1, -1) / s) ** 2)
    k = k / k.sum(dim=1, keepdim=True)
    delta = torch.zeros_like(k)
    delta[:, radius] = 1.0
    return torch.where(sigma.view(-1, 1) <= 1e-3, delta, k)


def blur(x: torch.Tensor, sigma) -> torch.Tensor:
    """Separable Gaussian blur, one sigma (px) per image (or one for all), reflect padding."""
    B, C, H, W = x.shape
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).expand(B).contiguous()
    smax = float(sigma.max()) if B else 0.0
    if smax <= 1e-3:
        return x
    r = min(max(1, int(math.ceil(3.0 * smax))), (min(H, W) - 1) // 2)
    k = gaussian_kernels(sigma, r).to(x.dtype).repeat_interleave(C, dim=0)        # [B*C, K]
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(B * C, 1, 1, -1), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(B * C, 1, -1, 1), groups=B * C)
    return y.view(B, C, H, W)


def erode(m: torch.Tensor, k: int) -> torch.Tensor:
    """Square erosion of a [B,1,H,W] mask (separable: a k x 1 then a 1 x k pass); the
    frame edge counts as outside (a truncated FOV's chord is a real boundary)."""
    k = max(1, int(k)) | 1
    p = k // 2
    e = -F.max_pool2d(-F.pad(m.float(), (0, 0, p, p), value=0.0), (k, 1), stride=1)
    return -F.max_pool2d(-F.pad(e, (p, p, 0, 0), value=0.0), (1, k), stride=1) > 0.5


def _masked_quantiles(v: torch.Tensor, keep: torch.Tensor, q: Sequence[float]) -> torch.Tensor:
    """Quantiles over the last dim of ``v`` where ``keep``; [len(q), *v.shape[:-1]]
    (computed on the CPU for backends without nanquantile)."""
    dev = v.device
    work = "cpu" if dev.type == "mps" else dev
    vv = v.to(work).float().masked_fill(~keep.to(work), float("nan"))
    return torch.nanquantile(vv, torch.tensor(list(q), device=work), dim=-1).to(dev, v.dtype)


def _strided(x: torch.Tensor, m: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """A deterministic pixel subsample: every ``step``-th row and column -> ([B,C,N], [B,1,N])."""
    xs, ms = x[..., ::step, ::step], m[..., ::step, ::step]
    return xs.reshape(x.shape[0], x.shape[1], -1), ms.reshape(x.shape[0], 1, -1)


def _to_source(px: torch.Tensor, py: torch.Tensor, zoom: torch.Tensor, a_f: torch.Tensor, b_f: torch.Tensor,
               offset: torch.Tensor, rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handheld-frame points (the aperture is the unit circle; [B, ...] tensors) ->
    source frame points. The source FOV is an axis-aligned ellipse with semi-axes
    ``a_f``, ``b_f`` (frame half-widths; the loader's square resize of a truncated
    circle makes b_f > a_f): rotate by ``rot`` in FOV units, shrink by the
    FOV-relative ``zoom``, scale onto the ellipse, move to ``offset`` (the FOV
    centre plus the framing shift)."""
    v = lambda t: t.view(-1, *([1] * (px.dim() - 1)))
    c, s = torch.cos(rot), torch.sin(rot)
    xr, yr = v(c) * px - v(s) * py, v(s) * px + v(c) * py
    return (v(a_f) * xr / v(zoom) + v(offset[:, 0]), v(b_f) * yr / v(zoom) + v(offset[:, 1]))


def warp(x: torch.Tensor, zoom: torch.Tensor, a_f: torch.Tensor, b_f: torch.Tensor, offset: torch.Tensor,
         rot: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """The handheld frame sampled bilinearly from the source (see :func:`_to_source`);
    outside the source is black."""
    B = x.shape[0]
    xx, yy = frame_coords(B, out_hw[0], out_hw[1], x.device)
    gx, gy = _to_source(xx, yy, zoom, a_f, b_f, offset, rot)
    return F.grid_sample(x, torch.stack([gx, gy], dim=-1).to(x.dtype), mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def aperture(radius: torch.Tensor, width: float, H: int, W: int) -> torch.Tensor:
    """[B,1,H,W] soft circular aperture centred in the frame; ``radius`` in frame
    half-widths (1 = touches the edges), logistic roll-off ``width`` likewise
    (10-90 % width = 4.4 x ``width``)."""
    B = radius.shape[0]
    xx, yy = frame_coords(B, H, W, radius.device)
    r = torch.sqrt(xx ** 2 + yy ** 2)
    return torch.sigmoid((radius.view(B, 1, 1) - r) / max(width, 1e-4)).unsqueeze(1)


def _rim_inside(fov: torch.Tensor, zoom: torch.Tensor, a_f: torch.Tensor, b_f: torch.Tensor,
                offset: torch.Tensor, rot: torch.Tensor, radius: float = 0.97, n_pts: int = 48) -> torch.Tensor:
    """Fraction of the handheld aperture's rim that lands inside the source FOV mask.
    ``zoom``/``offset`` may carry a candidate axis: [B,K] and [B,K,2] -> [B,K]."""
    B, dev = fov.shape[0], fov.device
    K = zoom.shape[1]
    ang = torch.linspace(0, 2 * math.pi, n_pts + 1, device=dev)[:-1]
    px = (radius * torch.cos(ang)).view(1, 1, -1).expand(B, K, -1)
    py = (radius * torch.sin(ang)).view(1, 1, -1).expand(B, K, -1)
    c, s = torch.cos(rot).view(B, 1, 1), torch.sin(rot).view(B, 1, 1)
    xr, yr = c * px - s * py, s * px + c * py
    gx = a_f.view(B, 1, 1) * xr / zoom.unsqueeze(-1) + offset[..., 0].unsqueeze(-1)
    gy = b_f.view(B, 1, 1) * yr / zoom.unsqueeze(-1) + offset[..., 1].unsqueeze(-1)
    inside = F.grid_sample(fov.float(), torch.stack([gx, gy], dim=-1), mode="nearest", padding_mode="zeros",
                           align_corners=False)
    return inside[:, 0].mean(dim=-1)


def min_zoom(fov: torch.Tensor, want: torch.Tensor, a_f: torch.Tensor, b_f: torch.Tensor, centre: torch.Tensor,
             rot: torch.Tensor, need: float = 0.98, top: float = 1.8, n: int = 16) -> torch.Tensor:
    """The smallest zoom >= ``want`` (up to ``top``) whose aperture stays inside the
    source FOV: a truncated tabletop FOV needs ~1.25 before the circle clears its
    flat top and bottom."""
    B, dev = fov.shape[0], fov.device
    f = torch.linspace(0, 1, n, device=dev).view(1, -1)
    cand = want.view(B, 1) + f * (torch.maximum(want, torch.full_like(want, top)) - want).view(B, 1)
    ok = _rim_inside(fov, cand, a_f, b_f, centre.view(B, 1, 2).expand(B, n, 2), rot) >= need
    first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((B,), n - 1, device=dev))
    return cand.gather(1, first.view(B, 1))[:, 0]


def safe_shift(fov: torch.Tensor, want: torch.Tensor, zoom: torch.Tensor, a_f: torch.Tensor, b_f: torch.Tensor,
               centre: torch.Tensor, rot: torch.Tensor, need: float = 0.98,
               steps: Sequence[float] = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0)) -> torch.Tensor:
    """The largest fraction of each wanted shift (frame units) for which the
    aperture's rim still lands inside the source FOV; no shift when none does."""
    B, dev = fov.shape[0], fov.device
    ts = torch.tensor(list(steps), device=dev)
    off = centre.view(B, 1, 2) + want.view(B, 1, 2) * ts.view(1, -1, 1)
    ok = _rim_inside(fov, zoom.view(B, 1).expand(B, len(ts)), a_f, b_f, off, rot) >= need
    first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((B,), len(ts) - 1, device=dev))
    return want * ts[first].view(B, 1)


def correlated_noise(shape, sigma_c: float, rho: float, gen: Optional[torch.Generator],
                     device) -> torch.Tensor:
    """Unit-variance Gaussian noise, spatially correlated by a Gaussian of ``sigma_c``
    px and correlated across channels with coefficient ``rho`` (luma-like)."""
    B, C, H, W = shape
    w = torch.randn((B, C + 1, H, W), generator=gen, device=device)
    eps = math.sqrt(rho) * w[:, :1] + math.sqrt(max(0.0, 1.0 - rho)) * w[:, 1:]
    if sigma_c > 1e-3:
        r = min(max(1, int(math.ceil(3 * sigma_c))), (min(H, W) - 1) // 2)
        k = gaussian_kernels(torch.tensor([sigma_c], device=device), r)
        eps = blur(eps, sigma_c) / float(k.pow(2).sum())          # 2-D variance factor (sum k^2)^2
    return eps


def smooth_field(B: int, H: int, W: int, length: float, gen: Optional[torch.Generator], device,
                 grid: int = 48) -> torch.Tensor:
    """[B,1,H,W] Gaussian random field whose autocorrelation falls to 1/e at
    ``length`` frame half-widths (blurred white noise on a coarse grid, bilinear
    up), with the low-order illumination basis [1, r^2, r^4, x, y] projected out
    inside the unit circle and unit RMS there: the blotchiness a low-order
    illumination model leaves unexplained, which is how the phone photos' own
    blotchiness is measured. The projection shortens the field's correlation:
    ``length`` 0.7 leaves ~0.31 half-widths, close to the handheld's measured 0.17
    FOV diameters = 0.34 half-widths (1.7 disc diameters); finer fields read as
    banding and haemorrhage-sized dark patches."""
    # white noise over twice the frame (so the blur kernel is never truncated), blurred,
    # the centre kept
    w = torch.randn((B, 1, 2 * grid, 2 * grid), generator=gen, device=device)
    f = blur(w, max(0.5, length / 2 * grid / 2))[..., grid // 2:grid // 2 + grid, grid // 2:grid // 2 + grid]
    f = f.reshape(B, -1)
    xx, yy = frame_coords(B, grid, grid, device)
    inside = ((xx ** 2 + yy ** 2) <= 1.0).view(B, -1).float()
    Phi = _illum_basis(xx, yy).view(B, 5, -1) * inside.unsqueeze(1)
    A = torch.einsum("bkn,bjn->bkj", Phi, Phi) + 1e-6 * torch.eye(5, device=device)
    coef = torch.linalg.solve(A.cpu().double(), torch.einsum("bkn,bn->bk", Phi, f * inside).cpu().double()
                              .unsqueeze(-1)).squeeze(-1).float().to(device)
    f = (f - torch.einsum("bk,bkn->bn", coef, _illum_basis(xx, yy).view(B, 5, -1))) * inside
    f = f / ((f ** 2).sum(dim=1, keepdim=True) / inside.sum(dim=1, keepdim=True).clamp_min(1)).sqrt().clamp_min(1e-6)
    return F.interpolate(f.view(B, 1, grid, grid), size=(H, W), mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# Measurements (the same functions build the bank, drive the render and score it)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fov_shape(x01: torch.Tensor, min_rows: float = 0.5) -> Dict[str, torch.Tensor]:
    """The FOV as an axis-aligned ellipse fitted to its side boundaries
    (half-width^2 = c0 + c1 y + c2 y^2 over rows whose FOV run touches neither
    frame side). ``aspect`` = a/b = sqrt(-c2) (1 = circle; a Canon frame after the
    loader's square resize reads 0.80), ``a_f`` the horizontal semi-axis and
    ``yc`` the centre row (frame half-widths), ``reliable`` when at least
    ``min_rows`` of the rows were usable; ``area`` = FOV fraction of the frame,
    ``radius`` = the equal-area circle in frame half-widths."""
    B, C, H, W = x01.shape
    m = fov_mask(x01)[:, 0].float()
    cols = torch.arange(W, device=x01.device, dtype=torch.float32)
    has = m.sum(dim=2) > 0
    left = torch.where(m > 0, cols, torch.full_like(m, float(W))).amin(dim=2)
    right = torch.where(m > 0, cols, torch.full_like(m, -1.0)).amax(dim=2)
    hw = (right - left + 1) / 2
    inner = has & (left > 0) & (right < W - 1) & (hw > 2)
    y = (torch.arange(H, device=x01.device, dtype=torch.float32) - (H - 1) / 2).expand(B, H)
    A = torch.stack([torch.ones_like(y), y, y ** 2], dim=-1) * inner.unsqueeze(-1)
    sol = torch.linalg.lstsq(A.cpu().double(), ((hw ** 2) * inner).unsqueeze(-1).cpu().double()
                             ).solution.squeeze(-1).float().to(x01.device)
    c0, c1, c2 = sol[:, 0], sol[:, 1], sol[:, 2]
    rows = inner.float().mean(dim=1)
    neg = c2 < 0
    aspect = torch.where(neg, torch.sqrt((-c2).clamp_min(1e-6)), torch.ones_like(c2))
    c2s = torch.where(neg, c2, -torch.ones_like(c2))
    yv = -c1 / (2 * c2s)
    a_px = torch.sqrt((c0 - c1 ** 2 / (4 * c2s)).clamp_min(1.0))
    a_f, yc = a_px / (W / 2), yv / (H / 2)
    reliable = (rows >= min_rows) & neg & (aspect > 0.6) & (aspect < 1.05) & (a_f > 0.8) & (a_f < 1.25) \
        & (yc.abs() < 0.2)
    area = m.mean(dim=(1, 2))
    return {"aspect": aspect, "a_f": a_f, "yc": yc, "reliable": reliable,
            "area": area, "radius": torch.sqrt(area * 4 / math.pi)}


def _pooled(c: torch.Tensor, m: torch.Tensor, grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ms = F.adaptive_avg_pool2d(m, grid)
    inm = ms > 0.5
    v = F.adaptive_avg_pool2d(c * m, grid) / ms.clamp_min(1e-3)
    fill = (v * inm).sum(dim=(2, 3), keepdim=True) / inm.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    return torch.where(inm, v, fill), inm


def valleys(g: torch.Tensor, step: int = 2) -> torch.Tensor:
    """Thin dark lines (vessels): how much darker a pixel is than BOTH neighbours
    ``step`` px away along the best of four orientations, relative to the local
    level. A step edge scores zero (one side is darker), so the bright edge of a
    reflection does not read as vessels."""
    p = F.pad(g, (step, step, step, step), mode="replicate")
    H, W = g.shape[-2:]
    at = lambda dy, dx: p[..., step + dy:step + dy + H, step + dx:step + dx + W]
    best = torch.zeros_like(g)
    for dy, dx in ((0, step), (step, 0), (step, step), (step, -step)):
        best = torch.maximum(best, torch.minimum(at(dy, dx), at(-dy, -dx)) - g)
    return best / blur(g, 6.0).clamp_min(1e-3)


@torch.no_grad()
def find_disc(x01: torch.Tensor, grid: int = 96, n_cand: int = 3) -> Dict[str, torch.Tensor]:
    """The optic disc: among the ``n_cand`` strongest disc-scale bright blobs (a
    difference of Gaussians on luminance, damped near the frame edge, searched
    inside the FOV eroded by a tenth of the frame), the one with the most vessel
    signal around it -- the vessels converge on the disc, a reflection washes
    them out. Returns centre ``cx, cy`` (frame coordinates), radius ``r`` (frame
    half-widths), ``prom`` (peak over the blob map's robust spread) and
    ``vessel`` (vessel signal near it over the FOV's mean)."""
    B, dev = x01.shape[0], x01.device
    m = fov_mask(x01).float()
    lum, inm = _pooled(x01.mean(dim=1, keepdim=True), m, grid)
    g, _ = _pooled(x01[:, 1:2], m, grid)
    D = blur(lum, grid * 0.025) - blur(lum, grid * 0.12)
    xx, yy = frame_coords(B, grid, grid, dev)
    D = D * ((0.95 - torch.sqrt(xx ** 2 + yy ** 2)) / 0.2).clamp(0.3, 1.0).unsqueeze(1)
    inner = erode(inm, max(3, int(grid * 0.10)))
    # vessels: darker than the local level AND darker than both sides (a thin line,
    # not the dark side of a bright reflection's edge)
    edge = F.relu(blur(g, 1.5) - g) / blur(g, 6.0).clamp_min(1e-3)
    V = torch.where(inner, torch.sqrt(edge * valleys(blur(g, 0.7), 1) + 1e-12), torch.zeros_like(g))
    v_mean = V.sum(dim=(2, 3)) / inner.sum(dim=(2, 3)).clamp_min(1)
    gyy, gxx = torch.meshgrid(torch.arange(grid, device=dev), torch.arange(grid, device=dev), indexing="ij")
    Dm = torch.where(inner, D, torch.full_like(D, -1e9))
    peaks, pxs, pys, ves = [], [], [], []
    for _ in range(n_cand):
        v, i = Dm.view(B, -1).max(dim=1)
        cy_, cx_ = (i // grid).view(B, 1, 1, 1), (i % grid).view(B, 1, 1, 1)
        d2 = (gxx.view(1, 1, grid, grid) - cx_) ** 2 + (gyy.view(1, 1, grid, grid) - cy_) ** 2
        win = (d2 <= (0.10 * grid) ** 2) & inner
        ves.append(((V * win).sum(dim=(2, 3)) / win.sum(dim=(2, 3)).clamp_min(1) / v_mean.clamp_min(1e-6))[:, 0])
        peaks.append(v)
        pxs.append(cx_.view(B).float())
        pys.append(cy_.view(B).float())
        Dm = torch.where(d2 <= (0.25 * grid) ** 2, torch.full_like(Dm, -1e9), Dm)
    Pk, Vs = torch.stack(peaks, 1), torch.stack(ves, 1)
    score = torch.where(Pk > -1e8, Pk.clamp_min(0) / Pk[:, :1].clamp_min(1e-6) * Vs.clamp_min(1e-3),
                        torch.full_like(Pk, -1.0))
    k = score.argmax(dim=1, keepdim=True)
    gx, gy = torch.stack(pxs, 1).gather(1, k)[:, 0], torch.stack(pys, 1).gather(1, k)[:, 0]
    peak, vessel = Pk.gather(1, k)[:, 0], Vs.gather(1, k)[:, 0]
    Din = torch.where(inner, D, torch.zeros_like(D)).view(B, -1)
    mad = (Din - Din.median(dim=1, keepdim=True).values).abs().median(dim=1).values * 1.4826 + 1e-6
    d2 = (gxx.reshape(1, -1) - gx.view(B, 1)) ** 2 + (gyy.reshape(1, -1) - gy.view(B, 1)) ** 2
    half = (D.view(B, -1) > 0.5 * peak.view(B, 1)) & inner.view(B, -1) & (d2 <= (0.15 * grid) ** 2)
    return {"cx": (gx + 0.5) / grid * 2 - 1, "cy": (gy + 0.5) / grid * 2 - 1,
            "r": torch.sqrt(half.float().sum(dim=1) / math.pi) / (grid / 2),
            "prom": peak / mad, "vessel": vessel}


@torch.no_grad()
def find_fovea(x01: torch.Tensor, d: Dict[str, torch.Tensor], a_f: torch.Tensor, b_f: torch.Tensor,
               grid: int = 96, cone_deg: float = 35.0) -> Dict[str, torch.Tensor]:
    """The fovea: the most prominent dark blob of the illumination-flattened
    luminance 0.45-0.95 FOV radii from the disc, inside a cone from the disc
    through the FOV centre (the fovea sits ~2.5 disc diameters temporal of the
    disc, on the far side of the centre or near it). Returns its centre, ``prom``
    and ``dfd``, the disc-fovea distance in FOV radii -- the angular-scale
    yardstick (disc size reads the phone's bloomed discs as larger than they are)."""
    B, dev = x01.shape[0], x01.device
    m = fov_mask(x01).float()
    flat = x01 / illum_field(illum_coeffs(x01), x01.shape[-2], x01.shape[-1]).clamp(0.2, 5.0)
    lum, inm = _pooled(flat.mean(dim=1, keepdim=True), m, grid)
    Dk = blur(lum, grid * 0.10) - blur(lum, grid * 0.025)                 # dark blob -> positive
    xx, yy = frame_coords(B, grid, grid, dev)
    ux = (xx - d["cx"].view(B, 1, 1)) / a_f.view(B, 1, 1)
    uy = (yy - d["cy"].view(B, 1, 1)) / b_f.view(B, 1, 1)
    dist = torch.sqrt(ux ** 2 + uy ** 2)
    tx, ty = -d["cx"] / a_f, -d["cy"] / b_f                               # disc -> FOV centre
    cos = (ux * tx.view(B, 1, 1) + uy * ty.view(B, 1, 1)) / (dist * torch.sqrt(tx ** 2 + ty ** 2).view(B, 1, 1)).clamp_min(1e-6)
    inner = erode(inm, max(3, int(grid * 0.06)))
    win = ((dist >= 0.45) & (dist <= 0.95) & (cos >= math.cos(math.radians(cone_deg)))).unsqueeze(1) & inner
    v, i = torch.where(win, Dk, torch.full_like(Dk, -1e9)).view(B, -1).max(dim=1)
    Din = torch.where(inner, Dk, torch.zeros_like(Dk)).view(B, -1)
    mad = (Din - Din.median(dim=1, keepdim=True).values).abs().median(dim=1).values * 1.4826 + 1e-6
    return {"fx": (i % grid).float().add(0.5) / grid * 2 - 1, "fy": (i // grid).float().add(0.5) / grid * 2 - 1,
            "prom": torch.where(v > -1e8, v / mad, torch.zeros_like(v)),
            "dfd": dist.view(B, -1).gather(1, i.view(B, 1))[:, 0]}


def disc_confidence(d: Dict[str, torch.Tensor], min_prom: float = 6.0) -> torch.Tensor:
    """0..1 from the blob's prominence (true discs on the samples 7-137, the one false
    pick 6.2), softened when no vessels meet it; a blown-out phone disc washes its
    own vessels out (true discs read 0.33-3.0 on the vessel score), so the vessel
    score only damps, it never vetoes. Zero near the frame edge."""
    pos = torch.sqrt(d["cx"] ** 2 + d["cy"] ** 2)
    return (((d["prom"] - min_prom) / min_prom).clamp(0, 1) * (d["vessel"] / 0.5).clamp(0, 1)
            * (pos < 0.95).float())


def _illum_basis(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
    r2 = xx ** 2 + yy ** 2
    return torch.stack([torch.ones_like(xx), r2, r2 ** 2, xx, yy], dim=1)


@torch.no_grad()
def illum_coeffs(x01: torch.Tensor, grid: int = 32, r_max: float = 0.85, disc_q: float = 0.96,
                 with_residual: bool = False):
    """[B,3,5] per-channel model log I = c . [1, r^2, r^4, x, y] (frame coordinates),
    fitted on a coarse grid inside ``r_max`` with the brightest cells (the disc)
    left out, robustly (three Cauchy-reweighted passes). ``with_residual`` also
    returns [B] the RMS of what the model leaves in the luminance log field,
    smoothed by a Gaussian of 1/16 of the FOV diameter (finer, it reads anatomy):
    the blotchy uneven illumination of a handheld capture (0.15 handheld vs 0.06
    tabletop on the samples)."""
    B, C, H, W = x01.shape
    m = fov_mask(x01).float()
    ms = F.adaptive_avg_pool2d(m, grid)
    val = F.adaptive_avg_pool2d(x01 * m, grid) / ms.clamp_min(1e-6)
    xx, yy = frame_coords(B, grid, grid, x01.device)
    ok = (ms[:, 0] > 0.99) & (torch.sqrt(xx ** 2 + yy ** 2) <= r_max)
    lum = val.mean(dim=1)
    thr = _masked_quantiles(lum.view(B, -1), ok.view(B, -1), [disc_q])[0]
    thr = torch.where(torch.isfinite(thr), thr, lum.view(B, -1).amax(dim=1))
    disc = F.max_pool2d((lum > thr.view(B, 1, 1)).float().unsqueeze(1), 3, stride=1, padding=1)[:, 0] > 0.5
    ok = ok & ~disc
    Phi = _illum_basis(xx, yy).reshape(B, 5, -1)
    L = torch.log(val.clamp_min(1e-3)).reshape(B, C, -1)
    base = ok.reshape(B, 1, -1).float().expand(B, C, -1)
    w = base.clone()
    ridge = torch.eye(5, device=x01.device) * 1e-3
    c = torch.zeros(B, C, 5, device=x01.device)
    for _ in range(3):
        Amat = torch.einsum("bkn,bcn,bjn->bckj", Phi, w, Phi) + ridge
        rhs = torch.einsum("bkn,bcn,bcn->bck", Phi, w, L)
        c = torch.linalg.solve(Amat.cpu().double(), rhs.unsqueeze(-1).cpu().double()).squeeze(-1).float().to(x01.device)
        res = L - torch.einsum("bck,bkn->bcn", c, Phi)
        s = (res.abs() * base).sum(-1, keepdim=True) / base.sum(-1, keepdim=True).clamp_min(1) * 1.4826 + 1e-3
        w = base / (1.0 + (res / (2.5 * s)) ** 2)
    if not with_residual:
        return c
    rl = blur(res.mean(dim=1).view(B, 1, grid, grid), grid / 16).view(B, -1)       # sigma ~ D/16
    okf = ok.view(B, -1).float()
    mu = (rl * okf).sum(1) / okf.sum(1).clamp_min(1)
    rms = torch.sqrt((((rl - mu.view(B, 1)) ** 2) * okf).sum(1) / okf.sum(1).clamp_min(1))
    return c, rms


def illum_field(c: torch.Tensor, H: int, W: int, terms: slice = slice(1, None)) -> torch.Tensor:
    """exp of the model's ``terms`` (default: all but the constant) -> [B,3,H,W]."""
    B = c.shape[0]
    xx, yy = frame_coords(B, H, W, c.device)
    return torch.exp(torch.einsum("bck,bkhw->bchw", c[..., terms], _illum_basis(xx, yy)[:, terms]))


@torch.no_grad()
def robust_colour(x01: torch.Tensor, m: torch.Tensor, step: int = 3) -> Dict[str, torch.Tensor]:
    """The fundus background (FOV pixels between the 2nd and 96th luminance
    percentiles: no disc, no vessel cores): per-channel median ``med`` and robust
    spread ``scale`` (IQR / 1.349), and the luminance spread ``lscale``."""
    v, k = _strided(x01, m, step)
    lum = v.mean(dim=1, keepdim=True)
    lo_hi = _masked_quantiles(lum, k, [0.02, 0.96])                                  # [2,B,1]
    keep = k & (lum >= lo_hi[0].unsqueeze(-1)) & (lum <= lo_hi[1].unsqueeze(-1))
    q = _masked_quantiles(v, keep.expand_as(v), [0.25, 0.5, 0.75])                    # [3,B,3]
    ql = _masked_quantiles(lum, keep, [0.25, 0.75])[..., 0]                           # [2,B]
    return {"med": q[1], "scale": ((q[2] - q[0]) / 1.349).clamp_min(1e-3),
            "lscale": ((ql[1] - ql[0]) / 1.349).clamp_min(1e-3)}


@torch.no_grad()
def colour_moments(x01: torch.Tensor, m: torch.Tensor, step: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean [B,3] and covariance [B,3,3] of the fundus background (FOV pixels
    between the 2nd and 96th luminance percentiles)."""
    v, k = _strided(x01, m, step)
    lum = v.mean(dim=1, keepdim=True)
    lo_hi = _masked_quantiles(lum, k, [0.02, 0.96])
    keep = (k & (lum >= lo_hi[0].unsqueeze(-1)) & (lum <= lo_hi[1].unsqueeze(-1))).float()       # [B,1,N]
    n = keep.sum(dim=-1).clamp_min(2)
    mu = (v * keep).sum(dim=-1) / n
    d = (v - mu.unsqueeze(-1)) * keep
    return mu, torch.einsum("bin,bjn->bij", d, d) / (n - 1).view(-1, 1, 1)


def _dog(x: torch.Tensor, s1: float, s2: float) -> torch.Tensor:
    return blur(x, s1) - blur(x, s2)


def _centre_zone(B: int, H: int, W: int, disc_xy: Optional[torch.Tensor], device) -> torch.Tensor:
    """[B,1,H,W] the illumination-flat centre: inside half the aperture radius, at
    least 0.3 radius (0.15 FOV diameters) from the optic disc."""
    xx, yy = frame_coords(B, H, W, device)
    zone = (xx ** 2 + yy ** 2) < 0.25
    if disc_xy is not None:
        zone = zone & (((xx - disc_xy[:, 0].view(B, 1, 1)) ** 2 + (yy - disc_xy[:, 1].view(B, 1, 1)) ** 2) > 0.09)
    return zone.unsqueeze(1)


@torch.no_grad()
def coupling_stats(xf: torch.Tensor, m: torch.Tensor, zone: torch.Tensor) -> Dict[str, torch.Tensor]:
    """What the coupled colour stage matches, on an illumination-flattened photo:
    centre-zone channel means [B,3], the band-pass (DoG 1 -> 8 px) correlation of R
    and of B with G [B,2], and each channel's fine amplitude (SD of DoG 1 -> 4 px) [B,3]."""
    me = erode(m, 25).float()
    zf = (zone & m).float()
    nz = zf.sum(dim=(2, 3)).clamp_min(1)
    ctr = (xf * zf).sum(dim=(2, 3)) / nz
    n = me.sum(dim=(1, 2, 3)).clamp_min(1).view(-1, 1)
    bd = _dog(xf, 1.0, 8.0)
    bd = bd - (bd * me).sum(dim=(2, 3), keepdim=True) / n.view(-1, 1, 1, 1)
    v = ((bd ** 2) * me).sum(dim=(2, 3)) / n
    cov = lambda i: ((bd[:, i] * bd[:, 1]) * me[:, 0]).sum(dim=(1, 2)) / n[:, 0]
    rho = torch.stack([cov(0), cov(2)], 1) / torch.sqrt(v[:, [0, 2]] * v[:, 1:2]).clamp_min(1e-12)
    fd = _dog(xf, 1.0, 4.0)
    fd = fd - (fd * me).sum(dim=(2, 3), keepdim=True) / n.view(-1, 1, 1, 1)
    amp = torch.sqrt(((fd ** 2) * me).sum(dim=(2, 3)) / n)
    return {"ctr": ctr, "band_rho": rho, "fine_amp": amp}


@torch.no_grad()
def coupled_colour(yf: torch.Tensor, m: torch.Tensor, zone: torch.Tensor, T: Dict[str, torch.Tensor],
                   kappa: Tuple[float, float] = (0.4, 1.4), s: float = 1.0, toe: float = 8 / 255) -> torch.Tensor:
    """The phone's colour on an illumination-flattened photo, green left alone.
    (1) R and B take on G's structure: C <- (1-w) C + w mu_C (G / mu_G)^kappa_C (a
    positive surrogate, so a dark lesion's blue is not crushed to 0), w the smallest
    weight that brings the band-pass correlation with G to the drawn photo's; (2)
    ratio-preserving gains: G and B scaled to the drawn photo's centre means (lesion /
    background ratios unchanged), R = a R + b with a the fine-amplitude ratio; (3) a
    soft toe below 8/255. Whole-image moment matching (3x3 included) raised lesion and
    vessel contrast 1.3-2.2x on the samples where real phones raise it 1.1-1.2x."""
    B = yf.shape[0]
    src = coupling_stats(yf, m, zone)
    mu = src["ctr"].clamp_min(1e-3)
    G = yf[:, 1:2].clamp_min(0)
    me = erode(m, 25).float()
    n = me.sum(dim=(1, 2, 3)).clamp_min(1)
    out = yf.clone()
    bG = _dog(G, 1.0, 8.0)
    for j, (c, kap) in enumerate(((0, kappa[0]), (2, kappa[1]))):
        S = mu[:, c].view(B, 1, 1, 1) * (G / mu[:, 1].view(B, 1, 1, 1)) ** kap
        bC, bS = _dog(yf[:, c:c + 1], 1.0, 8.0), _dog(S, 1.0, 8.0)
        cen = lambda t: t - (t * me).sum(dim=(2, 3), keepdim=True) / n.view(-1, 1, 1, 1)
        bC, bS, bg = cen(bC), cen(bS), cen(bG)
        mom = lambda a, b: ((a * b) * me).sum(dim=(1, 2, 3)) / n
        Vcc, Vss, Vcs, Vcg, Vsg, Vgg = mom(bC, bC), mom(bS, bS), mom(bC, bS), mom(bC, bg), mom(bS, bg), mom(bg, bg)
        w = torch.linspace(0, 1, 41, device=yf.device).view(1, -1)
        var = ((1 - w) ** 2 * Vcc.view(-1, 1) + w ** 2 * Vss.view(-1, 1) + 2 * w * (1 - w) * Vcs.view(-1, 1))
        corr = ((1 - w) * Vcg.view(-1, 1) + w * Vsg.view(-1, 1)) / torch.sqrt(var * Vgg.view(-1, 1)).clamp_min(1e-12)
        reach = corr >= T["band_rho"][:, j].view(-1, 1)
        k = torch.where(reach.any(1), reach.float().argmax(1), torch.full((B,), 40, device=yf.device))
        wk = (w.view(-1)[k] * s).view(B, 1, 1, 1)
        out[:, c:c + 1] = (1 - wk) * yf[:, c:c + 1] + wk * S
    cs = coupling_stats(out, m, zone)
    gain = (T["ctr"] / cs["ctr"].clamp_min(1e-3)).clamp(0.2, 5.0)                     # [B,3]
    a_r = (T["fine_amp"][:, 0] / cs["fine_amp"][:, 0].clamp_min(1e-4)).clamp(0.5, 2.0)
    b_r = T["ctr"][:, 0] - a_r * cs["ctr"][:, 0]
    tgt = out.clone()
    tgt[:, 1:2] = out[:, 1:2] * gain[:, 1].view(B, 1, 1, 1)
    tgt[:, 2:3] = out[:, 2:3] * gain[:, 2].view(B, 1, 1, 1)
    tgt[:, 0:1] = a_r.view(B, 1, 1, 1) * out[:, 0:1] + b_r.view(B, 1, 1, 1)
    res = out + s * (tgt - out)
    return torch.where(res < toe, toe * torch.exp((res - toe) / toe), res)


def mk_map(cov_s: torch.Tensor, cov_t: torch.Tensor, eps: float = 1e-5,
           lim: Tuple[float, float] = (0.25, 3.0)) -> torch.Tensor:
    """[B,3,3] linear map taking colour covariance ``cov_s`` to ``cov_t`` with the
    least change (Gaussian optimal transport, Monge-Kantorovich): T = S^-1/2
    (S^1/2 C S^1/2)^1/2 S^-1/2, symmetric, its gains held to ``lim``. Unlike a
    per-channel map it sets the channels' correlation: the phone's blue carries
    green's structure."""
    dev = cov_s.device
    S = cov_s.cpu().double() + eps * torch.eye(3, dtype=torch.float64)
    C = cov_t.cpu().double() + eps * torch.eye(3, dtype=torch.float64)
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(eps)
    sh = V @ torch.diag_embed(w.sqrt()) @ V.transpose(-1, -2)
    smh = V @ torch.diag_embed(w.rsqrt()) @ V.transpose(-1, -2)
    wm, Vm = torch.linalg.eigh(sh @ C @ sh)
    mh = Vm @ torch.diag_embed(wm.clamp_min(0).sqrt()) @ Vm.transpose(-1, -2)
    T = smh @ mh @ smh
    wt, Vt = torch.linalg.eigh(0.5 * (T + T.transpose(-1, -2)))
    T = Vt @ torch.diag_embed(wt.clamp(*lim)) @ Vt.transpose(-1, -2)
    return T.float().to(dev)


def _inner(m: torch.Tensor, k: int) -> torch.Tensor:
    return erode(m, k).float()


def band_energy(g: torch.Tensor, m: torch.Tensor, me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B] variance inside the FOV of a one-channel batch's sigma 1 - 2.5 px
    difference-of-Gaussians band: vessel edges and lesion-scale detail, little
    pixel noise."""
    d = blur(g, 1.0) - blur(g, 2.5)
    me = _inner(m, 9) if me is None else me.float()
    n = me.sum(dim=(1, 2, 3)).clamp(min=1)
    mu = (d * me).sum(dim=(1, 2, 3)) / n
    return (((d - mu.view(-1, 1, 1, 1)) ** 2) * me).sum(dim=(1, 2, 3)) / n


def noise_level(g: torch.Tensor, m: torch.Tensor, flat_q: float = 0.3, step: int = 2,
                me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B] robust SD (1.4826 x MAD) of g - blur(g, 1 px) over the flattest ``flat_q``
    of the FOV (lowest gradient energy at 2 px): a pixel-noise reading. Its
    absolute scale depends on the noise's correlation, so it is only ever
    compared with itself (the bank's reading against the render's)."""
    r = g - blur(g, 1.0)
    gs = blur(g, 2.0)
    gy, gx = torch.gradient(gs, dim=(2, 3))
    e = blur(gx ** 2 + gy ** 2, 2.0)
    me = erode(m, 9) if me is None else me
    rv, kk = _strided(r, me, step)
    ev, _ = _strided(e, me, step)
    thr = _masked_quantiles(ev, kk, [flat_q])[0]                                       # [B,1]
    sel = kk & (ev <= thr.unsqueeze(-1))
    med = _masked_quantiles(rv, sel, [0.5])[0]
    mad = _masked_quantiles((rv - med.unsqueeze(-1)).abs(), sel, [0.5])[0]
    return (1.4826 * mad)[:, 0]


def _band(x01: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    return x01 - blur(x01, sigma)


def structure_corr(x01: torch.Tensor, m: torch.Tensor, sigma: float = 3.0,
                   band: Optional[torch.Tensor] = None, me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B,2] correlation of the R and of the B channel's structure (x - blur(x, 3 px):
    vessels, lesions, texture) with the G channel's, over the FOV. A phone ISP's
    colour matrix and demosaic make R and B carry G's vessel pattern: 0.84 / 0.88
    handheld vs 0.61 / 0.56 tabletop on the samples, a near-perfect fingerprint."""
    d = _band(x01, sigma) if band is None else band
    me = (erode(m, 9) if me is None else me).float()
    n = me.sum(dim=(1, 2, 3)).clamp_min(1)
    d = d - (d * me).sum(dim=(2, 3), keepdim=True) / n.view(-1, 1, 1, 1)
    v = ((d ** 2) * me).sum(dim=(2, 3)) / n.view(-1, 1)                           # [B,3]
    cov = lambda i: ((d[:, i] * d[:, 1]) * me[:, 0]).sum(dim=(1, 2)) / n
    return torch.stack([cov(0), cov(2)], dim=1) / torch.sqrt(v[:, [0, 2]] * v[:, 1:2]).clamp_min(1e-12)


def mix_structure(y: torch.Tensor, m: torch.Tensor, target: torch.Tensor, s: float = 1.0,
                  me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """R and B structure moved toward G's until their correlation with G reaches
    ``target`` [B,2] (never lowered): d_c <- ((1-l) d_c + l k d_G) / norm, k keeping
    the channel's own structure amplitude, l the smallest mix that gets there.
    Luminance contrast of every structure is kept; small lesions' hue moves toward
    neutral, as on the phone."""
    B = y.shape[0]
    d = _band(y)
    me = (erode(m, 9) if me is None else me).float()
    n = me.sum(dim=(1, 2, 3)).clamp_min(1)
    sd = torch.sqrt(((d ** 2) * me).sum(dim=(2, 3)) / n.view(-1, 1)).clamp_min(1e-9)   # [B,3]
    rho0 = structure_corr(y, m, band=d, me=me).clamp(-0.99, 0.99)
    lam = torch.linspace(0, 1, 41, device=y.device).view(1, 1, -1)
    r0 = rho0.unsqueeze(-1)
    f = ((1 - lam) * r0 + lam) / torch.sqrt((1 - lam) ** 2 + lam ** 2 + 2 * lam * (1 - lam) * r0)    # [B,2,L]
    reach = f >= target.unsqueeze(-1)
    idx = torch.where(reach.any(-1), reach.float().argmax(-1), torch.full_like(rho0, 40, dtype=torch.long))
    l = (lam.view(-1)[idx] * s).clamp(0, 1)                                      # [B,2]
    out = y.clone()
    for j, c in ((0, 0), (1, 2)):
        lj = l[:, j].view(B, 1, 1)
        k = (sd[:, c] / sd[:, 1]).view(B, 1, 1)
        norm = torch.sqrt((1 - lj) ** 2 + lj ** 2 + 2 * lj * (1 - lj) * rho0[:, j].view(B, 1, 1)).clamp_min(1e-3)
        out[:, c] = y[:, c] - d[:, c] + ((1 - lj) * d[:, c] + lj * k * d[:, 1]) / norm
    return out


def shadow_fraction(x01: torch.Tensor, m: torch.Tensor, grid: int = 96) -> torch.Tensor:
    """[B] fraction of the FOV whose smoothed luminance (sigma ~ D/32) is below half
    its median: how much retina sits in shadow."""
    lum, inm = _pooled(x01.mean(dim=1, keepdim=True), m.float(), grid)
    lb = blur(lum, grid / 32).view(x01.shape[0], -1)
    ok = inm.view(x01.shape[0], -1)
    med = _masked_quantiles(lb.unsqueeze(1), ok.unsqueeze(1), [0.5])[0]
    return ((lb < 0.5 * med) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)


def fine_chroma(x01: torch.Tensor, m: torch.Tensor, me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B] chroma-to-luma ratio of the fine band (x - blur(x, 1 px)) over the FOV:
    RMS of each channel's deviation from the channel-mean residual over the RMS of
    that mean. A phone ISP's chroma noise reduction leaves fine detail almost pure
    luma (0.31 handheld vs 0.58 tabletop in model view)."""
    r = x01 - blur(x01, 1.0)
    lum = r.mean(dim=1, keepdim=True)
    me = (erode(m, 9) if me is None else me).float()
    n = me.sum(dim=(1, 2, 3)).clamp_min(1)
    ch = (((r - lum) ** 2).mean(dim=1, keepdim=True) * me).sum(dim=(1, 2, 3)) / n
    lu = ((lum ** 2) * me).sum(dim=(1, 2, 3)) / n
    return torch.sqrt(ch / lu.clamp_min(1e-12))


def disc_highlight(x01: torch.Tensor, d: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The handheld photo's disc: how blown out it is (fraction near-clipped, any
    channel >= 0.92, saturating at 30 % -> 1) and its bright core's colour (per
    channel 90th percentile inside the disc)."""
    B, C, H, W = x01.shape
    xx, yy = frame_coords(B, H, W, x01.device)
    inside = ((xx - d["cx"].view(B, 1, 1)) ** 2 + (yy - d["cy"].view(B, 1, 1)) ** 2
              <= d["r"].clamp_min(0.03).view(B, 1, 1) ** 2).unsqueeze(1)
    n = inside.float().sum(dim=(1, 2, 3)).clamp_min(1)
    clip = ((x01.amax(dim=1, keepdim=True) >= 0.92) & inside).float().sum(dim=(1, 2, 3)) / n
    core = _masked_quantiles(x01.reshape(B, C, -1), inside.reshape(B, 1, -1).expand(B, C, -1), [0.9])[0]
    return (clip / 0.3).clamp(0, 1), torch.nan_to_num(core, nan=0.0)


@torch.no_grad()
def rim_profile(x01: torch.Tensor, c: torch.Tensor, band: Tuple[float, float] = (0.86, 0.97)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The rim against the photo's own inner illumination model, per sector: the
    additive excess per channel (haze, >= 0) [B,3,S] and the luminance ratio where
    the rim is darker than the model (<= 1) [B,S]."""
    B, C, H, W = x01.shape
    xx, yy = frame_coords(B, H, W, x01.device)
    r = torch.sqrt(xx ** 2 + yy ** 2)
    sector = ((torch.atan2(yy, xx) + math.pi) / (2 * math.pi) * N_SECTORS).long().clamp(0, N_SECTORS - 1)
    sel = ((r >= band[0]) & (r <= band[1])).unsqueeze(1) & fov_mask(x01)
    pred = illum_field(c, H, W, slice(0, None))
    oh = F.one_hot(sector, N_SECTORS).permute(0, 3, 1, 2).float() * sel.float()        # [B,S,H,W]
    n = oh.sum(dim=(2, 3)).clamp_min(1)
    obs = torch.einsum("bchw,bshw->bcs", x01, oh) / n.unsqueeze(1)
    mod = torch.einsum("bchw,bshw->bcs", pred, oh) / n.unsqueeze(1)
    add = (obs - mod).clamp_min(0)
    ratio = (obs.mean(dim=1) / mod.mean(dim=1).clamp_min(1e-3)).clamp(0.2, 1.0)
    empty = n < 4
    return add.masked_fill(empty.unsqueeze(1), 0.0), ratio.masked_fill(empty, 1.0)


@torch.no_grad()
def describe(x01: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One bank row per photo (model view, [B,3,H,W] in [0,1])."""
    m = fov_mask(x01)
    me = erode(m, max(3, x01.shape[-1] // 64))
    g = x01.mean(dim=1, keepdim=True)
    d = find_disc(x01)
    good = disc_confidence(d) > 0.25
    clip, hcol = disc_highlight(x01, d)
    c, blotch = illum_coeffs(x01, with_residual=True)
    add, ratio = rim_profile(x01, c)
    shape = fov_shape(x01)
    nan = torch.full_like(d["r"], float("nan"))
    a_f = torch.where(shape["reliable"], shape["a_f"], torch.ones_like(shape["a_f"]))
    b_f = a_f / torch.where(shape["reliable"], shape["aspect"], torch.ones_like(shape["aspect"]))
    fv = find_fovea(x01, d, a_f, b_f)
    flat = x01 / illum_field(c, x01.shape[-2], x01.shape[-1]).clamp(0.2, 5.0)
    rc = robust_colour(flat, me)
    mu, cov = colour_moments(flat, me)
    disc_xy = torch.where(good.view(-1, 1), torch.stack([d["cx"], d["cy"]], 1), torch.full((len(good), 2), 9.0))
    cpl = coupling_stats(flat, m, _centre_zone(x01.shape[0], x01.shape[-2], x01.shape[-1], disc_xy, x01.device))
    return {**rc, "mu": mu, "cov": cov, "ctr": cpl["ctr"], "band_rho": cpl["band_rho"], "fine_amp": cpl["fine_amp"],
            "illum": c, "detail": band_energy(g, m) / rc["lscale"] ** 2, "noise": noise_level(g, m),
            "chroma": fine_chroma(x01, m), "shadow": shadow_fraction(x01, me), "xcorr": structure_corr(x01, m),
            "disc_dist": torch.where(good, torch.sqrt(d["cx"] ** 2 + d["cy"] ** 2), nan),
            "disc_x": torch.where(good, d["cx"], nan),
            "disc_r": torch.where(good, d["r"], nan), "bloom": torch.where(good, clip, torch.zeros_like(clip)),
            "hcol": hcol, "fov_r": shape["radius"], "aspect": shape["aspect"],
            "aspect_ok": shape["reliable"].float(), "rim_add": add, "rim_ratio": ratio, "blotch": blotch,
            "disc_r_fov": torch.where(good, d["r"] / torch.sqrt(a_f * b_f), nan),
            "dfd": torch.where(good & (fv["prom"] > 4), fv["dfd"], nan)}


BANK_KEYS = ("med", "scale", "lscale", "mu", "cov", "ctr", "band_rho", "fine_amp", "illum", "blotch", "detail", "noise", "chroma", "shadow",
             "xcorr", "disc_dist", "disc_x", "disc_r", "bloom", "hcol", "fov_r", "rim_add", "rim_ratio")


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@dataclass
class PhoneProfile:
    """What ``fit`` measured: the handheld bank (one row per photo), the global
    stage parameters, where it came from, and the fit report."""
    image_size: int
    bank: Dict[str, np.ndarray]
    params: Dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    source: Dict[str, object] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)
    format: str = FORMAT

    def __post_init__(self) -> None:
        missing = [k for k in BANK_KEYS if k not in self.bank]
        if missing:
            raise ValueError(f"phone profile bank is missing {missing}")
        self.bank = {k: np.asarray(v, dtype=np.float32) for k, v in self.bank.items()}
        n = {len(v) for v in self.bank.values()}
        if len(n) != 1 or 0 in n:
            raise ValueError(f"phone profile bank rows disagree or are empty: {n}")
        self.params = {**DEFAULT_PARAMS, **self.params}

    def __len__(self) -> int:
        return len(self.bank["med"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {"format": self.format, "image_size": self.image_size, "params": self.params,
                "source": self.source, "report": self.report,
                "bank": {k: np.round(v, 7).tolist() for k, v in self.bank.items()}}
        with open(path, "w") as f:
            json.dump(blob, f, allow_nan=True)

    @classmethod
    def load(cls, path: str) -> "PhoneProfile":
        with open(path) as f:
            blob = json.load(f)
        if blob.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} profile (format {blob.get('format')!r})")
        return cls(image_size=int(blob["image_size"]), bank=blob["bank"], params=blob["params"],
                   source=blob.get("source", {}), report=blob.get("report", {}))


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
def _device_generator(device, seed: int) -> Optional[torch.Generator]:
    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    except (RuntimeError, TypeError):                 # a backend without device generators
        return None


def _detail_op(x: torch.Tensor, op: Tuple[float, float]) -> torch.Tensor:
    a, r = op
    if a < 0:
        return x + (-a) * (x - blur(x, r))
    return blur(x, a) if a > 0 else x


def match_detail(x01: torch.Tensor, target: torch.Tensor, m: torch.Tensor,
                 ops: Sequence[Tuple[float, float]] = DETAIL_OPS, me: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per image, the point on the blur ladder whose band energy is ``target``
    (the two bracketing outputs mixed linearly in log energy); an image already
    at or below the target is left alone, and nothing is blurred past the
    ladder's end."""
    B, dev = x01.shape[0], x01.device
    g = x01.mean(dim=1, keepdim=True)
    me = erode(m, 9) if me is None else me
    le = torch.stack([band_energy(_detail_op(g, op), m, me) for op in ops]).clamp_min(1e-12).log()   # [S,B]
    lt = target.to(dev).float().clamp_min(1e-12).log()
    below = le <= lt.view(1, -1)
    j = torch.where(below.any(0), below.float().argmax(0), torch.full((B,), len(ops) - 1, device=dev)).clamp(min=1)
    j0 = j - 1
    l0, l1 = le.gather(0, j0.view(1, -1))[0], le.gather(0, j.view(1, -1))[0]
    t = ((l0 - lt) / (l0 - l1).clamp_min(1e-6)).clamp(0, 1)
    t = torch.where(below[0], torch.zeros_like(t), t)
    out = torch.zeros_like(x01)
    for k in torch.unique(torch.cat([j0, j])).tolist():
        w = torch.where(j0 == k, 1 - t, torch.zeros_like(t)) + torch.where(j == k, t, torch.zeros_like(t))
        sel = w > 0
        if bool(sel.any()):
            out[sel] += _detail_op(x01[sel], ops[k]) * w[sel].view(-1, 1, 1, 1)
    return out


def _mirror_index(n: int, device) -> torch.Tensor:
    """Sector k of a rim profile -> its left-right mirror (angle theta -> pi - theta)."""
    th = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n * 2 * math.pi - math.pi
    mth = torch.remainder(math.pi - th + math.pi, 2 * math.pi)                    # in [0, 2 pi)
    return torch.round(mth / (2 * math.pi) * n - 0.5).long() % n


class PhoneSimulator(Stylizer):
    """Tabletop -> handheld, ``[B,3,H,W]`` in [0,1] -> ``[B,3,S,S]`` in [0,1].

    ``out_size`` (default: the input size) is the model's image size; training
    passes a ``supersample``-times larger view. ``strength`` scales every
    photometric change and the geometric offsets (0 keeps only the handheld
    aperture). ``stages`` picks a subset of :data:`STAGES` (ablations)."""
    name = "phone_sim"

    def __init__(self, profile: PhoneProfile, strength: float = 1.0, stages: Sequence[str] = STAGES,
                 out_size: Optional[int] = None) -> None:
        bad = sorted(set(stages) - set(STAGES))
        if bad:
            raise ValueError(f"unknown phone_sim stages {bad}; known: {STAGES}")
        if not 0.0 <= strength <= 1.5:
            raise ValueError(f"phone_sim strength {strength} outside [0, 1.5]")
        self.profile, self.strength, self.stages, self.out_size = profile, float(strength), tuple(stages), out_size
        self.p = profile.params
        self._bank = {k: torch.from_numpy(np.asarray(v)) for k, v in profile.bank.items()}
        dd = self._bank["disc_dist"]
        self.disc_dist = float(dd[torch.isfinite(dd)].median()) if torch.isfinite(dd).any() \
            else float(self.p["disc_dist"])
        si = self.p.get("src_illum")
        self.src_illum = torch.tensor(si, dtype=torch.float32).view(1, 3, 5) if si is not None else None
        self._last: Optional[Dict[str, torch.Tensor]] = None

    def check_size(self, out_size: int) -> None:
        """The bank's detail and noise readings are per pixel at the profile's size."""
        if abs(out_size - self.profile.image_size) > 0.05 * self.profile.image_size:
            raise ValueError(f"phone profile measured at {self.profile.image_size} px, rendering at "
                             f"{out_size} px: refit it with --image-size {out_size}")

    # ---- parameters ------------------------------------------------------ #
    def sample(self, n: int, gen: torch.Generator) -> Dict[str, torch.Tensor]:
        """Per-image draws (CPU): a bank row and the geometric jitter."""
        p = self.p
        return {"idx": torch.randint(len(self.profile), (n,), generator=gen),
                "zoom": float(p["zoom"]) * torch.exp(float(p["zoom_sd"]) * torch.randn(n, generator=gen)),
                "rot": math.radians(float(p["rot_sd_deg"])) * torch.randn(n, generator=gen),
                "jit": float(p["jitter"]) * torch.randn(n, 2, generator=gen),
                "seed": torch.randint(0, 2 ** 62, (1,), generator=gen)}

    def _field(self, x01: torch.Tensor, prm: Dict[str, torch.Tensor], T: Dict[str, torch.Tensor]):
        B, dev, s = x01.shape[0], x01.device, self.strength
        shape = fov_shape(x01)
        rel = shape["reliable"]
        aspect = torch.where(rel, shape["aspect"], torch.full_like(shape["aspect"], float(self.p["source_aspect"])))
        a_f = torch.where(rel, shape["a_f"], torch.ones_like(aspect))
        b_f = a_f / aspect
        centre = torch.stack([torch.zeros_like(aspect), torch.where(rel, shape["yc"], torch.zeros_like(aspect))], 1)
        rot = s * prm["rot"].to(dev)
        fov = fov_mask(x01)
        z = min_zoom(fov, 1 + s * (prm["zoom"].to(dev) - 1), a_f, b_f, centre, rot)
        d = find_disc(x01)
        conf = disc_confidence(d, float(self.p["disc_prom"]))
        ab = torch.stack([a_f, b_f], dim=1)
        dv = (torch.stack([d["cx"], d["cy"]], dim=1) - centre) / ab                    # FOV units
        tgt = torch.where(torch.isfinite(T["disc_dist"]), T["disc_dist"], torch.full_like(z, self.disc_dist))
        lam = (tgt / (z * dv.norm(dim=1)).clamp_min(1e-6)).clamp(max=1.0)
        want = s * (torch.where((conf > 0).view(B, 1), dv * (1 - lam).view(B, 1), torch.zeros_like(dv))
                    + prm["jit"].to(dev)) * ab
        offset = centre + safe_shift(fov, want, z, a_f, b_f, centre, rot)
        p = z.view(B, 1) * (torch.stack([d["cx"], d["cy"]], dim=1) - offset) / ab        # disc, rotated frame
        cr, sr = torch.cos(-rot), torch.sin(-rot)
        disc_xy = torch.stack([cr * p[:, 0] - sr * p[:, 1], sr * p[:, 0] + cr * p[:, 1]], dim=1)
        return {"zoom": z, "a_f": a_f, "b_f": b_f, "offset": offset, "rot": rot,
                "radius": torch.ones_like(z), "disc_xy": disc_xy,
                "disc_r": (z * d["r"] / torch.sqrt(a_f * b_f)).clamp(0.06, 0.22), "disc_conf": conf}

    # ---- the chain ------------------------------------------------------- #
    @torch.no_grad()
    def render(self, x01: torch.Tensor, prm: Dict[str, torch.Tensor], out_size: Optional[int] = None) -> torch.Tensor:
        B, C, Hin, Win = x01.shape
        H = W = int(out_size or self.out_size or Hin)
        self.check_size(H)
        dev, s, p = x01.device, self.strength, self.p
        x01 = x01.float()
        T = {k: v[prm["idx"]].to(dev) for k, v in self._bank.items()}
        seed = int(prm["seed"][0]) if "seed" in prm else 0
        gen_dev = _device_generator(dev, seed)            # None: the backend's default generator
        geo = None
        if "field" in self.stages:
            geo = self._field(x01, prm, T)
            self._last = {k: v.detach() for k, v in geo.items()}
            y = warp(x01, geo["zoom"], geo["a_f"], geo["b_f"], geo["offset"], geo["rot"], (H, W))
            A = aperture(geo["radius"], float(p["rim_width"]), H, W)
            y = y * A
        else:
            y = x01 if Hin == H else F.interpolate(x01, size=(H, W), mode="bilinear", antialias=True,
                                                   align_corners=False)
            A = fov_mask(y).float()
        inside = A > 0.5
        me9 = erode(inside, 9)
        if geo is not None:
            # the drawn photo's horizontal light and rim belong to its eye's side: mirror
            # them when its disc sits on the other side from this one's
            flip = (torch.isfinite(T["disc_x"]) & (geo["disc_conf"] > 0)
                    & (torch.sign(torch.nan_to_num(T["disc_x"])) != torch.sign(geo["disc_xy"][:, 0])))
            if bool(flip.any()):
                T = dict(T)
                ill = T["illum"].clone()
                ill[flip, :, 3] = -ill[flip, :, 3]
                T["illum"] = ill
                mi = _mirror_index(N_SECTORS, dev)
                T["rim_add"] = torch.where(flip.view(B, 1, 1), T["rim_add"][..., mi], T["rim_add"])
                T["rim_ratio"] = torch.where(flip.view(B, 1), T["rim_ratio"][..., mi], T["rim_ratio"])
        if "light" in self.stages:
            c_s, blotch_s = illum_coeffs(y, with_residual=True)
            if self.src_illum is not None:
                sh = float(p["shrink"])
                c_s = sh * c_s + (1 - sh) * self.src_illum.to(dev)
            dlt = T["illum"] - c_s
            dl = dlt.mean(dim=1)
            xx, yy = frame_coords(B, H, W, dev)
            Phi = _illum_basis(xx, yy)
            g_l = torch.einsum("bk,bkhw->bhw", dl[:, 1:], Phi[:, 1:]).unsqueeze(1)
            g_c = torch.einsum("bck,bkhw->bchw", dlt[..., 1:3] - dl[:, None, 1:3], Phi[:, 1:3]).clamp(max=0.0)
            amp = float(p["blotch_gain"]) * torch.sqrt((T["blotch"] ** 2 - blotch_s ** 2).clamp_min(0)).view(B, 1, 1, 1)
            Z = (amp * smooth_field(B, H, W, float(p["blotch_len"]), gen_dev, dev)).clamp(
                -float(p["blotch_max"]), float(p["blotch_max"]))
            lo, hi = p["light_lim"]
            G = g_l + g_c + Z
            # no more retina in shadow than the drawn photo had (+0.05): scale the
            # change back, image by image, until it fits
            cap = (T["shadow"] + 0.05).view(B)
            f = torch.ones(B, device=dev)
            for fk in (0.8, 0.6, 0.4, 0.2):
                sh = shadow_fraction((y * torch.exp(s * f.view(B, 1, 1, 1) * G).clamp(float(lo), float(hi))).clamp(0, 1),
                                     inside)
                f = torch.where(sh > cap, torch.full_like(f, fk), f)
            y = (y * torch.exp(s * f.view(B, 1, 1, 1) * G).clamp(float(lo), float(hi))).clamp(0, 1)
            if "rim" in self.stages:                      # the drawn photo's dark rim bands
                ratio, wr = self._rim_maps(T["rim_ratio"].unsqueeze(1), geo, B, H, W, dev)
                y = y * (1 + (ratio.clamp(0.2, 1.0) ** s - 1) * wr)
        m = erode(fov_mask(y) & inside, max(3, H // 64))
        if "colour" in self.stages:
            # tone-map the reflectance, keep the illumination: the map runs on the image
            # divided by its own low-order field (the bank's statistics are measured the
            # same way), and the field is multiplied back
            Fy = illum_field(illum_coeffs(y), H, W).clamp(0.2, 5.0)
            yf = y / Fy
            if p["colour_mode"] == "coupled":
                zone = _centre_zone(B, H, W, geo["disc_xy"] if geo is not None else None, dev)
                f = coupled_colour(yf, m, zone, T, tuple(float(k) for k in p["kappa"]), 1.0)
            elif p["colour_mode"] == "mk":
                mu_s, cov_s = colour_moments(yf, m)
                Tm = mk_map(cov_s, T["cov"])
                f = T["mu"][..., None, None] + torch.einsum("bij,bjhw->bihw", Tm, yf - mu_s[..., None, None])
            else:
                src = robust_colour(yf, m)
                k = (T["lscale"] / src["lscale"]).clamp(*[float(v) for v in p["colour_common"]]).unsqueeze(1)
                sp = float(p["colour_spread"])
                kc = torch.minimum(torch.maximum(T["scale"] / src["scale"], k / sp), k * sp)
                f = T["med"][..., None, None] + kc[..., None, None] * (yf - src["med"][..., None, None])
            y = ((yf + s * (f - yf)) * Fy).clamp(0, 1)
        if "bloom" in self.stages and geo is not None:
            xx, yy = frame_coords(B, H, W, dev)
            dist = torch.sqrt((xx - geo["disc_xy"][:, 0].view(B, 1, 1)) ** 2 + (yy - geo["disc_xy"][:, 1].view(B, 1, 1)) ** 2)
            r = geo["disc_r"].view(B, 1, 1)
            t = ((1.3 * r - dist) / (0.3 * r)).clamp(0, 1)                # full to 1.0 r, gone by 1.3 r
            region = (t * t * (3 - 2 * t)).unsqueeze(1)
            lum = y.mean(dim=1, keepdim=True)
            mf = m.float()
            bg = (lum * mf).sum(dim=(2, 3), keepdim=True) / mf.sum(dim=(2, 3), keepdim=True).clamp_min(1)
            peak = (lum * region).amax(dim=(2, 3), keepdim=True)
            rel = ((lum - bg) / (peak - bg).clamp_min(1e-3)).clamp(0, 1)
            w = (s * T["bloom"].clamp(0, 1) * geo["disc_conf"]).view(B, 1, 1, 1) * region * rel.sqrt()
            y = (y + w * (T["hcol"].view(B, 3, 1, 1) - y)).clamp(0, 1)
        if "detail" in self.stages:
            # the bank's detail is per unit background contrast: rescale to this image's
            lsc = robust_colour(y / illum_field(illum_coeffs(y), H, W).clamp(0.2, 5.0), m)["lscale"]
            target = T["detail"] * lsc ** 2
            if s != 1.0:
                cur = band_energy(y.mean(dim=1, keepdim=True), inside, me9).clamp_min(1e-12)
                target = torch.exp(cur.log() + s * (target.clamp_min(1e-12).log() - cur.log()))
            y = match_detail(y, target, inside, me=me9).clamp(0, 1)
        if "noise" in self.stages:
            # the phone ISP's chroma noise reduction: fine detail kept as luma
            have_c = fine_chroma(y, inside, me9) if p["fine_chroma"] else T["chroma"]
            keep = (T["chroma"] / have_c.clamp_min(1e-6)).clamp(0, 1)
            keep = 1 - s * (1 - keep)
            r = y - blur(y, 1.0)
            lum_r = r.mean(dim=1, keepdim=True)
            y = (y - (1 - keep).view(B, 1, 1, 1) * (r - lum_r)).clamp(0, 1)
            eps = correlated_noise((B, C, H, W), float(p["noise_corr"]), float(p["noise_rho"]), gen_dev, dev)
            ref = noise_level(eps.mean(dim=1, keepdim=True), inside, me=me9)
            have = noise_level(y.mean(dim=1, keepdim=True), inside, me=me9)
            amp = torch.nan_to_num(torch.sqrt((T["noise"] ** 2 - have ** 2).clamp_min(0)) / ref.clamp_min(1e-6))
            # no grain where the phone has none: the clipped disc (grain there would read as
            # fine new vessels) and the black surround
            top = y.amax(dim=1, keepdim=True)
            taper = ((top - 8 / 255) / (8 / 255)).clamp(0, 1) * ((250 / 255 - top) / (15 / 255)).clamp(0, 1)
            y = y + s * eps * amp.view(B, 1, 1, 1) * taper
        if "colour" in self.stages:
            # the phone's demosaic / colour matrix acts after its sensor noise: R and B
            # fine structure (detail band only, grain included) pulled toward G's, up to
            # the drawn photo's correlation; G and the low frequencies untouched
            y = mix_structure(y.clamp(0, 1), m, T["xcorr"], s, me9).clamp(0, 1)
        if "rim" in self.stages:                          # the drawn photo's rim haze
            add, wr = self._rim_maps(T["rim_add"], geo, B, H, W, dev)
            y = y + s * add * wr
        return (y * A).clamp(0, 1)

    @staticmethod
    def _rim_maps(v: torch.Tensor, geo, B: int, H: int, W: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
        """A per-sector rim profile [B,C,S] spread over the frame (circular linear
        interpolation in angle) and its radial weight (0 inside 0.8 of the aperture
        radius, 1 at 0.97)."""
        xx, yy = frame_coords(B, H, W, dev)
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        if geo is not None:
            rr = rr / geo["radius"].view(B, 1, 1)
        pos = (torch.atan2(yy, xx) + math.pi) / (2 * math.pi) * N_SECTORS - 0.5
        i0 = torch.floor(pos).long() % N_SECTORS
        i1 = (i0 + 1) % N_SECTORS
        f = (pos - torch.floor(pos)).unsqueeze(1)
        C = v.shape[1]
        pick = lambda i: torch.gather(v, -1, i.view(B, 1, -1).expand(B, C, -1)).view(B, C, H, W)
        t = ((rr - 0.80) / 0.17).clamp(0, 1).unsqueeze(1)
        return pick(i0) * (1 - f) + pick(i1) * f, t * t * (3 - 2 * t)

    # ---- Stylizer interface ---------------------------------------------- #
    @torch.no_grad()
    def __call__(self, x01: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        gen = gen or torch.Generator().manual_seed(0)
        return self.render(x01, self.sample(x01.shape[0], gen))

    @torch.no_grad()
    def geometry(self, x01: torch.Tensor) -> torch.Tensor:
        """The last call's warp and aperture on ``x01`` (same batch), so a structural
        check compares like with like; without one, the profile's mean warp."""
        B = x01.shape[0]
        H = W = int(self.out_size or x01.shape[-1])
        L = self._last
        if L is None or L["zoom"].shape[0] != B or "field" not in self.stages:
            prm = self.sample(B, torch.Generator().manual_seed(0))
            prm["zoom"] = torch.full((B,), float(self.p["zoom"]))
            prm["rot"], prm["jit"] = torch.zeros(B), torch.zeros(B, 2)
            T = {k: v[prm["idx"]].to(x01.device) for k, v in self._bank.items()}
            L = self._field(x01.float(), prm, T)
        L = {k: v.to(x01.device) for k, v in L.items()}
        y = warp(x01.float(), L["zoom"], L["a_f"], L["b_f"], L["offset"], L["rot"], (H, W))
        return (y * aperture(L["radius"], float(self.p["rim_width"]), H, W)).to(x01.dtype)

    # ---- training -------------------------------------------------------- #
    @torch.no_grad()
    def augment(self, x01: torch.Tensor, prob: float, gen: torch.Generator,
                out_size: Optional[int] = None) -> torch.Tensor:
        """Training: each image re-rendered with probability ``prob`` (draws from
        ``gen``), the rest only resized to ``out_size`` (antialiased bilinear,
        torchvision's Resize on tensors), so a supersampled batch comes out at
        the model's size either way."""
        B, C, H, W = x01.shape
        S = int(out_size or H)
        out = x01 if S == H else F.interpolate(x01, size=(S, S), mode="bilinear", antialias=True,
                                               align_corners=False)
        pick = torch.rand(B, generator=gen) < prob
        if not bool(pick.any()):
            return out
        idx = pick.nonzero()[:, 0]
        out = out.clone()
        out[idx.to(x01.device)] = self.render(x01[idx.to(x01.device)], self.sample(len(idx), gen),
                                              out_size=S).to(out.dtype)
        return out


# --------------------------------------------------------------------------- #
# Realism report: per-statistic camera separability
# --------------------------------------------------------------------------- #
@torch.no_grad()
def camera_stats(x01: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-photo statistics a camera can be told apart by (model view)."""
    B, C, H, W = x01.shape
    m = fov_mask(x01)
    me = erode(m, max(3, H // 64))
    v, k = _strided(x01, me, 3)
    lum = v.mean(dim=1, keepdim=True)
    med = _masked_quantiles(v, k.expand_as(v), [0.5])[0]
    q = _masked_quantiles(lum, k, [0.25, 0.75, 0.99])[..., 0]
    ssum = v.sum(dim=1, keepdim=True).clamp_min(1e-6)
    chroma = _masked_quantiles(v / ssum, k.expand_as(v), [0.5])[0]
    p99 = _masked_quantiles(v, k.expand_as(v), [0.99])[0]
    clip = ((v.amax(dim=1, keepdim=True) >= 0.92) & k).float().sum(dim=(1, 2)) / k.float().sum(dim=(1, 2)).clamp_min(1)
    g = x01.mean(dim=1, keepdim=True)
    lb = blur(g, H / 32)
    xx, yy = frame_coords(B, H, W, x01.device)
    rr = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(1)
    mf = me.float()
    ring = lambda a, b: (lb * mf * (rr >= a) * (rr < b)).sum(dim=(1, 2, 3)) / (mf * (rr >= a) * (rr < b)).sum(dim=(1, 2, 3)).clamp_min(1)
    shape = fov_shape(x01)
    c, blotch = illum_coeffs(x01, with_residual=True)
    tilt = c.mean(dim=1)[:, 3:5].norm(dim=1)
    ac1, rg = noise_structure(x01)
    xc = structure_corr(x01, m)
    return {"median_R": med[:, 0], "median_G": med[:, 1], "median_B": med[:, 2],
            "contrast_IQR": q[1] - q[0], "highlight_p99": q[2], "chroma_red": chroma[:, 0],
            "chroma_blue": chroma[:, 2], "p99_B": p99[:, 2], "clipped": clip,
            "detail": band_energy(g, m), "noise": noise_level(g, m), "rim_over_centre": ring(0.75, 0.88) / ring(0.0, 0.3).clamp_min(1e-4),
            "blotch": blotch, "tilt": tilt, "shadow": shadow_fraction(x01, me), "resid_ac1": ac1, "resid_rg": rg,
            "struct_rg": xc[:, 0], "struct_bg": xc[:, 1],
            "fov_area": shape["area"], "fov_aspect": shape["aspect"]}


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie): 0.5 = indistinguishable."""
    a, b = np.asarray(neg, float), np.asarray(pos, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((b[None, :] > a[:, None]).mean() + 0.5 * (b[None, :] == a[:, None]).mean())


def realism_table(tab: Dict[str, np.ndarray], sim: Dict[str, np.ndarray], real: Dict[str, np.ndarray]):
    """Per statistic: medians, and AUROC of tabletop vs real handheld and of
    simulated vs real handheld (0.5 = the cameras cannot be told apart by it)."""
    import pandas as pd
    rows = []
    for k in real:
        rows.append({"statistic": k, "tabletop": float(np.nanmedian(tab[k])), "simulated": float(np.nanmedian(sim[k])),
                     "handheld": float(np.nanmedian(real[k])), "auc_tabletop": auroc(real[k], tab[k]),
                     "auc_simulated": auroc(real[k], sim[k])})
    df = pd.DataFrame(rows)
    df["gap_tabletop"] = (df.auc_tabletop - 0.5).abs() * 2
    df["gap_simulated"] = (df.auc_simulated - 0.5).abs() * 2
    return df


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _cat(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else {}


@torch.no_grad()
def noise_structure(x01: torch.Tensor, flat_q: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B] lag-1 autocorrelation of the grey residual and [B] R-G residual
    correlation, over the flattest ``flat_q`` of the FOV."""
    r = x01 - blur(x01, 1.0)
    g = x01.mean(dim=1, keepdim=True)
    gy, gx = torch.gradient(blur(g, 2.0), dim=(2, 3))
    e = blur(gx ** 2 + gy ** 2, 2.0)
    me = erode(fov_mask(x01), 9)
    thr = _masked_quantiles(e.view(e.shape[0], 1, -1), me.view(me.shape[0], 1, -1), [flat_q])[0]
    flat = (me & (e <= thr.view(-1, 1, 1, 1))).float()
    rg = r.mean(dim=1, keepdim=True)
    f2 = flat[..., :, 1:] * flat[..., :, :-1]
    corr = lambda a, b, w: (((a - (a * w).sum((1, 2, 3), keepdim=True) / w.sum((1, 2, 3), keepdim=True).clamp_min(1))
                             * (b - (b * w).sum((1, 2, 3), keepdim=True) / w.sum((1, 2, 3), keepdim=True).clamp_min(1)) * w)
                            .sum((1, 2, 3)) / (((a ** 2) * w).sum((1, 2, 3)).sqrt() * ((b ** 2) * w).sum((1, 2, 3)).sqrt()).clamp_min(1e-12))
    ac1 = corr(rg[..., :, 1:], rg[..., :, :-1], f2)
    rho = corr(r[:, 0:1], r[:, 1:2], flat)
    return ac1, rho


def _sigma_for_ac1(ac1: float, size: int = 192) -> float:
    """Spatial correlation (Gaussian sigma, px) of unit noise whose residual has
    lag-1 autocorrelation ``ac1`` under :func:`noise_structure`'s estimator."""
    grid = [0.0, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
    g = torch.Generator().manual_seed(0)
    vals = []
    for sc in grid:
        n = correlated_noise((2, 3, size, size), sc, 1.0, g, "cpu") * 0.02 + 0.5
        vals.append(float(noise_structure(n)[0].mean()))
    return float(np.interp(ac1, vals, grid))


@torch.no_grad()
def fit_profile(target: Iterable[torch.Tensor], source: Iterable[torch.Tensor], image_size: int,
                params: Optional[Dict[str, object]] = None, max_rows: int = 0,
                log: Callable[[str], None] = print) -> Tuple[PhoneProfile, Dict[str, torch.Tensor]]:
    """The handheld bank from ``target`` batches and the global parameters from
    both cameras ([B,3,S,S] in [0,1], model view at ``image_size``). Returns the
    profile (without a report) and the source description (for the report)."""
    rows, ac1s, rhos, n = [], [], [], 0
    for x in target:
        if x.shape[-1] != image_size:
            raise ValueError(f"target batch at {x.shape[-1]} px, profile at {image_size}")
        rows.append({k: v.float().cpu() for k, v in describe(x).items()})
        a1, rh = noise_structure(x)
        ac1s.append(a1.cpu()); rhos.append(rh.cpu())
        n += x.shape[0]
        if max_rows and n >= max_rows:
            break
    bank = _cat(rows)
    if not bank:
        raise ValueError("no handheld photos to fit on")
    ok = torch.ones(len(bank["med"]), dtype=torch.bool)
    for k in ("med", "scale", "lscale", "mu", "cov", "ctr", "band_rho", "fine_amp", "illum", "blotch", "detail",
              "noise", "chroma", "shadow", "xcorr", "fov_r", "hcol", "rim_add", "rim_ratio"):
        ok &= torch.isfinite(bank[k].view(len(ok), -1)).all(dim=1)
    if int(ok.sum()) == 0:
        raise ValueError("every handheld photo failed a measurement")
    if int((~ok).sum()):
        log(f"[fit] dropped {int((~ok).sum())} handheld photos with an undefined measurement")
    bank = {k: v[ok] for k, v in bank.items()}
    src = _cat([{k: v.float().cpu() for k, v in describe(x).items()} for x in source])
    p = {**DEFAULT_PARAMS, **(params or {})}
    med = lambda t: float(t[torch.isfinite(t)].median()) if torch.isfinite(t).any() else float("nan")
    if src:
        rel = src["aspect_ok"] > 0.5
        if bool(rel.any()):
            p["source_aspect"] = float(src["aspect"][rel].median())
        p["src_illum"] = src["illum"][torch.isfinite(src["illum"].view(len(rel), -1)).all(1)].mean(0).tolist()
        dfd_s, dfd_t = med(src["dfd"]), med(_cat(rows)["dfd"]) if rows else float("nan")
        r_s, r_t = med(src["disc_r_fov"]), med(_cat(rows)["disc_r_fov"])
        n_s, n_t = int(torch.isfinite(src["dfd"]).sum()), int(torch.isfinite(_cat(rows)["dfd"]).sum())
        by_disc = r_t / r_s if np.isfinite(r_s) and np.isfinite(r_t) and r_s > 0 else float("nan")
        by_dfd = dfd_t / dfd_s if np.isfinite(dfd_s) and np.isfinite(dfd_t) and dfd_s > 0 else float("nan")
        if np.isfinite(by_dfd) and min(n_s, n_t) >= 5:
            p["zoom"] = float(np.clip(by_dfd, 1.0, 1.6))
        elif np.isfinite(by_disc):
            p["zoom"] = float(np.clip(by_disc, 1.0, 1.6))
        log(f"[fit] angular scale: disc-fovea distance handheld {dfd_t:.3f} / tabletop {dfd_s:.3f} FOV radii "
            f"({n_t} / {n_s} photos) -> {by_dfd:.3f}; disc radius {r_t:.3f} / {r_s:.3f} -> {by_disc:.3f} "
            f"(an upper bound: phone discs bloom); zoom {p['zoom']:.3f}; tabletop aspect {p['source_aspect']:.3f}")
    dd = med(bank["disc_dist"])
    if np.isfinite(dd):
        p["disc_dist"] = dd
    ac1, rho = med(torch.cat(ac1s)), med(torch.cat(rhos))
    if np.isfinite(ac1):
        p["noise_corr"] = float(np.clip(_sigma_for_ac1(ac1), 0.4, 0.8))
    log(f"[fit] handheld noise: residual lag-1 autocorrelation {ac1:.2f} -> correlation sigma "
        f"{p['noise_corr']:.2f} px; residual R-G correlation {rho:.2f} at this size (texture "
        f"included; the noise itself is set to rho {p['noise_rho']:.2f})")
    prof = PhoneProfile(image_size=image_size, bank={k: v.numpy() for k, v in bank.items() if k in BANK_KEYS},
                        params=p, report={"target_resid_ac1": ac1, "target_resid_rg": rho,
                                          "kept": ok.nonzero()[:, 0].tolist()})
    return prof, src


@torch.no_grad()
def calibrate_grain(profile: PhoneProfile, src_hi: torch.Tensor, out_size: int, seed: int = 0,
                    sigmas: Sequence[float] = (0.0, 0.3, 0.45, 0.6, 0.8),
                    rhos: Sequence[float] = (0.6, 0.75, 0.85, 0.94, 0.99),
                    log: Callable[[str], None] = print) -> Tuple[float, float]:
    """The grain's spatial correlation, then its cross-channel correlation, by
    simulation: render ``src_hi`` at each candidate and keep the value whose output
    residual statistic (lag-1 autocorrelation; R-G correlation) matches the handheld
    photos' under the same estimator, so the texture underneath biases both alike."""
    t_ac1 = float(profile.report.get("target_resid_ac1", float("nan")))
    t_rg = float(profile.report.get("target_resid_rg", float("nan")))
    sc, rho = float(profile.params["noise_corr"]), float(profile.params["noise_rho"])

    def sweep(key, grid, stat):
        got = []
        for v in grid:
            prof = PhoneProfile(profile.image_size, profile.bank, {**profile.params, "noise_corr": sc,
                                                                    "noise_rho": rho, key: v})
            y = PhoneSimulator(prof, out_size=out_size)(src_hi, torch.Generator().manual_seed(seed))
            got.append(float(noise_structure(y)[stat].median()))
        return got

    if np.isfinite(t_ac1):
        got = sweep("noise_corr", sigmas, 0)
        order = np.argsort(got)
        sc = float(np.interp(t_ac1, np.array(got)[order], np.array(sigmas)[order]))
        log(f"[fit] grain size: residual lag-1 autocorrelation {dict(zip(sigmas, np.round(got, 3)))} vs "
            f"handheld {t_ac1:.3f} -> sigma {sc:.2f} px")
    if np.isfinite(t_rg):
        got = sweep("noise_rho", rhos, 1)
        order = np.argsort(got)
        rho = float(np.interp(t_rg, np.array(got)[order], np.array(rhos)[order]))
        log(f"[fit] grain colour: residual R-G correlation {dict(zip(rhos, np.round(got, 3)))} vs "
            f"handheld {t_rg:.3f} -> rho {rho:.2f}")
    return sc, rho


# --------------------------------------------------------------------------- #
# Data: a dataset's own split, or a folder of photos (same decode / crop / resize)
# --------------------------------------------------------------------------- #
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def folder_dataset(path: str, size: int):
    """Every photo in ``path`` through the trainers' loader (FOV crop, square resize)."""
    import pandas as pd
    from dataset import MBRSETDataset
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
    if not files:
        raise ValueError(f"no photos in {path}")
    return MBRSETDataset(csv=pd.DataFrame({"file": files, "unlabelled": 0}), images_dir=path, task=None,
                         label_col="unlabelled", split="val", image_size=size)


def split_dataset(root: str, dataset: str, size: int, split_seed: int = 42, part: str = "train",
                  camera: Optional[str] = None, select: str = "all"):
    """One part of the trainer's own patient split (``multitask_split`` on referable
    DR after the ``--source-camera`` filter), as ``train_retinareach.py`` builds it.
    ``select='icdr0'`` keeps grade-0 photos only (reads the grade)."""
    import pandas as pd
    from brset_dataset import filter_camera, load_any
    from capability_gate import multitask_split
    from metadata_model import load_table
    from train_retinareach import MultiTargetDataset
    df, _ = load_table(root, dataset)
    if camera:
        df = filter_camera(df, camera.strip())
    part_df = multitask_split(df, "dr_referable", seed=split_seed)[part]
    if select == "icdr0":
        part_df = part_df[pd.to_numeric(part_df["final_icdr"], errors="coerce") == 0]
    elif select != "all":
        raise ValueError(f"select must be all or icdr0, not {select!r}")
    return MultiTargetDataset(part_df.reset_index(drop=True), load_any(root, dataset)["images_dir"],
                              ["dr_referable"], "val", size)


def _source(a, size: int, part: str):
    if a.source_dir:
        return folder_dataset(a.source_dir, size)
    return split_dataset(a.source_root, a.source_dataset, size, a.split_seed, part, a.source_camera)


def _target(a, size: int, part: str):
    if a.target_dir:
        return folder_dataset(a.target_dir, size)
    return split_dataset(a.target_root, a.target_dataset, size, a.split_seed, part, None,
                         getattr(a, "select", "all"))


def _subset(ds, n: int, seed: int):
    from torch.utils.data import Subset
    if not n or n >= len(ds):
        return ds
    return Subset(ds, sorted(np.random.default_rng(seed).choice(len(ds), n, replace=False).tolist()))


def batches(ds, batch: int, workers: int, dev) -> Iterable[torch.Tensor]:
    from torch.utils.data import DataLoader
    for b in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        yield denorm(b["image"].to(dev))


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    from train_mbrset import pick_device
    return pick_device()


def save_rows(path: str, rows: Sequence[torch.Tensor], labels: Sequence[str] = ()) -> None:
    """A JPEG of image rows (each [n,3,S,S] in [0,1]), top to bottom."""
    from PIL import Image, ImageDraw
    n = max(r.shape[0] for r in rows)
    S = rows[0].shape[-1]
    canvas = torch.zeros(3, S * len(rows), S * n)
    for i, r in enumerate(rows):
        for j in range(r.shape[0]):
            canvas[:, i * S:(i + 1) * S, j * S:(j + 1) * S] = r[j].float().cpu().clamp(0, 1)
    img = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i, lab in enumerate(labels):
        d.text((6, i * S + 4), lab, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path, quality=92)


@torch.no_grad()
def cross_fitted_report(profile: PhoneProfile, src_hi: torch.Tensor, real: torch.Tensor, real_rows: np.ndarray,
                        seed: int, out_size: int, strength: float = 1.0) -> Tuple[object, torch.Tensor]:
    """Two folds: the source photos rendered from half the bank rows, compared
    with the real handheld photos behind the OTHER half, so no simulated photo is
    scored against the photo it drew its statistics from. ``real_rows`` maps each
    real photo to its bank row (-1 = not in the bank)."""
    N = len(profile)
    half = np.arange(N) % 2
    sims, reals, tabs = [], [], []
    gen = torch.Generator().manual_seed(seed)
    for f in (0, 1):
        sub = PhoneProfile(profile.image_size, {k: v[half == f] for k, v in profile.bank.items()}, profile.params)
        sim = PhoneSimulator(sub, strength=strength, out_size=out_size)
        part = slice(f, None, 2)
        y = sim(src_hi[part], gen)
        sims.append(camera_stats(y))
        sel = torch.from_numpy((real_rows >= 0) & (half[np.clip(real_rows, 0, N - 1)] != f))
        reals.append(camera_stats(real[sel]))
    sim_y = PhoneSimulator(profile, strength=strength, out_size=out_size)(src_hi, torch.Generator().manual_seed(seed))
    cat = lambda L: {k: torch.cat([d[k] for d in L]).numpy() for k in L[0]}
    tab = {k: v.numpy() for k, v in camera_stats(F.interpolate(src_hi, size=(out_size, out_size), mode="bilinear",
                                                                antialias=True, align_corners=False)).items()}
    return realism_table(tab, cat(sims), cat(reals)), sim_y


def _fmt_table(df) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def cmd_fit(a) -> int:
    dev = _pick_device(a.device)
    S, ss = a.image_size, a.supersample
    S_hi = int(round(S * ss))
    tgt = _subset(_target(a, S, "train"), a.max_target, a.seed)
    src = _subset(_source(a, S, "train"), a.max_source, a.seed + 1)
    print(f"[fit] handheld photos {len(tgt)} ({a.target_dir or a.target_root}, train split, select {a.select}); "
          f"tabletop photos {len(src)} ({a.source_dir or a.source_root}"
          f"{', camera ' + repr(a.source_camera) if a.source_camera else ''}); {S} px on {dev}")
    prof, src_desc = fit_profile(batches(tgt, a.batch_size, a.num_workers, dev),
                                 batches(src, a.batch_size, a.num_workers, dev), S,
                                 params={"supersample": ss})
    prof.source = {"target": a.target_dir or a.target_root, "target_dataset": None if a.target_dir else a.target_dataset,
                   "source": a.source_dir or a.source_root, "source_dataset": None if a.source_dir else a.source_dataset,
                   "source_camera": a.source_camera, "split_seed": a.split_seed, "split": "train",
                   "select": a.select, "n_target": len(prof), "n_source": len(src), "image_size": S}
    out_dir = os.path.dirname(os.path.abspath(a.out))
    report_lines = [f"phone_sim profile {a.out}: {len(prof)} handheld photos in the bank, {S} px"]
    if a.validate:
        n = min(a.validate, len(src))
        src_hi_ds = _subset(_source(a, S_hi, "train"), n, a.seed + 2)
        real_ds = _subset(_target(a, S, "train"), min(a.validate, len(tgt)), a.seed + 3)
        X_hi = torch.cat(list(batches(src_hi_ds, a.batch_size, a.num_workers, dev)))
        R = torch.cat(list(batches(real_ds, a.batch_size, a.num_workers, dev)))
        prof.params["noise_corr"], prof.params["noise_rho"] = calibrate_grain(prof, X_hi[:32], S, a.seed)
        # the real photos behind the bank rows: the bank is fitted on tgt in order, so match by file
        tgt_files = [tgt.dataset.files[i] for i in tgt.indices] if hasattr(tgt, "indices") else list(tgt.files)
        real_files = [real_ds.dataset.files[i] for i in real_ds.indices] if hasattr(real_ds, "indices") else list(real_ds.files)
        # bank row r came from target photo kept[r] (photos with an undefined measurement were dropped)
        pos = {tgt_files[k]: r for r, k in enumerate(prof.report["kept"])}
        rows = np.array([pos.get(f, -1) for f in real_files])
        df, Y = cross_fitted_report(prof, X_hi, R, rows, a.seed, S)
        prof.report.update({"realism": df.to_dict(orient="list"), "n_validate": int(n)})
        report_lines += ["", "REALISM (cross-fitted: a simulated photo is never scored against the photo its "
                         "statistics came from)", "auc_* = AUROC against real handheld photos, 0.5 = cannot tell "
                         "them apart; gap = |auc - 0.5| x 2", _fmt_table(df),
                         f"mean gap: tabletop {df.gap_tabletop.mean():.3f} -> simulated {df.gap_simulated.mean():.3f}"]
        k = min(8, len(X_hi), len(R))
        save_rows(os.path.join(out_dir, "fit_grid.jpg"),
                  [F.interpolate(X_hi[:k], size=(S, S), mode="bilinear", antialias=True, align_corners=False),
                   Y[:k], R[:k]], ["tabletop", "simulated", "real handheld"])
        df.to_csv(os.path.join(out_dir, "fit_report.csv"), index=False)
    report_lines.insert(1, "parameters: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                        for k, v in prof.params.items() if k != "src_illum"}))
    prof.save(a.out)
    txt = "\n".join(report_lines)
    print(txt)
    with open(os.path.join(out_dir, "fit_report.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"[fit] wrote {a.out}, fit_report.txt" + (", fit_report.csv, fit_grid.jpg" if a.validate else ""))
    return 0


def cmd_render(a) -> int:
    dev = _pick_device(a.device)
    prof = PhoneProfile.load(a.profile)
    S = prof.image_size
    S_hi = int(round(S * float(prof.params.get("supersample", 1.5))))
    src = _subset(_source(a, S_hi, "test"), a.n, a.seed)
    X = torch.cat(list(batches(src, a.batch_size, a.num_workers, dev)))
    sim = PhoneSimulator(prof, strength=a.strength, out_size=S)
    Y = sim(X, torch.Generator().manual_seed(a.seed))
    X_lo = F.interpolate(X, size=(S, S), mode="bilinear", antialias=True, align_corners=False)
    rows, labels = [X_lo, Y], ["tabletop", "simulated"]
    if a.target_dir or a.target_root:
        R = torch.cat(list(batches(_subset(_target(a, S, "test"), a.n, a.seed), a.batch_size, a.num_workers, dev)))
        rows.append(R); labels.append("real handheld")
    os.makedirs(a.out, exist_ok=True)
    save_rows(os.path.join(a.out, "grid.jpg"), rows, labels)
    # one photo per row, the chain stage by stage
    k = min(a.stage_rows, len(X))
    strip, names = [X_lo[:k]], ["tabletop"]
    for i in range(1, len(STAGES) + 1):
        st = PhoneSimulator(prof, strength=a.strength, stages=STAGES[:i], out_size=S)
        strip.append(st(X[:k], torch.Generator().manual_seed(a.seed)))
        names.append("+" + STAGES[i - 1])
    tiles = torch.stack(strip, dim=1)                                  # [k, stages, 3, S, S]
    save_rows(os.path.join(a.out, "stages.jpg"), [tiles[i] for i in range(k)])
    with open(os.path.join(a.out, "stages.txt"), "w") as f:
        f.write("columns, left to right: " + ", ".join(names) + "\n")
    print(f"[render] {len(X)} photos -> {a.out}/grid.jpg (rows: {labels}); {a.out}/stages.jpg "
          f"(columns: {names})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tabletop -> handheld fundus photo simulator.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data(q, need_target: bool):
        q.add_argument("--source-root", default=None, help="tabletop dataset root (BRSET)")
        q.add_argument("--source-dataset", default="brset", choices=["brset", "mbrset"])
        q.add_argument("--source-camera", default=None, help="one BRSET camera, e.g. 'Canon CR'")
        q.add_argument("--source-dir", default=None, help="or a folder of tabletop photos")
        q.add_argument("--target-root", default=None, help="handheld dataset root (mBRSET)")
        q.add_argument("--target-dataset", default="mbrset", choices=["brset", "mbrset"])
        q.add_argument("--target-dir", default=None, help="or a folder of handheld photos")
        q.add_argument("--split-seed", type=int, default=42)
        q.add_argument("--batch-size", type=int, default=16)
        q.add_argument("--num-workers", type=int, default=4)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    q = sub.add_parser("fit", help="measure the handheld camera into a profile")
    data(q, True)
    q.add_argument("--image-size", type=int, default=384, help="the model's image size")
    q.add_argument("--supersample", type=float, default=1.5,
                   help="training renders from a view this much larger than the model's")
    q.add_argument("--select", default="all", choices=["all", "icdr0"],
                   help="handheld photos: all of the train split (no label read) or grade 0 only")
    q.add_argument("--max-target", type=int, default=2000, help="bank rows at most (0 = all)")
    q.add_argument("--max-source", type=int, default=600)
    q.add_argument("--validate", type=int, default=120,
                   help="tabletop photos rendered for the cross-fitted realism report (0 = skip)")
    q.add_argument("--out", default="ck_phone_sim/profile.json")
    q.set_defaults(fn=cmd_fit)

    q = sub.add_parser("render", help="tabletop -> simulated (-> real handheld) grid and a per-stage strip")
    data(q, False)
    q.add_argument("--profile", required=True)
    q.add_argument("--n", type=int, default=8)
    q.add_argument("--stage-rows", type=int, default=4)
    q.add_argument("--strength", type=float, default=1.0)
    q.add_argument("--out", default="exp_phone_sim/render")
    q.set_defaults(fn=cmd_render)

    a = p.parse_args(argv)
    if not (a.source_dir or a.source_root):
        p.error("give --source-root or --source-dir")
    if a.cmd == "fit" and not (a.target_dir or a.target_root):
        p.error("fit needs --target-root or --target-dir (the handheld photos)")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
