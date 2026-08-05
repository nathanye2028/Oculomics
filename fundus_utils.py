"""
fundus_utils.py
===============
Shared utilities for the fundus pipeline:

* **Reproducibility** — ``seed_everything``, a DataLoader ``seed_worker`` hook,
  and ``make_rng`` for per-sample augmentation RNG that is *both* reproducible
  and actually diverse across workers (fixes the "all workers share one seed"
  bug).
* **Fundus preprocessing** — ``fov_bbox`` / ``crop_to_fov`` to strip the black
  border, and ``circular_fov_mask``.
* **Imbalance-aware losses** — ``tversky_loss`` / ``focal_tversky_loss`` for
  tiny-lesion segmentation, where plain BCE/Dice collapses to background.
* **High-resolution patching** — ``random_patch`` (lesion-biased crops at native
  resolution) and ``tiled_predict`` (stitch full-image inference from tiles) so
  microaneurysms aren't destroyed by downscaling.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG, and (by default) force deterministic GPU kernels.

    ``deterministic`` matters more than it looks. Seeding Python/NumPy/PyTorch
    is NOT sufficient on CUDA: cuDNN autotunes and picks non-deterministic
    convolution algorithms, so two runs with the SAME seed diverge. We hit this
    for real — IDRiD runs with identical seeds produced test Dice 0.449 vs
    0.328 across two experiments, which made "paired by seed" meaningless and
    turned a 3-seed +0.084 into an 8-seed -0.010.

    Costs roughly 10-20% throughput; worth it for results you can defend.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN: fixed algorithm choice, no autotuning.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS reductions (must precede CUDA init).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:    # warn_only: a few ops have no deterministic kernel; don't crash.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/random seeds from torch's
    per-worker base seed so each worker is seeded differently yet deterministically."""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base)
    random.seed(base)


def make_rng(base_seed: int, index: int) -> np.random.Generator:
    """Per-call RNG keyed on (worker base seed, dataset base seed, sample index).

    In a worker process ``torch.initial_seed()`` differs per worker *and* per
    epoch, so augmentation is diverse across workers and epochs while staying
    reproducible for a fixed global seed. In the main process it falls back to
    the dataset's base seed + index.
    """
    info = torch.utils.data.get_worker_info()
    worker_seed = int(torch.initial_seed()) if info is not None else int(base_seed)
    ss = np.random.SeedSequence([worker_seed % (2 ** 32), int(base_seed), int(index)])
    return np.random.default_rng(ss)


# --------------------------------------------------------------------------- #
# Fundus preprocessing
# --------------------------------------------------------------------------- #
def fov_bbox(rgb: np.ndarray, tol: int = 12) -> Tuple[int, int, int, int]:
    """Bounding box (r0, r1, c0, c1) of the non-black field of view."""
    gray = rgb.max(axis=2) if rgb.ndim == 3 else rgb
    rows = np.where(gray.max(axis=1) > tol)[0]
    cols = np.where(gray.max(axis=0) > tol)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, gray.shape[0], 0, gray.shape[1]
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def crop_to_fov(rgb: np.ndarray, tol: int = 12) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    r0, r1, c0, c1 = fov_bbox(rgb, tol)
    return rgb[r0:r1, c0:c1], (r0, r1, c0, c1)


def circular_fov_mask(h: int, w: int) -> np.ndarray:
    """Boolean circular mask inscribed in an h x w frame (retinal disc proxy)."""
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = min(cy, cx)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


# --------------------------------------------------------------------------- #
# Imbalance-aware segmentation losses
# --------------------------------------------------------------------------- #
def _tversky_index(logits, target, alpha, beta, smooth, valid=None):
    """Per-channel Tversky index, optionally masking unannotated channels.

    ``valid`` is [B, C] with 1 = this lesion is annotated for that sample.
    Samples whose channel is unannotated contribute nothing (their pixels are
    excluded from tp/fp/fn), so "not labelled" is never treated as "negative".
    Returns (index[C], channel_has_data[C]).
    """
    probs = torch.sigmoid(logits)
    if valid is not None:
        w = valid.to(probs.dtype)[:, :, None, None]      # [B,C,1,1]
        probs = probs * w
        target = target * w
        has = valid.sum(0) > 0                            # [C]
    else:
        has = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
    dims = (0, 2, 3)
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth), has


def _masked_mean(per_channel, has):
    """Mean over channels that actually had annotated data this batch."""
    if has.all():
        return per_channel.mean()
    hasf = has.to(per_channel.dtype)
    return (per_channel * hasf).sum() / hasf.sum().clamp(min=1.0)


def tversky_loss(logits, target, alpha=0.7, beta=0.3, smooth=1.0, valid=None):
    """Tversky loss. alpha weights false negatives (>beta -> recall-favouring,
    good for tiny lesions). Per-channel, averaged over annotated channels."""
    t, has = _tversky_index(logits, target, alpha, beta, smooth, valid)
    return _masked_mean(1.0 - t, has)


def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0, valid=None):
    """Focal Tversky: focuses learning on hard/rare structures (gamma<1).

    ``valid`` [B, C] lets datasets with partial lesion annotation (e.g. e-ophtha
    labels exudates only) contribute without their unlabelled channels being
    scored as all-negative.
    """
    t, has = _tversky_index(logits, target, alpha, beta, smooth, valid)
    return _masked_mean((1.0 - t).clamp(min=1e-6) ** gamma, has)


# --------------------------------------------------------------------------- #
# High-resolution patching
# --------------------------------------------------------------------------- #
def random_patch(
    img: np.ndarray,
    masks: np.ndarray,
    patch: int,
    rng: np.random.Generator,
    fg_bias: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random ``patch x patch`` crop at NATIVE resolution (no downscale).

    With prob ``fg_bias`` the crop is centred on a random lesion pixel so tiny
    lesions aren't perpetually cropped out; otherwise it's uniform random.
    ``img`` is [H,W,3], ``masks`` is [C,H,W].
    """
    H, W = img.shape[:2]
    ph = min(patch, H)
    pw = min(patch, W)
    fg = masks.any(axis=0)
    if fg_bias > 0 and fg.sum() > 0 and rng.random() < fg_bias:
        ys, xs = np.where(fg)
        k = int(rng.integers(len(ys)))
        top = int(np.clip(ys[k] - ph // 2, 0, H - ph))
        left = int(np.clip(xs[k] - pw // 2, 0, W - pw))
    else:
        top = int(rng.integers(0, H - ph + 1))
        left = int(rng.integers(0, W - pw + 1))
    return img[top:top + ph, left:left + pw], masks[:, top:top + ph, left:left + pw]


def _tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def tiled_predict(
    model,
    image: torch.Tensor,
    tile: int,
    overlap: int,
    device,
    tile_batch: int = 8,
    fg_map: Optional[torch.Tensor] = None,
    autocast: bool = True,
) -> torch.Tensor:
    """Full-resolution inference by stitching tiles. Built for scale.

    ``image`` is [3,H,W] (single image, native resolution). Returns sigmoid
    probabilities [C,H,W] on ``device``. Preserves small lesions a global
    downscale would erase.

    Efficiency levers (most-to-least impactful on this task):
      * **FOV tile skipping** — if ``fg_map`` ([H,W] bool) is given, tiles with
        no field-of-view pixel are skipped entirely (left as background). A
        fundus image is a circle in a black frame, so a large fraction of
        corner tiles cost nothing. This is the dominant, hardware-independent
        win (the work is compute-bound, so fewer forwards == less time).
      * **batched forwards** (``tile_batch``) — helps a lot on CUDA; modest on
        MPS. Kept small by default to avoid the MPS large-batch memory cliff.
      * on-device accumulation + ``inference_mode`` + optional CUDA autocast.
    """
    model.eval()
    _, H, W = image.shape
    tile = min(tile, H, W)                       # keep all tiles equal-sized
    ys = _tile_starts(H, tile, overlap)
    xs = _tile_starts(W, tile, overlap)
    coords = [(y, x) for y in ys for x in xs]
    if fg_map is not None:
        coords = [(y, x) for (y, x) in coords if bool(fg_map[y:y + tile, x:x + tile].any())]

    image = image.to(device, non_blocking=True)
    n_lesions = None
    prob_sum = None
    weight = torch.zeros(1, H, W, device=device)
    use_ac = autocast and device.type == "cuda"

    for i in range(0, len(coords), tile_batch):
        chunk = coords[i:i + tile_batch]
        batch = torch.stack([image[:, y:y + tile, x:x + tile] for (y, x) in chunk], dim=0)
        with torch.autocast(device_type="cuda", enabled=use_ac):
            probs = torch.sigmoid(model(batch)).float()         # [b,C,tile,tile]
        if prob_sum is None:
            n_lesions = probs.shape[1]
            prob_sum = torch.zeros(n_lesions, H, W, device=device)
        for j, (y, x) in enumerate(chunk):
            prob_sum[:, y:y + tile, x:x + tile] += probs[j]
            weight[:, y:y + tile, x:x + tile] += 1
    if prob_sum is None:                          # whole image was background
        return torch.zeros(1, H, W, device=device)
    return prob_sum / weight.clamp(min=1e-6)
