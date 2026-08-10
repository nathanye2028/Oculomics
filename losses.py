"""
losses.py
=========
Supplementary segmentation losses and per-class diagnostics for sparse lesion
segmentation.

Relationship to fundus_utils
----------------------------
:mod:`fundus_utils` owns the primary objective — ``tversky_loss`` and
``focal_tversky_loss``, both with ``valid=`` masking for partially-annotated
sources. **This module does not reimplement them**; it imports them, and adds
the pieces that were missing:

* :func:`focal_bce_loss`  — focal binary cross-entropy (per-pixel gradient that
  behaves early in training, before any region overlap exists for Tversky to
  measure).
* :func:`lesion_seg_loss` — focal Tversky + focal BCE, the combination.
* :func:`per_class_dice` / :func:`per_class_sens_prec` — per-lesion diagnostics.

.. warning::
   ``fundus_utils`` uses **alpha = false negatives, beta = false positives**
   (so its ``alpha=0.7, beta=0.3`` default is recall-favouring). Some Tversky
   implementations use the opposite convention. Everything here follows
   ``fundus_utils``. Do not mix them.

Why plain Dice+BCE is not enough
--------------------------------
Measured positive-pixel fractions on FGADR, among images containing the lesion:

    MA   0.063%      EX   0.350%
    SE   0.259%      HE   0.601%

At 0.063% positives an all-background prediction already scores ~0.9994 pixel
accuracy, so BCE's gradient toward predicting anything is vanishingly small,
and Dice is unstable when its denominator is a few hundred pixels out of a
million. Tversky with alpha>beta makes a miss cost more than a false alarm —
both the right training signal and the right clinical bias for screening.

Why the per-class diagnostics matter
------------------------------------
Dice alone cannot distinguish "predicts nothing" from "predicts everything";
both score near zero, but only one is fixable by moving the threshold. A
checkpoint reporting ``MA dice 0.0`` is ambiguous until you see sensitivity
alongside it. :func:`per_class_sens_prec` resolves that.

All functions treat channels as **independent binary problems** (multi-label,
not softmax) — FGADR and IDRiD annotate each lesion separately and a pixel can
belong to more than one.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from fundus_utils import focal_tversky_loss, tversky_loss  # noqa: F401  (re-exported)

__all__ = [
    "tversky_loss",
    "focal_tversky_loss",
    "focal_bce_loss",
    "lesion_seg_loss",
    "per_class_dice",
    "per_class_sens_prec",
]


def _apply_valid(
    logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero out channels a sample does not annotate, matching fundus_utils."""
    if valid is None:
        return logits, target
    w = valid.to(logits.dtype)[:, :, None, None]          # [B,C,1,1]
    return logits * w, target * w


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal binary cross-entropy — down-weights the easy background majority.

    ``alpha`` weights the positive class. Computed from logits in a numerically
    stable way (no explicit sigmoid-then-log). ``valid`` [B,C] excludes
    unannotated channels, consistent with ``fundus_utils.focal_tversky_loss``.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)             # prob of the true class
    a_t = alpha * target + (1 - alpha) * (1 - target)
    loss = a_t * (1 - p_t).pow(gamma) * bce

    if valid is None:
        return loss.mean()
    w = valid.to(loss.dtype)[:, :, None, None]
    return (loss * w).sum() / w.expand_as(loss).sum().clamp(min=1.0)


def lesion_seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    focal_gamma: float = 2.0,
    bce_weight: float = 0.5,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal Tversky + weighted focal BCE.

    The Tversky term gives a region-overlap signal that survives extreme
    imbalance; the focal-BCE term gives a well-behaved per-pixel gradient early
    in training, when predictions and targets barely overlap and Tversky has
    little to work with. ``alpha``/``beta``/``gamma`` follow ``fundus_utils``.
    """
    ft = focal_tversky_loss(logits, target, alpha=alpha, beta=beta,
                            gamma=gamma, valid=valid)
    fb = focal_bce_loss(logits, target, gamma=focal_gamma, valid=valid)
    return ft + bce_weight * fb


@torch.no_grad()
def per_class_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresh: float = 0.5,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hard Dice per class, [C]. Classes absent from both prediction and target
    score 1.0 (nothing to find, nothing predicted)."""
    lg, tg = _apply_valid(logits, target, valid)
    pred = (torch.sigmoid(lg) > thresh).float()
    if valid is not None:
        pred = pred * valid.to(pred.dtype)[:, :, None, None]
    dims = (0, 2, 3)
    inter = (pred * tg).sum(dims)
    denom = pred.sum(dims) + tg.sum(dims)
    return torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))


@torch.no_grad()
def per_class_sens_prec(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresh: float = 0.5,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-class (sensitivity, precision) at ``thresh`` — each [C].

    The pair that reveals *how* a sparse-lesion model is failing. Watch
    sensitivity during early training: near-zero means the model has collapsed
    to all-background and needs a stronger false-negative penalty (raise
    ``alpha``); near-one with near-zero precision means it is over-predicting
    and will recover as training proceeds.
    """
    lg, tg = _apply_valid(logits, target, valid)
    pred = (torch.sigmoid(lg) > thresh).float()
    if valid is not None:
        pred = pred * valid.to(pred.dtype)[:, :, None, None]
    dims = (0, 2, 3)
    tp = (pred * tg).sum(dims)
    sens = tp / tg.sum(dims).clamp(min=1.0)
    prec = tp / pred.sum(dims).clamp(min=1.0)
    return sens, prec


def _report(name: str, value: float, grad: float) -> None:
    print(f"  {name:<26} loss={value:.4f}  grad_norm={grad:.6f}")


if __name__ == "__main__":
    # Self-check on synthetic targets at FGADR's measured MA prevalence.
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 128, 128
    target = (torch.rand(B, C, H, W) < 0.0006).float()          # ~0.06% positives
    base = torch.randn(B, C, H, W) * 0.1
    print(f"target positive fraction: {target.mean().item()*100:.4f}%")

    for name, fn in [
        ("dice (alpha=beta=0.5)", lambda x, t: tversky_loss(x, t, 0.5, 0.5)),
        ("tversky (0.7/0.3)", lambda x, t: tversky_loss(x, t, 0.7, 0.3)),
        ("focal_tversky", lambda x, t: focal_tversky_loss(x, t)),
        ("focal_bce", focal_bce_loss),
        ("lesion_seg_loss", lesion_seg_loss),
    ]:
        x = base.clone().requires_grad_(True)
        loss = fn(x, target)
        loss.backward()
        _report(name, loss.item(), x.grad.norm().item())

    # valid= must exclude a channel entirely, not score it as all-negative.
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * B)
    x = base.clone().requires_grad_(True)
    loss = lesion_seg_loss(x, target, valid=valid)
    loss.backward()
    print(f"\nwith valid={valid[0].tolist()}:")
    print(f"  loss={loss.item():.4f}  grad on masked channel="
          f"{x.grad[:, 3].abs().sum().item():.6e}  (should be 0)")

    # An all-background predictor must be punished, and diagnosable.
    empty = torch.full((B, C, H, W), -10.0)
    sens, prec = per_class_sens_prec(empty, target)
    print(f"\nall-background prediction:")
    print(f"  dice        : {[f'{v:.3f}' for v in per_class_dice(empty, target).tolist()]}")
    print(f"  sensitivity : {[f'{v:.3f}' for v in sens.tolist()]}  <- diagnoses collapse")
    print("losses.py self-check passed.")
