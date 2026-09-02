"""
losses.py
=========
Supplementary segmentation losses for sparse lesion segmentation.

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

Per-lesion *diagnostics* (Dice / IoU / sensitivity / precision from accumulated
counts) live in :mod:`metrics` — ``dice_iou_from_counts`` and
``sens_prec_from_counts`` — which is what the trainers and evaluators use. This
module used to carry its own ``per_class_dice`` / ``per_class_sens_prec``; they
were removed because nothing outside this file called them and they disagreed
with :mod:`metrics` on the one edge case that matters (a lesion absent from
both prediction and target scored 1.0 here, NaN there — averaging a 1.0 in
silently inflates the mean).

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

Dice alone cannot distinguish "predicts nothing" from "predicts everything";
both score near zero, but only one is fixable by moving the threshold. Read
sensitivity next to Dice (``metrics.sens_prec_from_counts``) to tell them apart.

All functions treat channels as **independent binary problems** (multi-label,
not softmax) — FGADR and IDRiD annotate each lesion separately and a pixel can
belong to more than one.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from fundus_utils import focal_tversky_loss, tversky_loss  # noqa: F401  (re-exported)

__all__ = [
    "tversky_loss",
    "focal_tversky_loss",
    "focal_bce_loss",
    "lesion_seg_loss",
]


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

    NB for gradient accumulation (``train_idrid.py --accum-steps``): the BCE
    term is a per-pixel mean, so summing it over micro-batches equals the
    full-batch value; the Tversky term pools tp/fp/fn over the whole batch
    *before* dividing, so accumulated micro-batches are NOT gradient-equivalent
    to one large batch for that term.
    """
    ft = focal_tversky_loss(logits, target, alpha=alpha, beta=beta,
                            gamma=gamma, valid=valid)
    fb = focal_bce_loss(logits, target, gamma=focal_gamma, valid=valid)
    return ft + bce_weight * fb


def _report(name: str, value: float, grad: float) -> None:
    print(f"  {name:<26} loss={value:.4f}  grad_norm={grad:.6f}")


if __name__ == "__main__":
    # Self-check of the two live losses on synthetic targets at FGADR's
    # measured MA prevalence.
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
    masked_grad = x.grad[:, 3].abs().sum().item()
    print(f"\nwith valid={valid[0].tolist()}:")
    print(f"  loss={loss.item():.4f}  grad on masked channel={masked_grad:.6e}  (should be 0)")
    assert masked_grad == 0.0, "valid= leaked gradient into an unannotated channel"

    # An all-background predictor must be punished: its loss must exceed a
    # mildly informative one, or the objective cannot pull the model off the
    # all-negative solution that 99.94% background makes so attractive.
    empty = torch.full((B, C, H, W), -10.0)
    hint = torch.where(target > 0, torch.full_like(target, 2.0), torch.full_like(target, -10.0))
    l_empty, l_hint = lesion_seg_loss(empty, target).item(), lesion_seg_loss(hint, target).item()
    print(f"\nall-background loss {l_empty:.4f}  vs  partially-right loss {l_hint:.4f}")
    assert l_empty > l_hint
    print("losses.py self-check passed.")
