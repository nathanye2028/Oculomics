"""
metrics.py
==========
Evaluation metrics + lightweight logging shared by the trainers.

Classification:
  * ``quadratic_weighted_kappa`` — the clinical standard for ordinal DR grading
    (penalises being-off-by-2 more than off-by-1).
Segmentation:
  * ``dice_iou_from_counts`` — per-channel Dice **and** IoU from accumulated
    intersection / pred-sum / target-sum (cheap; dataset-level).
  * ``binary_average_precision`` — AUPRC (the right metric for tiny, rare
    lesions where Dice at a fixed 0.5 threshold is harsh). For whole-image eval.
Logging:
  * ``CSVLogger`` — append per-epoch metric rows to a CSV (open in any tool,
    no TensorBoard dependency).
  * ``get_tb_writer`` — optional TensorBoard SummaryWriter if installed.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(preds, targets, num_classes: int) -> float:
    """Cohen's quadratic-weighted kappa for ordinal labels (e.g. DR grade 0-4)."""
    preds = np.asarray(preds).astype(int)
    targets = np.asarray(targets).astype(int)
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(targets, preds):
        O[t, p] += 1
    idx = np.arange(num_classes)
    W = (idx[:, None] - idx[None, :]) ** 2 / float((num_classes - 1) ** 2 or 1)
    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    n = O.sum()
    E = np.outer(act, pred) / max(n, 1.0)
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def dice_iou_from_counts(inter: torch.Tensor, psum: torch.Tensor, tsum: torch.Tensor,
                         lesions: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-lesion Dice and IoU from accumulated counts.

    inter = sum(pred*target), psum = sum(pred), tsum = sum(target)  (per channel)

    A lesion absent from BOTH prediction and target over the whole eval set
    (psum + tsum == 0, e.g. SE missing from a small val split) is scored NaN,
    not 0 — there was nothing to segment, so a 0 would silently deflate any
    mean taken over the channels. Average with :func:`mean_present`.
    """
    absent = (psum + tsum) == 0
    dice = (2 * inter / (psum + tsum).clamp(min=1e-6)).cpu()
    union = (psum + tsum - inter).clamp(min=1e-6)
    iou = (inter / union).cpu()
    nan = float("nan")
    d = {c: (nan if absent[i] else dice[i].item()) for i, c in enumerate(lesions)}
    j = {c: (nan if absent[i] else iou[i].item()) for i, c in enumerate(lesions)}
    return d, j


def sens_prec_from_counts(inter: torch.Tensor, psum: torch.Tensor, tsum: torch.Tensor,
                          lesions: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-lesion sensitivity (inter/tsum) and precision (inter/psum) from the
    same accumulated counts as :func:`dice_iou_from_counts`.

    The pair that disambiguates a near-zero Dice: sensitivity ~0 means the model
    collapsed to all-background; sensitivity high with precision ~0 means it is
    over-predicting (recoverable by threshold/training). NaN where undefined.
    """
    nan = float("nan")
    sens = {c: (inter[i] / tsum[i]).item() if tsum[i] > 0 else nan
            for i, c in enumerate(lesions)}
    prec = {c: (inter[i] / psum[i]).item() if psum[i] > 0 else nan
            for i, c in enumerate(lesions)}
    return sens, prec


def mean_present(d: Dict[str, float]) -> float:
    """Mean over the non-NaN values of a per-lesion metric dict."""
    vals = [v for v in d.values() if v == v]          # NaN != NaN
    return sum(vals) / len(vals) if vals else float("nan")


def binary_average_precision(probs, targets) -> float:
    """AUPRC over flattened probabilities/targets. NaN if no positives."""
    from sklearn.metrics import average_precision_score
    t = np.asarray(targets).ravel().astype(int)
    if t.sum() == 0:
        return float("nan")
    p = np.asarray(probs).ravel().astype(np.float64)
    return float(average_precision_score(t, p))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class CSVLogger:
    """Append metric dict rows to a CSV (header inferred from the first row)."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = None
        self._w = None
        self._fields: Optional[list] = None

    def log(self, row: Dict[str, object]) -> None:
        if self._f is None:
            self._fields = list(row.keys())
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self._fields)
            self._w.writeheader()
        self._w.writerow({k: row.get(k, "") for k in self._fields})
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def get_tb_writer(log_dir: Optional[str]):
    """Return a TensorBoard SummaryWriter if available + requested, else None."""
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None
