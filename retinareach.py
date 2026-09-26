"""
retinareach.py
==============
**RetinaReach** -- a phone-deployable, multi-target fundus screening model that
(1) re-calibrates itself to an unfamiliar camera from *unlabelled* captures and
(2) declares, per target and per device, whether it can screen for it at all.

This module is the model and its deploy-time machinery. Training and the full
evaluation live in ``train_retinareach.py``; the capability gate's statistics in
``capability_gate.py``; the Core ML graphs in ``export_retinareach.py``.

The network
-----------
One standard published trunk (default timm MobileNetV4-Conv-Small, the DR
student of record) and one **linear head per target** on its pooled embedding.
The heads are the rows of a single ``Linear(D, K)``: K independent logistic
read-outs that share nothing but the trunk. Image only -- clinical metadata is
never an input (plan section 9: no fusion); it exists only as the comparator
the capability gate measures the image against.

Two kinds of target, because the labels are asymmetric across devices:

* **source targets** (default ``dr_referable``, ``edema``): labelled on the
  tabletop source (BRSET). Trunk and heads are trained there; the handheld
  device never contributes a label. These carry the transfer experiment.
* **probe targets** (default: mBRSET's systemic comorbidities): labelled only on
  the handheld set. They are *linear probes* fit on the frozen, self-calibrated
  trunk, so the trunk still never sees a handheld label. Because they are fit in
  the calibrated feature space they only exist under self-calibration -- there is
  no "uncalibrated" systemic number, by construction.

Self-calibration (label-free, forward-only, no retraining)
----------------------------------------------------------
:func:`self_calibrate` is AdaBN (Li et al. 2016): BatchNorm statistics
re-estimated on N unlabelled captures from the new camera; weights untouched.
It reuses ``train_mbrset.adapt_bn`` so the number is the same procedure the DR
sweeps measured (+0.010 AUROC, significant in two 7-seed sweeps). With a small
N the target estimate is noisy, so ``prior_strength`` N0 shrinks it toward the
source statistics as in Schneider et al. (2020):

    mu  = (N0 * mu_source  + N * mu_target)  / (N0 + N)
    var = (N0 * var_source + N * var_target) / (N0 + N)

N0 = 0 is plain AdaBN; N0 -> infinity is no calibration.

The phone cannot run ``adapt_bn`` (Core ML folds BN into the convolutions), so
the deployable form is two graphs over the same weights:

* :class:`BatchStatCollector` -- a batch of B captures in, every BN layer's
  batch mean and unbiased variance out, with each layer normalised by its own
  batch statistics (exactly PyTorch train-mode BN). The app averages the output
  over its batches -- the same equal-weight cumulative average ``adapt_bn``
  computes; that measured vector drives the camera lookup, and a copy blended
  with the source prior drives inference.
* :class:`StatInputNet` -- one image + that statistics vector in, per-target
  probabilities out. BN is evaluated from the input vector instead of baked
  constants, so a new camera needs no re-export and no gradient step.

The camera signature
--------------------
The calibrated BN statistics *are* a label-free description of the camera.
:func:`camera_distance` compares two of them (per-channel symmetric KL between
the Gaussians, median over channels, mean over layers). Each validated device
ships a :class:`DeviceProfile`: its calibrated statistics, the sampling
envelope of that distance at each calibration size, and its capability table.
A new camera is matched to the nearest profile; if it falls outside that
profile's envelope, every target is reported ``NOT_VALIDATED`` -- the model has
never been checked on a camera like this and says so (:func:`device_capability`).
This is a conservative lookup, not a performance estimate.

The capability statuses (decided in ``capability_gate.gate_status``)
--------------------------------------------------------------------
    SUPPORTED          image beats the strongest metadata-only model on the same
                       patients by more than the noise floor, AND the shipped
                       threshold still reaches the sensitivity floor
    THRESHOLD_DRIFT    discriminates, but the shipped threshold misses the
                       sensitivity floor on this device (needs local labels)
    NO_IMAGE_EVIDENCE  the image does not beat the intake-form metadata
    UNDERPOWERED       too few positive patients on this device to decide
    NOT_VALIDATED      no labels for this target on this device, or the camera
                       is outside every validated profile
Only SUPPORTED targets are shown to the user; every other one is suppressed
with its reason.
"""
from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SYSTEMIC_TASKS

DEFAULT_BACKBONE = "timm:mobilenetv4_conv_small.e2400_r224_in1k"
SOURCE_TARGETS: Tuple[str, ...] = ("dr_referable", "edema")
PROBE_TARGETS: Tuple[str, ...] = tuple(SYSTEMIC_TASKS)

STATUSES = ("SUPPORTED", "THRESHOLD_DRIFT", "NO_IMAGE_EVIDENCE", "UNDERPOWERED", "NOT_VALIDATED")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

Stats = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class RetinaReachNet(nn.Module):
    """timm trunk + one linear read-out per target (rows of one ``Linear``).

    ``targets`` fixes the output order; :attr:`source_targets` are trained with
    the trunk, :attr:`probe_targets` are filled in later by
    :meth:`set_probe`. A trunk without BatchNorm is refused: self-calibration is
    the point of the model and a LayerNorm trunk has nothing to calibrate.
    """

    def __init__(self, source_targets: Sequence[str], probe_targets: Sequence[str] = (),
                 backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 dropout: float = 0.2, probe_size: int = 224) -> None:
        super().__init__()
        overlap = set(source_targets) & set(probe_targets)
        if overlap:
            raise ValueError(f"targets {sorted(overlap)} are both source and probe targets")
        if not backbone.startswith("timm:"):
            raise ValueError(f"backbone must be 'timm:<name>', got {backbone!r}")
        import timm
        self.backbone_name = backbone
        self.source_targets = list(source_targets)
        self.probe_targets = list(probe_targets)
        self.targets = self.source_targets + self.probe_targets
        self.trunk = timm.create_model(backbone[len("timm:"):], pretrained=pretrained, num_classes=0)
        if not bn_layers(self.trunk):
            raise ValueError(f"{backbone} has no BatchNorm layers: nothing to self-calibrate")
        # Probe the pooled width in eval mode (timm's num_features is the
        # pre-head width, and a train-mode probe would fold zeros into the BN
        # running statistics).
        was = self.trunk.training
        self.trunk.eval()
        with torch.no_grad():
            self.feat_dim = int(self.trunk(torch.zeros(1, 3, probe_size, probe_size)).shape[1])
        self.trunk.train(was)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, len(self.targets))
        with torch.no_grad():                   # probe rows are empty until set_probe
            self.head.weight[len(self.source_targets):].zero_()
            self.head.bias[len(self.source_targets):].zero_()

    def index(self, target: str) -> int:
        return self.targets.index(target)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [B, K] in :attr:`targets` order."""
        return self.head(self.drop(self.embed(x)))

    @torch.no_grad()
    def set_probe(self, target: str, weight: np.ndarray, bias: float) -> None:
        """Write a fitted linear probe (already in raw-embedding units) into its row."""
        i = self.index(target)
        if target not in self.probe_targets:
            raise ValueError(f"{target!r} is a source target; its head is trained, not probed")
        self.head.weight[i] = torch.as_tensor(np.asarray(weight, dtype=np.float32))
        self.head.bias[i] = float(bias)

    def config(self) -> dict:
        return {"backbone": self.backbone_name, "source_targets": self.source_targets,
                "probe_targets": self.probe_targets, "feat_dim": self.feat_dim}


def build_from_checkpoint(ck: dict, stats: str = "trained") -> RetinaReachNet:
    """Rebuild a ``train_retinareach.py`` checkpoint. ``stats`` = ``trained``
    (the as-trained BN running statistics) or a profile name whose calibrated
    statistics are loaded instead (e.g. ``handheld``)."""
    cfg = ck["config"]
    net = RetinaReachNet(cfg["source_targets"], cfg["probe_targets"], backbone=cfg["backbone"],
                         pretrained=False)
    net.load_state_dict(ck["model"])
    if stats != "trained":
        prof = ck["profiles"].get(stats)
        if prof is None:
            raise KeyError(f"no profile {stats!r} in checkpoint; have {sorted(ck['profiles'])}")
        set_bn_stats(net, stats_from_lists(prof["bn_stats"]))
    return net.eval()


# --------------------------------------------------------------------------- #
# BatchNorm statistics
# --------------------------------------------------------------------------- #
def bn_layers(model: nn.Module) -> List[Tuple[str, nn.modules.batchnorm._BatchNorm]]:
    """Every BatchNorm layer, in ``named_modules`` order -- the fixed order of the
    statistics vector the Core ML graphs exchange."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def get_bn_stats(model: nn.Module) -> Stats:
    return {n: (m.running_mean.detach().clone().float().cpu(),
                m.running_var.detach().clone().float().cpu())
            for n, m in bn_layers(model)}


@torch.no_grad()
def set_bn_stats(model: nn.Module, stats: Stats) -> None:
    layers = dict(bn_layers(model))
    missing = set(layers) - set(stats)
    if missing:
        raise KeyError(f"statistics missing for {len(missing)} BN layers, e.g. {sorted(missing)[:3]}")
    for n, m in layers.items():
        mu, var = stats[n]
        m.running_mean.copy_(mu.to(m.running_mean))
        m.running_var.copy_(var.to(m.running_var))


def blend_stats(source: Stats, target: Stats, n_target: int, prior_strength: float) -> Stats:
    """Schneider et al. (2020): shrink an N-image target estimate toward the
    source statistics with pseudo-count ``prior_strength`` (N0)."""
    if prior_strength <= 0:
        return {k: (m.clone(), v.clone()) for k, (m, v) in target.items()}
    a = prior_strength / (prior_strength + max(n_target, 0))
    return {k: ((a * source[k][0] + (1 - a) * target[k][0]),
                (a * source[k][1] + (1 - a) * target[k][1])) for k in target}


def stats_to_vector(stats: Stats, order: Sequence[str]) -> torch.Tensor:
    """[mean_1, var_1, mean_2, var_2, ...] in ``order`` -- the Core ML exchange format."""
    return torch.cat([t for n in order for t in (stats[n][0].float(), stats[n][1].float())])


def vector_to_stats(vec: torch.Tensor, model: nn.Module) -> Stats:
    out, i = {}, 0
    for n, m in bn_layers(model):
        c = m.num_features
        out[n] = (vec[i:i + c].clone(), vec[i + c:i + 2 * c].clone())
        i += 2 * c
    if i != vec.numel():
        raise ValueError(f"statistics vector has {vec.numel()} values, model needs {i}")
    return out


def stats_to_lists(stats: Stats) -> Dict[str, List[List[float]]]:
    """JSON/pickle-friendly form stored in checkpoints and profiles.json."""
    return {k: [m.tolist(), v.tolist()] for k, (m, v) in stats.items()}


def stats_from_lists(d: Dict[str, List[List[float]]]) -> Stats:
    return {k: (torch.tensor(m, dtype=torch.float32), torch.tensor(v, dtype=torch.float32))
            for k, (m, v) in d.items()}


# --------------------------------------------------------------------------- #
# Self-calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationInfo:
    n_images: int
    n_batches: int
    n_bn_layers: int
    prior_strength: float


@torch.no_grad()
def self_calibrate(model: nn.Module, loader: Iterable, device: torch.device,
                   prior_strength: float = 0.0) -> Tuple[nn.Module, CalibrationInfo]:
    """AdaBN on unlabelled images (``train_mbrset.adapt_bn``), then the optional
    Schneider prior. Returns a calibrated deep copy; ``model`` is untouched.

    ``loader`` yields dicts with ``"image"``; labels are never read. Its batch
    size is the calibration batch size and should equal the exported
    collector's (batch statistics depend on it). Batches of one are skipped
    by ``adapt_bn``, and counted out of ``n_images`` here.
    """
    from train_mbrset import adapt_bn
    counted = _CountingLoader(loader)
    adapted, n_bn = adapt_bn(model, counted, device)
    if n_bn == 0:
        raise ValueError("model has no BatchNorm layers: self-calibration is a no-op")
    if prior_strength > 0:
        set_bn_stats(adapted, blend_stats(get_bn_stats(model), get_bn_stats(adapted),
                                          counted.n_images, prior_strength))
    return adapted, CalibrationInfo(counted.n_images, counted.n_batches, n_bn, float(prior_strength))


class _CountingLoader:
    """Pass-through iterable that counts the images ``adapt_bn`` actually uses
    (it skips batches of one)."""

    def __init__(self, loader: Iterable) -> None:
        self.loader, self.n_images, self.n_batches = loader, 0, 0

    def __iter__(self):
        for batch in self.loader:
            b = int(batch["image"].shape[0])
            if b >= 2:
                self.n_images += b
                self.n_batches += 1
            yield batch


# --------------------------------------------------------------------------- #
# Deployable graphs: statistics as data instead of baked constants
# --------------------------------------------------------------------------- #
def _bn_affine(bn: nn.modules.batchnorm._BatchNorm, x: torch.Tensor,
               mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """BatchNorm with explicit statistics, then timm's fused drop/act if present
    (``BatchNormAct2d`` applies them inside the norm layer)."""
    shape = (1, -1) + (1,) * (x.dim() - 2)
    scale = torch.rsqrt(var + bn.eps)
    if bn.weight is not None:
        scale = scale * bn.weight
    y = (x - mean.view(shape)) * scale.view(shape)
    if bn.bias is not None:
        y = y + bn.bias.view(shape)
    if hasattr(bn, "drop"):
        y = bn.drop(y)
    if hasattr(bn, "act"):
        y = bn.act(y)
    return y


def _stat_input_forward(self, x):
    mean, var = self._rr_stats
    return _bn_affine(self, x, mean, var)


def _batch_stat_forward(self, x):
    dims = [0] + list(range(2, x.dim()))
    n = 1
    for d in dims:                                   # a Python int: a constant in the traced graph
        n *= int(x.shape[d])
    bessel = n / (n - 1) if n > 1 else 1.0
    mean = x.mean(dim=dims)
    var_b = ((x - mean.view((1, -1) + (1,) * (x.dim() - 2))) ** 2).mean(dim=dims)
    self._rr_sink[self._rr_index] = (mean, var_b * bessel)   # unbiased, as running_var
    return _bn_affine(self, x, mean, var_b)                  # biased, as train-mode BN


class _Normalise(nn.Module):
    """Raw 0-255 RGB -> ImageNet-normalised, inside the graph (Core ML's
    ImageType scale is a scalar; the ImageNet std is per channel)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean255", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std255", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        return (x - self.mean255) / self.std255


class StatInputNet(nn.Module):
    """Inference graph: ``(image 0-255 [B,3,H,W], stats [2*C_total]) ->
    per-target probabilities [B, K]``. Equal to the eval-mode model with those
    statistics loaded (tested), but the statistics are an input, so the phone
    re-calibrates by passing a new vector -- no re-export, no gradient."""

    def __init__(self, net: RetinaReachNet, raw_input: bool = True) -> None:
        super().__init__()
        self.net = copy.deepcopy(net).eval()
        self.pre = _Normalise() if raw_input else nn.Identity()
        self.order = [n for n, _ in bn_layers(self.net)]
        self._layers = [m for _, m in bn_layers(self.net)]
        self.sizes = [m.num_features for m in self._layers]
        for m in self._layers:
            m.forward = types.MethodType(_stat_input_forward, m)

    @property
    def stats_len(self) -> int:
        return 2 * sum(self.sizes)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        i = 0
        for m, c in zip(self._layers, self.sizes):
            m._rr_stats = (stats[i:i + c], stats[i + c:i + 2 * c])
            i += 2 * c
        return torch.sigmoid(self.net(self.pre(x)))


class BatchStatCollector(nn.Module):
    """Calibration graph: ``images 0-255 [B,3,H,W] -> stats [2*C_total]``, each
    BN layer normalised by -- and reporting -- its own batch statistics. The
    equal-weight mean of the outputs over batches reproduces ``adapt_bn``'s
    running statistics (tested)."""

    def __init__(self, net: RetinaReachNet, raw_input: bool = True) -> None:
        super().__init__()
        self.net = copy.deepcopy(net).eval()
        self.pre = _Normalise() if raw_input else nn.Identity()
        self._layers = [m for _, m in bn_layers(self.net)]
        self._sink: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, m in enumerate(self._layers):
            m._rr_index = i
            m._rr_sink = self._sink
            m.forward = types.MethodType(_batch_stat_forward, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._sink.clear()
        self.net.trunk(self.pre(x))              # the head holds no BN
        if len(self._sink) != len(self._layers):
            raise RuntimeError(f"{len(self._layers) - len(self._sink)} BN layers never ran")
        return torch.cat([t for i in range(len(self._layers)) for t in self._sink[i]])


def calibrate_with_collector(collector: BatchStatCollector,
                             batches: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, int]:
    """What the app does: average the collector over its capture batches (equal
    weight per batch, as ``adapt_bn``). Returns the MEASURED statistics vector
    and the number of images used. Use this unblended vector for the camera
    lookup (:func:`device_capability` -- envelopes are built from unblended
    statistics) and :func:`blend_vector` only for the vector passed to
    inference."""
    acc, nb, n = None, 0, 0
    with torch.no_grad():
        for x in batches:
            if x.shape[0] < 2:
                continue
            v = collector(x)
            acc = v if acc is None else acc + v
            nb += 1
            n += int(x.shape[0])
    if acc is None:
        raise ValueError("no calibration batch of size >= 2")
    return acc / nb, n


def blend_vector(source: torch.Tensor, measured: torch.Tensor, n: int,
                 prior_strength: float) -> torch.Tensor:
    """:func:`blend_stats` on the flat Core ML vector: the inference statistics."""
    if prior_strength <= 0:
        return measured
    a = prior_strength / (prior_strength + n)
    return a * source + (1 - a) * measured


# --------------------------------------------------------------------------- #
# Camera signature and the label-free device lookup
# --------------------------------------------------------------------------- #
def camera_distance(a: Stats, b: Stats, rel_floor: float = 1e-3) -> float:
    """Label-free distance between two cameras' calibrated BN statistics.

    Per channel, the symmetric KL between N(mu_a, var_a) and N(mu_b, var_b);
    median over a layer's channels (robust to dead channels, whose variances
    are floored at ``rel_floor`` x the layer's median variance); mean over
    layers. Zero for identical statistics, symmetric, unitless.
    """
    per_layer = []
    for k in a:
        ma, va = a[k][0].double(), a[k][1].double()
        mb, vb = b[k][0].double(), b[k][1].double()
        floor = rel_floor * float(torch.median(torch.cat([va, vb]))) + 1e-12
        va, vb = va.clamp_min(floor), vb.clamp_min(floor)
        skl = 0.5 * (va / vb + vb / va - 2.0) + 0.5 * (ma - mb) ** 2 * (1.0 / va + 1.0 / vb)
        per_layer.append(float(torch.median(skl)))
    return float(np.mean(per_layer)) if per_layer else float("nan")


@dataclass
class DeviceProfile:
    """A validated camera: its calibrated statistics, the distance envelope a
    fresh N-capture calibration of the SAME camera stays inside (95th
    percentile, keyed by N), and its capability rows (from the gate)."""
    name: str
    dataset: str
    bn_stats: Stats
    n_pool: int
    envelope: Dict[int, float] = field(default_factory=dict)
    capability: List[dict] = field(default_factory=list)

    def radius(self, n: int) -> float:
        """Envelope radius for an N-image calibration: the smallest tabulated
        N >= n (more images -> tighter), else the largest tabulated N."""
        if not self.envelope:
            return float("nan")
        ks = sorted(self.envelope)
        k = next((k for k in ks if k >= n), ks[-1])
        return self.envelope[k]

    def to_dict(self) -> dict:
        return {"name": self.name, "dataset": self.dataset, "n_pool": self.n_pool,
                "bn_stats": stats_to_lists(self.bn_stats),
                "envelope": {str(k): v for k, v in self.envelope.items()},
                "capability": self.capability}

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfile":
        return cls(d["name"], d["dataset"], stats_from_lists(d["bn_stats"]), int(d["n_pool"]),
                   {int(k): float(v) for k, v in d.get("envelope", {}).items()},
                   list(d.get("capability", [])))


def device_capability(profiles: Sequence[DeviceProfile], stats: Stats, n_images: int,
                      targets: Sequence[str], protocol: str = "calibrated") -> dict:
    """The report the phone prints after calibrating on ``n_images`` captures.

    Nearest validated profile by :func:`camera_distance`; inside its envelope
    -> that profile's capability rows for ``protocol``; outside every envelope
    -> every target NOT_VALIDATED. Never promotes a status: an unfamiliar
    camera can only lose capability, not gain it.
    """
    dists = {p.name: camera_distance(stats, p.bn_stats) for p in profiles}
    near = min(profiles, key=lambda p: dists[p.name])
    radius = near.radius(n_images)
    inside = bool(dists[near.name] <= radius) if radius == radius else False
    rows = {r["target"]: r for r in near.capability if r.get("protocol") == protocol}
    report = []
    for t in targets:
        r = rows.get(t)
        if not inside:
            report.append({"target": t, "status": "NOT_VALIDATED",
                           "reason": f"camera outside every validated profile (nearest "
                                     f"{near.name}: distance {dists[near.name]:.3g} > "
                                     f"envelope {radius:.3g} at N={n_images})"})
        elif r is None:
            report.append({"target": t, "status": "NOT_VALIDATED",
                           "reason": f"no evaluation of {t} on {near.name}"})
        else:
            report.append({"target": t, "status": r["status"], "reason": r.get("reason", "")})
    return {"nearest_profile": near.name, "distances": dists, "radius": radius,
            "inside_envelope": inside, "n_images": int(n_images),
            "shown": [r["target"] for r in report if r["status"] == "SUPPORTED"],
            "targets": report}
