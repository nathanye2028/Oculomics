"""
model.py
========
Baseline 2D segmentation network for ocular-pathology (retinal-lesion)
segmentation, with injection points for **Guided Context Gating (GCG)** blocks.

Design constraints (per project spec)
-------------------------------------
* Pure image -> mask. **No systemic / demographic metadata fusion** — the
  network sees only the fundus image, so learned features are spatial-visual.
* Encoder: a lightweight, mobile-capable backbone. **MobileNetV3-Large**
  (torchvision) is the default and the one all pre-2026-08 results were measured
  with; ``encoder=`` selects a modern alternative (see :func:`build_encoder`).
* Decoder: a U-Net-style upsampling path with skip connections from the
  encoder stages.
* **GCG injection**: every decoder skip connection passes through a GCG block
  *before* being concatenated. GCG is a spatial-channel attention gate intended
  to force decoder channels to correlate with distinct lesion patterns
  (microaneurysms, exudates, ...) and suppress mobile-capture background noise.
  The default :class:`GuidedContextGating` here is a documented, working
  baseline; swap in your custom implementation via the ``gcg_factory`` arg.

The model returns per-pixel logits of shape ``[B, num_classes, H, W]`` at the
input resolution (single-channel for binary lesion-vs-background).

Where the mobile budget actually goes  (MEASURED — read before optimising)
--------------------------------------------------------------------------
Profiling the default (MobileNetV3-Large, dense decoder) at 512x512 gives
**14.09 GMAC**, split encoder **7.9%** / decoder **92.1%**. So on paper the
decoder is the whole mobile cost. **On the actual Core ML deployment path that
turns out not to matter at all**, and the numbers below are the reason this
module keeps the dense decoder as the default.

Core ML fp16 at 512x512, Apple silicon, median of 60 runs after 10 warmups
(``ComputeUnit.ALL`` = ANE/GPU eligible; ``CPU_ONLY`` for the fallback path):

===============================  =====  =======  =======  =======
config                            GMAC  ANE med  CPU med   params
===============================  =====  =======  =======  =======
mobilenetv3 / dense (default)    14.09    9.0 ms  46.5 ms   6.84M
mobilenetv3 / separable           3.26    9.2 ms  46.6 ms   3.53M
mobilenetv3 / dense + lat=256    12.41    8.4 ms  40.3 ms   5.38M
mobilenetv4_m / separable         6.86    9.3 ms  83.5 ms   7.84M
efficientvit_b1 / separable       4.61   12.7 ms  61.2 ms   5.05M
===============================  =====  =======  =======  =======

**A 4.33x MAC reduction bought 0% latency** — on either compute path. This
network is memory-bandwidth-bound, not compute-bound: at 512x512 the activation
tensors dominate, and factoring a dense 3x3 into DW+PW cuts arithmetic while
*adding* an intermediate activation to write and re-read. Depthwise convs have
very low arithmetic intensity, so they cannot convert a FLOP win into a time
win here. (This is precisely why MobileNetV4's authors searched against
on-device latency instead of FLOPs.)

Practical consequences, in order of importance:

1. **Never justify an architecture change here with MAC counts.** Export it and
   time it. ``export_coreml.py --benchmark`` is the ground truth; the ONNX-INT8
   CPU number in ``edge_export/seg_edge_metrics.json`` (135 ms) is ~15x the real
   Core ML figure and should not be quoted as the deployment latency.
2. **There is a lot of latency headroom.** ~9 ms at 512x512 means resolution is
   affordable, and resolution is the lever that matters for microaneurysms
   (~1-2 px). Same config, ANE median: 512 -> 7.8 ms, 768 -> 21.6 ms,
   1024 -> 34.3 ms, 1280 -> 48.3 ms.
3. ``lateral_channels`` is the one cost knob that paid off on both paths
   (1.07x ANE, 1.15x CPU, and -1.46M params), because it removes activation
   *traffic* — 960 -> 256 channels at stride 32 — rather than just arithmetic.

The two knobs, both defaulting to the original architecture so that every
checkpoint under ``checkpoints/`` and ``exp_*/`` still loads:

``decoder="separable"``
    Replaces each decoder stage's two dense 3x3 convs with depthwise-separable
    blocks (3x3 DW -> 1x1 PW). Cuts MACs ~8.7x per stage at ``c_out=256`` and
    params by half. Retained because it is the instrument that established the
    result above, and because on a compute-bound target (some Android NPUs, or
    a CPU-only fallback with better depthwise kernels) it may yet pay — but on
    Apple silicon it does not. Not recommended as a default.

``lateral_channels=N``
    1x1-projects the deepest encoder feature (960ch for both MobileNetV3-L and
    MobileNetV4) down to ``N`` before the decoder consumes it. Without it,
    decoder stage 0 fuses ``960 + 112 = 1072`` channels through a 256-wide 3x3,
    23% of the network's MACs. **This is the one to use**:
    ``--decoder dense --lateral-channels 256``.

``decoder="separable"`` switches ``lateral_channels`` on by default; pass
``lateral_channels`` explicitly to ablate the two apart.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Guided Context Gating (GCG) — default baseline block
# --------------------------------------------------------------------------- #
class GuidedContextGating(nn.Module):
    """Guided Context Gating block (baseline implementation).

    A spatial-channel attention gate applied to an encoder *skip* feature map,
    optionally *guided* by the coarser decoder feature that will be fused with
    it. The guidance lets deep semantic context (where lesions likely are)
    modulate which shallow, high-resolution channels survive — concentrating
    capacity on lesion-correlated channels and gating out background noise.

    Two attention paths are combined:
      * **Channel gate** — squeeze-and-excite style; "which lesion channels".
      * **Spatial gate**  — additive-attention map; "where the lesions are".

    Parameters
    ----------
    skip_channels : channels of the high-res encoder skip feature (the input
                    that gets gated and returned).
    guide_channels : channels of the coarse decoder guidance feature; pass 0 /
                    None for an unguided (self-attention) gate.
    reduction : bottleneck ratio for the channel gate.

    Shape
    -----
    forward(skip, guide) -> tensor with the same shape as ``skip``.
    The guide is resized to the skip's spatial size internally.

    NB: This is a drop-in baseline. To inject a custom GCG, pass a
    ``gcg_factory(skip_channels, guide_channels) -> nn.Module`` to the U-Net;
    the module must accept ``forward(skip, guide)`` and return a tensor shaped
    like ``skip``.
    """

    def __init__(
        self,
        skip_channels: int,
        guide_channels: Optional[int] = None,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.guided = bool(guide_channels)
        inter = max(skip_channels // reduction, 4)

        # Project guidance to the skip's channel count (for spatial gate).
        if self.guided:
            self.guide_proj = nn.Conv2d(guide_channels, skip_channels, 1, bias=True)
        self.skip_proj = nn.Conv2d(skip_channels, skip_channels, 1, bias=True)

        # Spatial attention: 1x1 -> ReLU -> 1x1 -> sigmoid producing an [B,1,H,W] map.
        self.spatial = nn.Sequential(
            nn.Conv2d(skip_channels, inter, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, 1, 1, bias=True),
        )

        # Channel attention (squeeze-excite) over the (optionally guided) skip.
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, inter, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, skip_channels, 1, bias=True),
        )

    def forward(self, skip: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.skip_proj(skip)
        if self.guided and guide is not None:
            g = F.interpolate(guide, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.guide_proj(g)
        x = F.relu(x, inplace=True)

        spatial_att = torch.sigmoid(self.spatial(x))          # [B,1,H,W]
        channel_att = torch.sigmoid(self.channel(x))          # [B,C,1,1]
        return skip * spatial_att * channel_att               # gated skip


# --------------------------------------------------------------------------- #
# Encoders (multi-scale feature taps)
# --------------------------------------------------------------------------- #
class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Large feature extractor returning 5 stages at strides 2..32.

    Taps the ``features`` blocks where the spatial resolution halves, yielding
    skip features for a U-Net decoder.

    ``pretrained`` defaults to **True**. Initialisation dominates on datasets
    this small — ImageNet transfer alone moved mean Dice 0.183 -> 0.387 on IDRiD
    (see :mod:`pretrain_encoder`), which is larger than any architectural change
    measured in this project. Training from scratch is the exception and must be
    asked for explicitly; scripts that immediately load a checkpoint over these
    weights (export, eval, deployment) pass ``pretrained=False`` to skip the
    download.
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_large

        weights = None
        if pretrained:
            from torchvision.models import MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_large(weights=weights)
        features = backbone.features

        if in_channels != 3:
            first = features[0][0]
            features[0][0] = nn.Conv2d(
                in_channels, first.out_channels, kernel_size=first.kernel_size,
                stride=first.stride, padding=first.padding, bias=False,
            )

        # Block index AFTER which the resolution has reached the given stride.
        # mobilenet_v3_large.features: indices 0..16. Resolution halves at the
        # start of blocks 1, 2, 4, 7, 13 (strides 2,4,8,16,32 respectively).
        self._stage_ends = [1, 3, 6, 12, 16]
        self.features = features
        # Output channel counts at each tapped stage (from torchvision arch).
        self.out_channels: List[int] = [16, 24, 40, 112, 960]
        # Spatial stride of each tapped stage, coarsest-last. The decoder walks
        # this ladder, so an encoder whose finest tap is coarser than 2 (e.g.
        # EfficientViT, which starts at 4) is handled without special-casing.
        self.reductions: List[int] = [2, 4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        end_set = set(self._stage_ends)
        for i, block in enumerate(self.features):
            x = block(x)
            if i in end_set:
                feats.append(x)
        return feats  # [s2, s4, s8, s16, s32]


class TimmEncoder(nn.Module):
    """``timm`` backbone in ``features_only`` mode, exposing the same contract.

    Provides ``out_channels`` / ``reductions`` and returns features finest-first,
    matching :class:`MobileNetV3Encoder`. Channel counts and strides are read
    from ``timm``'s ``feature_info`` rather than hardcoded, because they differ
    per backbone — notably EfficientViT emits only 4 taps (strides 4..32) with
    no stride-2 feature at all.
    """

    def __init__(self, timm_name: str, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, features_only=True, in_chans=in_channels,
        )
        info = self.backbone.feature_info
        self.out_channels: List[int] = list(info.channels())
        self.reductions: List[int] = list(info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(x))


# name -> timm model id. ``mobilenetv3`` is the torchvision default and is
# deliberately NOT here; it is the pre-existing, already-benchmarked encoder.
TIMM_ENCODERS: Dict[str, str] = {
    # MobileNetV4 (2024) — the direct successor to MobileNetV3. Universal
    # Inverted Bottleneck blocks, searched with on-NPU latency in the loop.
    # 5 taps at strides 2..32, so it is a drop-in for MobileNetV3-Large.
    "mobilenetv4_s": "mobilenetv4_conv_small",
    "mobilenetv4_m": "mobilenetv4_conv_medium",
    "mobilenetv4_hybrid_m": "mobilenetv4_hybrid_medium",
    # EfficientViT (MIT) — ReLU *linear* attention, so cost stays O(n) in tokens
    # instead of O(n^2); designed for high-resolution dense prediction on edge
    # hardware. NB: 4 taps at strides 4..32 (no stride-2 feature).
    "efficientvit_b1": "efficientvit_b1",
    "efficientvit_b2": "efficientvit_b2",
}

ENCODER_NAMES: Tuple[str, ...] = ("mobilenetv3",) + tuple(TIMM_ENCODERS)


def build_encoder(name: str = "mobilenetv3", pretrained: bool = True,
                  in_channels: int = 3) -> nn.Module:
    """Encoder factory. ``name`` is one of :data:`ENCODER_NAMES`.

    ``mobilenetv3`` is the default and the encoder every result recorded in this
    repo before 2026-08 was measured with — keep it for like-for-like
    comparisons. The rest need ``timm`` (see requirements.txt).
    """
    if name == "mobilenetv3":
        return MobileNetV3Encoder(pretrained=pretrained, in_channels=in_channels)
    if name in TIMM_ENCODERS:
        return TimmEncoder(TIMM_ENCODERS[name], pretrained=pretrained, in_channels=in_channels)
    raise ValueError(f"Unknown encoder {name!r}; expected one of {ENCODER_NAMES}")


# --------------------------------------------------------------------------- #
# Decoder building blocks
# --------------------------------------------------------------------------- #
class _ConvBNAct(nn.Sequential):
    """Dense 3x3 conv -> BN -> ReLU. Costs ``HW * c_in * c_out * 9`` MAC."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class _SepConvBNAct(nn.Sequential):
    """Depthwise-separable 3x3: DW -> BN -> ReLU -> PW 1x1 -> BN -> ReLU.

    Costs ``HW * c_in * (9 + c_out)`` MAC against the dense block's
    ``HW * c_in * c_out * 9`` — a ``1/9 + 1/c_out`` fraction, so ~8.7x cheaper
    at ``c_out=256`` and ~7.2x at ``c_out=32``. The depthwise conv keeps the 3x3
    spatial receptive field; only the channel mixing is factored out into the
    1x1, which is the trade MobileNet's segmentation heads already make.

    **That MAC win does not become a latency win on Apple silicon** — measured
    9.2 ms vs the dense decoder's 9.0 ms at 512x512, and identical on CPU. See
    the module docstring: this network is bandwidth-bound, and DW+PW writes an
    extra intermediate activation. Kept for the record and for compute-bound
    targets; do not reach for it expecting a speedup.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


DECODER_BLOCKS: Dict[str, Callable[[int, int], nn.Module]] = {
    "dense": _ConvBNAct,
    "separable": _SepConvBNAct,
}


class DecoderBlock(nn.Module):
    """Upsample the decoder feature, GCG-gate the skip, concat, then fuse.

    ``skip_channels=0`` builds a skip-less stage that only upsamples and fuses.
    That is needed for encoders whose finest tap is coarser than stride 2
    (EfficientViT starts at 4), where the decoder must cover the remaining
    octave to full resolution on its own. Skip-less stages carry no GCG block,
    since there is no skip to gate.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        gcg_factory: Optional[Callable[[int, int], nn.Module]] = None,
        block: str = "dense",
    ) -> None:
        super().__init__()
        conv = DECODER_BLOCKS[block]
        self.has_skip = skip_channels > 0
        self.gcg = gcg_factory(skip_channels, in_channels) if (gcg_factory and self.has_skip) else None
        self.fuse = nn.Sequential(
            conv(in_channels + skip_channels, out_channels),
            conv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.gcg is not None:
                skip = self.gcg(skip, x)      # guided gating, guide = upsampled decoder feat
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.fuse(x)


# --------------------------------------------------------------------------- #
# U-Net with a mobile encoder + GCG-gated skips
# --------------------------------------------------------------------------- #
class GCGUNet(nn.Module):
    """2D U-Net (mobile encoder) with GCG-gated skip connections.

    Parameters
    ----------
    num_classes : output channels of the per-pixel logits (1 for binary).
    pretrained  : load ImageNet weights into the encoder. **Defaults to True** —
                  see :class:`MobileNetV3Encoder` for why scratch training is
                  the opt-in case, not the default.
    encoder     : backbone name from :data:`ENCODER_NAMES`. Default
                  ``mobilenetv3`` reproduces every prior result in this repo.
    decoder     : ``"dense"`` (the original) or ``"separable"`` (mobile-first,
                  depthwise-separable fuse convs). ``"separable"`` also switches
                  ``lateral_channels`` on by default.
    lateral_channels : if >0, 1x1-project the deepest encoder feature to this
                  many channels before the decoder consumes it. Kills the
                  ``960+112 -> 256`` dense 3x3 that is 23% of the default net.
                  ``None`` means "pick by ``decoder``"; pass an int (or 0) to
                  ablate this independently of the block type.
    decoder_channels : channel widths of the 5 decoder stages (coarse->fine).
    use_gcg     : insert GCG blocks on the skip connections.
    gcg_factory : custom ``(skip_ch, guide_ch) -> nn.Module``; defaults to
                  :class:`GuidedContextGating`. Lets you inject your own GCG.
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        pretrained: bool = True,
        encoder: str = "mobilenetv3",
        decoder: str = "dense",
        lateral_channels: Optional[int] = None,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        use_gcg: bool = True,
        gcg_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if decoder not in DECODER_BLOCKS:
            raise ValueError(f"decoder must be one of {tuple(DECODER_BLOCKS)}, got {decoder!r}")
        self.encoder = build_encoder(encoder, pretrained=pretrained, in_channels=in_channels)
        enc_ch = list(self.encoder.out_channels)          # finest -> coarsest
        reductions = list(self.encoder.reductions)
        conv = DECODER_BLOCKS[decoder]

        # Default the lateral projection on for the separable decoder: it is what
        # makes stage 0 cheap, and leaving it off would understate the win.
        if lateral_channels is None:
            lateral_channels = decoder_channels[0] if decoder == "separable" else 0

        if use_gcg:
            factory = gcg_factory or (lambda s, g: GuidedContextGating(s, g))
        else:
            factory = None

        dec = list(decoder_channels)
        # One width per skip-fusing stage, one per skip-less octave needed to
        # reach stride 2, plus one for the full-resolution stage.
        n_octaves = max(reductions[0].bit_length() - 2, 0)   # stride 2 -> 0, 4 -> 1, 8 -> 2
        n_needed = (len(enc_ch) - 1) + n_octaves + 1
        if len(dec) < n_needed:
            raise ValueError(
                f"decoder_channels needs >= {n_needed} entries for a "
                f"{len(enc_ch)}-tap encoder starting at stride {reductions[0]}, "
                f"got {len(dec)}")

        # Optional 1x1 reduction of the deepest feature. Both MobileNetV3-L and
        # MobileNetV4 emit 960 channels at stride 32; feeding that straight into
        # the decoder is the single most expensive op in the network.
        deep_ch = enc_ch[-1]
        if lateral_channels:
            self.lateral = nn.Sequential(
                nn.Conv2d(deep_ch, lateral_channels, 1, bias=False),
                nn.BatchNorm2d(lateral_channels),
                nn.ReLU(inplace=True),
            )
            deep_ch = lateral_channels
        else:
            self.lateral = None

        # Decoder consumes encoder stages coarse->fine; skips are the shallower
        # taps. x starts as the (optionally projected) deepest feature.
        skips = enc_ch[-2::-1]                      # e.g. [112, 40, 24, 16]
        ins = [deep_ch] + dec[:len(skips) - 1]
        self.decoders = nn.ModuleList([
            DecoderBlock(ins[i], skips[i], dec[i], gcg_factory=factory, block=decoder)
            for i in range(len(skips))
        ])
        cur_ch = dec[len(skips) - 1]
        cur_red = reductions[0]                     # stride we have reached

        # Cover any octaves between the encoder's finest tap and stride 2 with
        # skip-less upsample stages (EfficientViT's finest tap is stride 4).
        mid: List[nn.Module] = []
        next_dec = len(skips)
        while cur_red > 2:
            out_ch = dec[next_dec]
            mid.append(DecoderBlock(cur_ch, 0, out_ch, gcg_factory=None, block=decoder))
            cur_ch, cur_red, next_dec = out_ch, cur_red // 2, next_dec + 1
        self.mid_ups = nn.ModuleList(mid) if mid else None

        # Full-resolution path. The encoder's shallowest tap is at stride >= 2,
        # so without this the decoder's finest learned features are half-res and
        # the head just bilinearly upsamples them -> tiny lesions
        # (microaneurysms, ~1-2 px) are unrecoverable. The stem is a stride-1
        # feature of the input; the final decoder stage upsamples to full res and
        # fuses it (gated like the others) so the head runs at full resolution.
        #
        # The stem's first conv stays dense: it reads 3 input channels, where a
        # depthwise factorisation would save ~nothing and cost representation.
        stem_ch = dec[-1]
        self.stem = nn.Sequential(
            _ConvBNAct(in_channels, stem_ch),
            conv(stem_ch, stem_ch),
        )
        self.up_full = DecoderBlock(cur_ch, stem_ch, dec[-1],
                                    gcg_factory=factory, block=decoder)
        self.head = nn.Conv2d(dec[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        stem = self.stem(x)                       # full-resolution features
        feats = self.encoder(x)                   # finest -> coarsest
        d = feats[-1]
        if self.lateral is not None:
            d = self.lateral(d)
        for i, block in enumerate(self.decoders):
            d = block(d, feats[-2 - i])
        if self.mid_ups is not None:
            for block in self.mid_ups:
                d = block(d)                      # skip-less octave
        d = self.up_full(d, stem)                 # -> full res, learned conv at H x W
        logits = self.head(d)
        if logits.shape[-2:] != input_hw:         # safety for odd input sizes
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


def build_model(
    arch: str = "gcg_unet",
    num_classes: int = 1,
    pretrained: bool = True,
    use_gcg: bool = True,
    gcg_factory: Optional[Callable[[int, int], nn.Module]] = None,
    encoder: str = "mobilenetv3",
    decoder: str = "dense",
    lateral_channels: Optional[int] = None,
) -> nn.Module:
    """Factory. ``arch='gcg_unet'`` -> mobile-encoder U-Net with GCG skips.

    ``pretrained`` defaults to True — see :class:`MobileNetV3Encoder`. Pass
    False only when a checkpoint is about to overwrite the encoder anyway
    (export/eval) or when scratch init is the point (ablation, overfit test).

    ``encoder`` / ``decoder`` / ``lateral_channels`` default to the original
    architecture, so callers that do not pass them are unaffected.
    """
    if arch == "gcg_unet":
        return GCGUNet(
            num_classes=num_classes, pretrained=pretrained,
            use_gcg=use_gcg, gcg_factory=gcg_factory,
            encoder=encoder, decoder=decoder, lateral_channels=lateral_channels,
        )
    raise ValueError(f"Unknown arch {arch!r}")


def arch_cfg_from_checkpoint(ck: object) -> Dict[str, object]:
    """Recover ``encoder`` / ``decoder`` / ``lateral_channels`` from a checkpoint.

    ``train_idrid.py`` records ``vars(args)`` under ``"args"``, so a checkpoint
    knows which architecture it was trained with. Eval and export read it from
    there instead of requiring the caller to remember matching flags — a
    mismatch would otherwise surface as a silent pile of missing/unexpected keys
    and a model scoring near chance.

    Checkpoints written before these flags existed carry no record of them and
    correctly fall back to the original architecture.
    """
    cfg: Dict[str, object] = {"encoder": "mobilenetv3", "decoder": "dense",
                              "lateral_channels": None}
    saved = ck.get("args") if isinstance(ck, dict) else None
    if isinstance(saved, dict):
        cfg["encoder"] = saved.get("encoder", cfg["encoder"])
        cfg["decoder"] = saved.get("decoder", cfg["decoder"])
        lat = saved.get("lateral_channels", -1)
        cfg["lateral_channels"] = None if lat is None or lat < 0 else lat
    return cfg


if __name__ == "__main__":
    # Tiny self-check (no data needed): shapes + one backward pass, for the
    # original build and each mobile-first variant.
    def macs_of(net: nn.Module, size: int = 512) -> Tuple[float, float]:
        """(total GMAC, encoder GMAC) via conv forward hooks."""
        tally: Dict[str, int] = {}

        def hook(name):
            def f(m, _i, o):
                tally[name] = tally.get(name, 0) + (
                    m.in_channels // m.groups * m.out_channels
                    * m.kernel_size[0] * m.kernel_size[1] * o.shape[-2] * o.shape[-1])
            return f

        handles = [m.register_forward_hook(hook(n))
                   for n, m in net.named_modules() if isinstance(m, nn.Conv2d)]
        with torch.no_grad():
            net(torch.randn(1, 3, size, size))
        for h in handles:
            h.remove()
        total = sum(tally.values())
        enc = sum(v for k, v in tally.items() if k.startswith("encoder"))
        return total / 1e9, enc / 1e9

    torch.manual_seed(0)
    configs = [
        ("mobilenetv3", "dense", None),          # the original architecture
        ("mobilenetv3", "separable", None),
        ("mobilenetv4_m", "separable", None),
        ("efficientvit_b1", "separable", None),  # 4-tap encoder (no stride-2)
    ]
    base = None
    for enc_name, dec_name, lat in configs:
        net = build_model(num_classes=4, pretrained=False, use_gcg=True,
                          encoder=enc_name, decoder=dec_name, lateral_channels=lat).eval()
        n_params = sum(p.numel() for p in net.parameters())
        x = torch.randn(2, 3, 256, 256)
        y = net(x)
        assert y.shape == (2, 4, 256, 256), (enc_name, dec_name, y.shape)
        loss = F.binary_cross_entropy_with_logits(y, torch.rand_like(y))
        loss.backward()
        g, ge = macs_of(net)
        base = base or g
        print(f"{enc_name:16s} {dec_name:10s} params {n_params/1e6:5.2f}M  "
              f"{g:6.2f} GMAC @512 (enc {100*ge/g:4.1f}%)  {base/g:4.2f}x cheaper  backward OK")
    print("model.py self-check passed.")
