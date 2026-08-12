"""
model.py
========
Baseline 2D segmentation network for ocular-pathology (retinal-lesion)
segmentation, with injection points for **Guided Context Gating (GCG)** blocks.

Design constraints (per project spec)
-------------------------------------
* Pure image -> mask. **No systemic / demographic metadata fusion** — the
  network sees only the fundus image, so learned features are spatial-visual.
* Encoder: **MobileNetV3-Large** (torchvision, optionally ImageNet-pretrained),
  chosen for a lightweight, mobile-capable backbone.
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
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

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
# MobileNetV3-Large encoder (multi-scale feature taps)
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        end_set = set(self._stage_ends)
        for i, block in enumerate(self.features):
            x = block(x)
            if i in end_set:
                feats.append(x)
        return feats  # [s2, s4, s8, s16, s32]


# --------------------------------------------------------------------------- #
# Decoder block
# --------------------------------------------------------------------------- #
class _ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample the decoder feature, GCG-gate the skip, concat, then fuse."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        gcg_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.gcg = gcg_factory(skip_channels, in_channels) if gcg_factory else None
        self.fuse = nn.Sequential(
            _ConvBNAct(in_channels + skip_channels, out_channels),
            _ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gcg is not None:
            skip = self.gcg(skip, x)          # guided gating, guide = upsampled decoder feat
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# --------------------------------------------------------------------------- #
# U-Net with MobileNetV3 encoder + GCG-gated skips
# --------------------------------------------------------------------------- #
class GCGUNet(nn.Module):
    """2D U-Net (MobileNetV3-Large encoder) with GCG-gated skip connections.

    Parameters
    ----------
    num_classes : output channels of the per-pixel logits (1 for binary).
    pretrained  : load ImageNet weights into the encoder. **Defaults to True** —
                  see :class:`MobileNetV3Encoder` for why scratch training is
                  the opt-in case, not the default.
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
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        use_gcg: bool = True,
        gcg_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=pretrained, in_channels=in_channels)
        enc_ch = self.encoder.out_channels            # [16, 24, 40, 112, 960]

        if use_gcg:
            factory = gcg_factory or (lambda s, g: GuidedContextGating(s, g))
        else:
            factory = None

        dec = list(decoder_channels)
        # Decoder consumes encoder stages coarse->fine; skips are the 4 shallower stages.
        # x starts as the deepest feature (enc_ch[-1]).
        skips = [enc_ch[3], enc_ch[2], enc_ch[1], enc_ch[0]]   # s16, s8, s4, s2
        ins = [enc_ch[4], dec[0], dec[1], dec[2]]
        outs = [dec[0], dec[1], dec[2], dec[3]]
        self.decoders = nn.ModuleList([
            DecoderBlock(ins[i], skips[i], outs[i], gcg_factory=factory)
            for i in range(4)
        ])

        # Full-resolution path. The MobileNetV3 encoder's shallowest tap is at
        # stride 2 (H/2), so without this the decoder's finest learned features
        # are half-res and the head just bilinearly upsamples them -> tiny lesions
        # (microaneurysms, ~1-2 px) are unrecoverable. The stem is a stride-1
        # feature of the input; the final decoder stage upsamples to full res and
        # fuses it (gated like the others) so the head runs at full resolution.
        stem_ch = dec[4]
        self.stem = nn.Sequential(
            _ConvBNAct(in_channels, stem_ch),
            _ConvBNAct(stem_ch, stem_ch),
        )
        self.up_full = DecoderBlock(dec[3], stem_ch, dec[4], gcg_factory=factory)
        self.head = nn.Conv2d(dec[4], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        stem = self.stem(x)                       # full-resolution features [B, dec4, H, W]
        s2, s4, s8, s16, s32 = self.encoder(x)
        d = s32
        d = self.decoders[0](d, s16)
        d = self.decoders[1](d, s8)
        d = self.decoders[2](d, s4)
        d = self.decoders[3](d, s2)               # H/2
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
) -> nn.Module:
    """Factory. ``arch='gcg_unet'`` -> MobileNetV3-U-Net with GCG skips.

    ``pretrained`` defaults to True — see :class:`MobileNetV3Encoder`. Pass
    False only when a checkpoint is about to overwrite the encoder anyway
    (export/eval) or when scratch init is the point (ablation, overfit test).
    """
    if arch == "gcg_unet":
        return GCGUNet(
            num_classes=num_classes, pretrained=pretrained,
            use_gcg=use_gcg, gcg_factory=gcg_factory,
        )
    raise ValueError(f"Unknown arch {arch!r}")


if __name__ == "__main__":
    # Tiny self-check (no data needed): shapes + one backward pass.
    torch.manual_seed(0)
    net = build_model(num_classes=1, pretrained=False, use_gcg=True)
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(f"params: {n_params/1e6:.2f}M  input: {tuple(x.shape)}  output: {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), y.shape
    loss = F.binary_cross_entropy_with_logits(y, torch.rand_like(y))
    loss.backward()
    print(f"loss: {loss.item():.4f}  backward OK")
    print("model.py self-check passed.")
