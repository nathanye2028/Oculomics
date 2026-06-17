"""
unet_baseline.py
================
A clean, **gating-free** 2D U-Net for segmenting abnormalities on fundus
images — the canonical Ronneberger-style encoder/decoder with plain
double-conv blocks and skip connections, and **no attention of any kind**.

What this is (and is NOT)
-------------------------
This is a **standard-architecture reference point**: "how does a textbook
U-Net do on this task?". It is image-only (no metadata fusion) and shares the
input/output contract (image ``[B,3,H,W]`` -> logits ``[B,num_classes,H,W]``).

It is **NOT the control for measuring Guided Context Gating.** This U-Net
differs from the GCG model in backbone, depth and pretraining, so a Dice gap
between them confounds "gating" with "architecture". The apples-to-apples GCG
ablation is the *same* network with gating toggled:

    model_seg.build_model(use_gcg=True)   vs   build_model(use_gcg=False)
    # i.e.  train_idrid.py --arch gcg_unet   vs   --arch gcg_unet --no-gcg

Use *that* pair to attribute an effect to gating; use this module only as an
independent literature baseline alongside both.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2 — the standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 (maxpool) then double-conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upscale by 2, concatenate the encoder skip, then double-conv.

    No gating on the skip — it is concatenated directly. This is the exact
    point where a Context Gating block would later wrap ``skip``.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out = in_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out = in_ch // 2
        self.conv = DoubleConv(up_out + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd input sizes made shapes drift.
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)        # <-- plain skip; GCG would gate here
        return self.conv(x)


class UNet(nn.Module):
    """Baseline 2D U-Net.

    Parameters
    ----------
    in_channels : input image channels (3 for RGB fundus).
    num_classes : output channels (1 for binary abnormality vs background).
    base : channel width of the first stage; doubles each downsample.
    depth : number of downsampling stages.
    bilinear : bilinear upsample (True) vs transposed conv (False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base: int = 64,
        depth: int = 4,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        chs: Sequence[int] = [base * (2 ** i) for i in range(depth + 1)]  # e.g. [64,128,256,512,1024]

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [Down(chs[i], chs[i + 1]) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [Up(chs[i + 1], chs[i], chs[i], bilinear=bilinear) for i in reversed(range(depth))]
        )
        self.outc = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        out = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            out = up(out, skip)
        return self.outc(out)


def build_baseline(num_classes: int = 1, in_channels: int = 3, base: int = 64, **kwargs) -> UNet:
    """Factory for the baseline U-Net benchmark."""
    return UNet(in_channels=in_channels, num_classes=num_classes, base=base, **kwargs)


if __name__ == "__main__":
    # Self-check (no data): shapes + one backward pass.
    torch.manual_seed(0)
    net = build_baseline(num_classes=1, base=32)   # base=32 keeps the self-check light
    n = sum(p.numel() for p in net.parameters())
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(f"params: {n/1e6:.2f}M  input: {tuple(x.shape)}  output: {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), y.shape
    F.binary_cross_entropy_with_logits(y, torch.rand_like(y)).backward()
    print("unet_baseline.py self-check passed.")
