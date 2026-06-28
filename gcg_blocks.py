"""
gcg_blocks.py
=============
Candidate **Guided Context Gating (GCG)** blocks + a registry, so the custom
gating mechanism can be swapped and benchmarked without touching the network.

Every block obeys the same contract expected by
``model_seg.build_model(gcg_factory=...)`` and ``DecoderBlock``:

    block = Factory(skip_channels, guide_channels)
    gated = block(skip, guide)        # gated.shape == skip.shape

`skip`  = high-res encoder feature being gated.
`guide` = coarser decoder feature (deep semantic context); may be None.

This file is a scaffold/template: drop your custom block in as a new class,
register it in ``GCG_VARIANTS``, and benchmark it with
``train_idrid.py --gcg-variant <name>`` against the control.

Variants provided:
  * ``attention`` — Attention U-Net additive attention gate (guide-conditioned).
  * ``cbam``      — channel (SE) + spatial attention, self-attention on skip.
  * ``se``        — channel-only squeeze-and-excite gate.
  * ``none``      — identity (no gating) control, for an in-architecture A/B.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match(guide: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    if guide.shape[-2:] != skip.shape[-2:]:
        guide = F.interpolate(guide, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return guide


class AttentionGate(nn.Module):
    """Attention U-Net additive attention gate, conditioned on the decoder guide."""

    def __init__(self, skip_channels: int, guide_channels: Optional[int] = None) -> None:
        super().__init__()
        inter = max(skip_channels // 2, 4)
        self.w_g = nn.Conv2d(guide_channels or skip_channels, inter, 1)
        self.w_x = nn.Conv2d(skip_channels, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, skip: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        g = _match(guide, skip) if guide is not None else skip
        att = torch.sigmoid(self.psi(F.relu(self.w_g(g) + self.w_x(skip), inplace=True)))
        return skip * att


class SEGate(nn.Module):
    """Channel-only squeeze-and-excite gate (self-attention on the skip)."""

    def __init__(self, skip_channels: int, guide_channels: Optional[int] = None, reduction: int = 8) -> None:
        super().__init__()
        inter = max(skip_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, inter, 1), nn.ReLU(inplace=True),
            nn.Conv2d(inter, skip_channels, 1),
        )

    def forward(self, skip: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return skip * torch.sigmoid(self.fc(skip))


class CBAMGate(nn.Module):
    """CBAM-style channel + spatial attention applied to the skip feature."""

    def __init__(self, skip_channels: int, guide_channels: Optional[int] = None, reduction: int = 8) -> None:
        super().__init__()
        inter = max(skip_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(skip_channels, inter, 1), nn.ReLU(inplace=True),
            nn.Conv2d(inter, skip_channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, skip: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        ch = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(skip, 1))
                           + self.mlp(F.adaptive_max_pool2d(skip, 1)))
        x = skip * ch
        sp = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)))
        return x * sp


class NoGate(nn.Module):
    """Identity passthrough — in-architecture 'no gating' control."""

    def __init__(self, skip_channels: int, guide_channels: Optional[int] = None) -> None:
        super().__init__()

    def forward(self, skip: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return skip


# name -> factory(skip_channels, guide_channels) -> nn.Module
GCG_VARIANTS: Dict[str, Callable[[int, Optional[int]], nn.Module]] = {
    "attention": lambda s, g: AttentionGate(s, g),
    "cbam": lambda s, g: CBAMGate(s, g),
    "se": lambda s, g: SEGate(s, g),
    "none": lambda s, g: NoGate(s, g),
}


if __name__ == "__main__":
    # Contract check: every variant maps (skip, guide) -> skip-shaped + backprops.
    skip = torch.randn(2, 40, 32, 32, requires_grad=True)
    guide = torch.randn(2, 112, 16, 16)
    for name, factory in GCG_VARIANTS.items():
        blk = factory(40, 112)
        out = blk(skip, guide)
        assert out.shape == skip.shape, (name, out.shape)
        out.mean().backward()
        skip.grad = None
        print(f"{name:>10}: out {tuple(out.shape)}  OK")
    print("gcg_blocks.py self-check passed.")
