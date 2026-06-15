"""
model.py
========
Lightweight clinical-label **classifier** for the mBRSET fundus pipeline.

A pretrained, mobile-friendly **MobileNetV3-Small** backbone with a **custom
classification head** that predicts the clinical labels produced by
:mod:`dataset` (``MBRSETDataset`` / ``LABEL_REGISTRY``):

  * multiclass  -> ``num_classes`` logits  (e.g. ``dr_grade``: 5)   [CrossEntropy]
  * binary      -> 2 logits                (e.g. ``dr_referable``)  [CrossEntropy]
  * multilabel  -> ``num_classes`` logits  (sigmoid / BCE)
  * regression  -> 1 output                (e.g. ``age``)           [MSE/L1]

Design
------
* Image-only — **no systemic / demographic metadata fusion** (matches the
  segmentation baseline in :mod:`model_seg`).
* MobileNetV3-Small is small (~2.5M backbone params) and mobile-capable,
  suiting the smartphone-captured mBRSET domain.
* The stock classifier is discarded and replaced by a custom head
  (pool -> dropout -> Linear -> Hardswish -> dropout -> Linear) sized to the
  task. Use :meth:`MBRSETClassifier.from_dataset` to build a head straight
  from a dataset instance's ``num_classes`` / ``multilabel`` flags.

Typical use
-----------
    from dataset import MBRSETDataset
    from model import MBRSETClassifier

    ds = MBRSETDataset(csv=..., images_dir=..., task="dr_grade")
    model = MBRSETClassifier.from_dataset(ds, pretrained=True)
    logits = model(images)          # [B, 5]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MBRSETClassifier(nn.Module):
    """MobileNetV3-Small backbone + custom clinical-label head.

    Parameters
    ----------
    num_classes : output classes. ``None`` (or 1) with ``regression=True``
                  yields a single linear output.
    multilabel  : emit ``num_classes`` independent logits for sigmoid/BCE.
    regression  : single continuous output (overrides classification).
    pretrained  : load ImageNet weights into the backbone.
    in_channels : input channels (3 for RGB).
    dropout     : dropout prob in the head.
    head_hidden : hidden width of the head MLP (default 256).
    freeze_backbone : train only the head (linear-probe).
    """

    def __init__(
        self,
        num_classes: Optional[int] = 5,
        multilabel: bool = False,
        regression: bool = False,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 256,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        weights = None
        if pretrained:
            from torchvision.models import MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)

        # Adapt the stem for non-RGB input if requested.
        if in_channels != 3:
            stem = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels, stem.out_channels, kernel_size=stem.kernel_size,
                stride=stem.stride, padding=stem.padding, bias=False,
            )

        self.features = backbone.features          # conv stages
        self.pool = backbone.avgpool               # -> [B, 576, 1, 1]
        feat_dim = backbone.classifier[0].in_features  # 576 for v3-small

        self.regression = regression
        self.multilabel = multilabel
        if regression:
            out_dim = 1
        else:
            if num_classes is None:
                raise ValueError("num_classes required for classification (or set regression=True).")
            out_dim = int(num_classes)
        self.num_classes = None if regression else out_dim

        # Custom classification head.
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, head_hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, out_dim),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.head(x)
        if self.regression:
            return logits.squeeze(-1)     # [B]
        return logits                     # [B, num_classes]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataset(cls, dataset, **kwargs) -> "MBRSETClassifier":
        """Build a head matching a ``MBRSETDataset`` instance's label type."""
        num_classes = getattr(dataset, "num_classes", None)
        multilabel = getattr(dataset, "multilabel", False)
        regression = num_classes is None and not multilabel
        return cls(
            num_classes=num_classes if not regression else None,
            multilabel=multilabel,
            regression=regression,
            **kwargs,
        )


def build_classifier(
    num_classes: Optional[int] = 5,
    multilabel: bool = False,
    regression: bool = False,
    pretrained: bool = True,
    **kwargs,
) -> MBRSETClassifier:
    """Factory mirroring :class:`MBRSETClassifier`."""
    return MBRSETClassifier(
        num_classes=num_classes, multilabel=multilabel,
        regression=regression, pretrained=pretrained, **kwargs,
    )


if __name__ == "__main__":
    # Self-check (no data): shapes for each label type + one backward pass.
    import torch.nn.functional as F

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    m = build_classifier(num_classes=5, pretrained=False)          # multiclass
    y = m(x)
    print(f"multiclass : params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  out={tuple(y.shape)}")
    assert y.shape == (2, 5)
    F.cross_entropy(y, torch.tensor([0, 3])).backward()

    mb = build_classifier(num_classes=4, multilabel=True, pretrained=False)
    yb = mb(x)
    print(f"multilabel : out={tuple(yb.shape)}")
    assert yb.shape == (2, 4)

    mr = build_classifier(regression=True, pretrained=False)        # age
    yr = mr(x)
    print(f"regression : out={tuple(yr.shape)}")
    assert yr.shape == (2,)

    print("model.py self-check passed.")
