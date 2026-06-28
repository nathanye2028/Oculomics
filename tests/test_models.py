import torch
import torch.nn.functional as F

from model import build_classifier
from model_seg import build_model
from unet_baseline import build_baseline


def test_classifier_heads():
    x = torch.randn(2, 3, 128, 128)
    m = build_classifier(num_classes=5, pretrained=False)
    y = m(x); assert y.shape == (2, 5)
    F.cross_entropy(y, torch.tensor([0, 3])).backward()
    assert build_classifier(num_classes=4, multilabel=True, pretrained=False)(x).shape == (2, 4)
    assert build_classifier(regression=True, pretrained=False)(x).shape == (2,)


def test_gcg_unet_forward_backward():
    x = torch.randn(2, 3, 128, 128)
    for use_gcg in (True, False):
        m = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=use_gcg)
        y = m(x)
        assert y.shape == (2, 4, 128, 128)
        F.binary_cross_entropy_with_logits(y, torch.rand_like(y)).backward()


def test_gcg_unet_accepts_variant_factory():
    from gcg_blocks import GCG_VARIANTS
    m = build_model(arch="gcg_unet", num_classes=4, pretrained=False,
                    use_gcg=True, gcg_factory=GCG_VARIANTS["attention"])
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 4, 96, 96)


def test_baseline_unet_forward():
    m = build_baseline(num_classes=1, base=16)
    y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1, 128, 128)
