import pytest
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


# --------------------------------------------------------------------------- #
# Mobile decoder / modern encoders
# --------------------------------------------------------------------------- #
def _conv_macs(model, size=128):
    """Total conv MAC for one forward pass, via hooks."""
    import torch.nn as nn
    total = []

    def hook(m, _i, o):
        total.append(m.in_channels // m.groups * m.out_channels
                     * m.kernel_size[0] * m.kernel_size[1] * o.shape[-2] * o.shape[-1])

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(torch.randn(1, 3, size, size))
    for h in handles:
        h.remove()
    return sum(total)


def test_default_build_is_the_original_architecture():
    """Regression guard: the default build must stay loadable by old checkpoints.

    Every result in checkpoints/ and exp_*/ was trained with the dense decoder and
    no lateral projection. If a default-valued knob ever silently flips, those
    checkpoints stop loading and prior results become unreproducible.
    """
    m = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=True)
    keys = set(m.state_dict())
    assert not any(k.startswith("lateral") for k in keys), "lateral proj must be OFF by default"
    assert not any(k.startswith("mid_ups") for k in keys), "no skip-less stages for a stride-2 encoder"
    assert len([k for k in keys if k.startswith("decoders.")]) > 0
    # decoder fuse convs must still be dense 3x3 (groups=1), not depthwise
    w = m.decoders[0].fuse[0][0]
    assert w.kernel_size == (3, 3) and w.groups == 1


def test_separable_decoder_cuts_macs_and_trains():
    """Guards the MAC/param reduction only.

    Deliberately NOT a latency claim: measured Core ML latency is 9.2 ms vs the
    dense decoder's 9.0 ms at 512x512 (bandwidth-bound). See model_seg's module
    docstring.
    """
    dense = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=True)
    sep = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=True,
                      decoder="separable")
    y = sep(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 4, 128, 128)
    F.binary_cross_entropy_with_logits(y, torch.rand_like(y)).backward()

    # depthwise fuse convs, and the deep 1x1 projection is on by default here
    assert any(k.startswith("lateral") for k in sep.state_dict())
    first = sep.decoders[0].fuse[0][0]
    assert first.groups == first.in_channels and first.kernel_size == (3, 3)

    ratio = _conv_macs(dense) / _conv_macs(sep)     # 4.33x at 512x512
    assert ratio > 3.0, f"expected >3x MAC reduction, got {ratio:.2f}x"
    assert sum(p.numel() for p in sep.parameters()) < sum(p.numel() for p in dense.parameters())


def test_lateral_projection_is_independently_ablatable():
    """separable blocks without the projection, so the two can be measured apart."""
    m = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=True,
                    decoder="separable", lateral_channels=0)
    assert not any(k.startswith("lateral") for k in m.state_dict())
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 4, 96, 96)


def test_timm_encoders_including_the_4_tap_case():
    """MobileNetV4 has 5 taps (strides 2..32); EfficientViT has only 4 (4..32).

    The 4-tap case is the one that needs the skip-less octave, so it is the case
    most likely to break — cover it explicitly.
    """
    timm = pytest.importorskip("timm")
    assert timm  # silence linters
    for enc, expect_mid in (("mobilenetv4_s", False), ("efficientvit_b1", True)):
        m = build_model(arch="gcg_unet", num_classes=4, pretrained=False,
                        use_gcg=True, encoder=enc, decoder="separable")
        y = m(torch.randn(1, 3, 128, 128))
        assert y.shape == (1, 4, 128, 128), enc
        F.binary_cross_entropy_with_logits(y, torch.rand_like(y)).backward()
        has_mid = any(k.startswith("mid_ups") for k in m.state_dict())
        assert has_mid is expect_mid, f"{enc}: mid_ups={has_mid}, expected {expect_mid}"


def test_arch_cfg_from_checkpoint_round_trip():
    from model_seg import arch_cfg_from_checkpoint

    # a pre-flag checkpoint must fall back to the original architecture
    assert arch_cfg_from_checkpoint({"model": {}}) == {
        "encoder": "mobilenetv3", "decoder": "dense", "lateral_channels": None}
    # the -1 sentinel means "decide from decoder", i.e. None at the model layer
    sentinel = arch_cfg_from_checkpoint(
        {"args": {"encoder": "mobilenetv4_m", "decoder": "separable",
                  "lateral_channels": -1}})
    assert sentinel["lateral_channels"] is None
    assert sentinel["encoder"] == "mobilenetv4_m"
    cfg = arch_cfg_from_checkpoint(
        {"args": {"encoder": "efficientvit_b1", "decoder": "separable",
                  "lateral_channels": 128}})
    assert cfg == {"encoder": "efficientvit_b1", "decoder": "separable",
                   "lateral_channels": 128}
