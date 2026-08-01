"""Validity-masked loss: partial-annotation datasets must not inject false negatives."""
import torch

from fundus_utils import focal_tversky_loss, tversky_loss


def _setup():
    B, C, H, W = 2, 4, 8, 8
    logits = torch.zeros(B, C, H, W)      # p = 0.5 everywhere
    target = torch.zeros(B, C, H, W)
    target[:, 2] = 1.0                     # channel 2 (EX) positive
    return logits, target, B, C


def test_valid_none_matches_all_valid():
    logits, target, B, C = _setup()
    a = focal_tversky_loss(logits, target)
    b = focal_tversky_loss(logits, target, valid=torch.ones(B, C))
    assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-6)


def test_unannotated_channels_change_loss():
    logits, target, B, C = _setup()
    ex_only = torch.zeros(B, C); ex_only[:, 2] = 1.0
    assert abs(focal_tversky_loss(logits, target).item()
               - focal_tversky_loss(logits, target, valid=ex_only).item()) > 1e-4


def test_no_gradient_to_unannotated_channels():
    _, target, B, C = _setup()
    logits = torch.randn(B, C, 8, 8, requires_grad=True)
    ex_only = torch.zeros(B, C); ex_only[:, 2] = 1.0
    focal_tversky_loss(logits, target, valid=ex_only).backward()
    assert logits.grad[:, 2].abs().sum() > 0        # annotated -> learns
    for c in (0, 1, 3):
        assert logits.grad[:, c].abs().sum() == 0   # unannotated -> ignored


def test_tversky_also_supports_valid():
    logits, target, B, C = _setup()
    ex_only = torch.zeros(B, C); ex_only[:, 2] = 1.0
    v = tversky_loss(logits, target, valid=ex_only)
    assert torch.isfinite(torch.tensor(v))


def test_mixed_batch_partial_annotation():
    """One IDRiD-like row (all valid) + one e-ophtha-like row (EX only)."""
    logits, target, B, C = _setup()
    valid = torch.tensor([[1., 1., 1., 1.], [0., 0., 1., 0.]])
    loss = focal_tversky_loss(logits, target, valid=valid)
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
