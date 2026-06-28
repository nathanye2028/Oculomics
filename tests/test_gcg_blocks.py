import pytest
import torch

from gcg_blocks import GCG_VARIANTS


@pytest.mark.parametrize("name", list(GCG_VARIANTS))
def test_variant_contract(name):
    """Every GCG variant: forward(skip, guide) -> skip-shaped tensor + backprops."""
    skip = torch.randn(2, 40, 32, 32, requires_grad=True)
    guide = torch.randn(2, 112, 16, 16)
    block = GCG_VARIANTS[name](40, 112)
    out = block(skip, guide)
    assert out.shape == skip.shape
    out.mean().backward()
    assert skip.grad is not None


@pytest.mark.parametrize("name", list(GCG_VARIANTS))
def test_variant_unguided(name):
    """Blocks must also work with guide=None (unguided)."""
    skip = torch.randn(1, 24, 24, 24)
    out = GCG_VARIANTS[name](24, None)(skip, None)
    assert out.shape == skip.shape
