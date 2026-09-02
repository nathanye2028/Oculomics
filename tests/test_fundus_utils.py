import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fundus_utils import (fov_bbox, crop_to_fov, focal_tversky_loss, tversky_loss,
                          random_patch, make_rng, pick_device, tiled_predict,
                          seed_everything, seed_worker)


class _DrawDataset(Dataset):
    """Yields (index, first make_rng draw). Module-level on purpose: macOS
    DataLoader workers are spawned, so the class must be importable by name."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed = n, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return i, float(make_rng(self.seed, i).random())


def test_fov_bbox_and_crop():
    img = np.zeros((100, 120, 3), np.uint8)
    img[20:80, 30:90] = 200
    assert fov_bbox(img) == (20, 80, 30, 90)
    cropped, box = crop_to_fov(img)
    assert cropped.shape[:2] == (60, 60)


def test_seg_losses_backward_and_range():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = (torch.rand(2, 3, 16, 16) > 0.9).float()
    for fn in (focal_tversky_loss, tversky_loss):
        loss = fn(logits, target)
        assert 0.0 <= loss.item() <= 1.0 + 1e-4
        loss.backward()
        assert logits.grad is not None
        logits.grad = None


def test_random_patch_shape_and_lesion_bias():
    img = np.random.randint(0, 255, (200, 240, 3), np.uint8)
    masks = np.zeros((4, 200, 240), np.uint8)
    masks[0, 100, 120] = 1
    rng = np.random.default_rng(0)
    ip, mp = random_patch(img, masks, 64, rng, fg_bias=1.0)
    assert ip.shape == (64, 64, 3) and mp.shape == (4, 64, 64)
    assert mp.sum() > 0       # fg_bias=1.0 -> patch centred on the lesion


def test_make_rng_reproducible_yet_diverse():
    # Main-process semantics (num_workers=0): successive calls for the SAME
    # sample must differ (else every epoch replays identical augmentation),
    # but the whole sequence must replay after re-seeding — reproducibility
    # is per-run, anchored by seed_everything.
    seed_everything(42)
    a1 = make_rng(42, 3).random()
    a2 = make_rng(42, 3).random()      # "next epoch" fetch of the same sample
    c = make_rng(42, 4).random()
    assert a1 != a2                     # diverse across epochs
    assert a1 != c                      # diverse across samples

    seed_everything(42)                 # re-run: exact same sequence
    b1 = make_rng(42, 3).random()
    b2 = make_rng(42, 3).random()
    assert (a1, a2) == (b1, b2)


def _draws_per_index(seed: int, epochs: int = 3, n: int = 6):
    """{index: [draw per epoch]} through 2 PERSISTENT workers, as train_idrid
    builds its loader. Persistent workers keep torch.initial_seed() fixed for
    the whole run, which is exactly the case that used to collapse to one draw."""
    seed_everything(seed)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(_DrawDataset(n, seed), batch_size=1, shuffle=True,
                        num_workers=2, persistent_workers=True,
                        worker_init_fn=seed_worker, generator=g)
    out = {i: [] for i in range(n)}
    for _ in range(epochs):
        for idx, draw in loader:
            out[int(idx)].append(float(draw))
    return out


def test_make_rng_diverse_across_epochs_with_persistent_workers():
    draws = _draws_per_index(seed=7, epochs=3)
    for i, vals in draws.items():
        assert len(vals) == 3
        assert len(set(vals)) == 3, f"index {i}: persistent worker replayed {vals}"
    # Still reproducible: same seed, same shuffle -> same worker/index/draw order.
    assert _draws_per_index(seed=7, epochs=3) == draws


def test_pick_device_explicit_and_fallback():
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in ("cuda", "mps", "cpu")
    # DDP never lands on MPS (no backend) — only cuda:<rank> or cpu.
    assert pick_device(local_rank=0, ddp=True).type in ("cuda", "cpu")


def test_tiled_predict_shape():
    seed_everything(0)
    model = nn.Conv2d(3, 4, 1)       # trivial [3,h,w] -> [4,h,w]
    img = torch.randn(3, 300, 360)
    out = tiled_predict(model, img, tile=128, overlap=32, device=torch.device("cpu"), tile_batch=4)
    assert out.shape == (4, 300, 360)
    assert (out >= 0).all() and (out <= 1).all()   # sigmoid probabilities
